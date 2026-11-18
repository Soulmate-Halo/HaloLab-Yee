"""CPU 侧按请求自适应门槛：DSV41_VERIFY_CAP=conf:T 的 adapt 模式（只依赖 numpy 和标准库）。

r653 主线接手编写（器灵 W1 未交付）。由 verify_cap.py 在 rank0 调用：
  prepare_step(rids, temps, now, need)  每个验证步开头：登记请求；need=True 时给出本步每请求门槛
  observe(rids, k, cum, is_rec, acc, t_prep)  某一步的结果到 CPU 后（overlap 下晚 1~2 步）
门槛 thr[r, j] 作用在原始累乘置信度 cum[r, j] 上：cum >= thr 的前导位置才验证。

六条逻辑（cfg["logics"] 可逐条关掉）：
  load    lam = clip(delta_ms * theta, lam_min, lam_max)，theta = 整批每毫秒产出的 token（EMA）。
          多验证一行要花 delta_ms，整批每毫秒产出 theta 个 token，所以只有接受概率 >= lam 的位置才值得验证。
  req     每请求校准：校准概率 c_hat_j = S_j * cum_j ** beta_r；用已验证位置的期望接受数和实际接受数做矩匹配更新 beta_r。
  pos     全局按位置校准 s_j（第 6 位以后超出草稿训练长度，置信度可能偏）：观测接受 / 预测接受，S_j = prod_{i<=j} s_i。
  recall  召回行（整行置信度为 1）每请求逐位接受率 q_r（Beta 计数），krec = floor(ln lam / ln q_r)。
  temp    温度档先验：请求第一次出现时 beta_r、q_r 的先验取所在温度档的均值（每个请求算一次），之后按自己的观测调整。
  explore 每请求每步以概率 explore 多验证 explore_add 个位置，避免校准只看到门槛以内。
门槛：thr[r, j] = clip((lam / S_j) ** (1 / beta_r), thr_min, thr_max)。全部关掉时 thr = cfg["thr"]、krec = kmax、kadd = 0，与 legacy 等价。

r703（Yee 3.8.0）选宽度器影子模式，只算不用。控制文件写 {"mode": "width"}（或 "width": {"enabled": true}，
后者随配置合并保留，要关得显式写 false）时开启。verify_cap.py 把未知 mode 当 legacy，所以引擎每步的 k 仍与
conf:T 相同；宽度代码不碰门槛、不用随机数，adapt 模式的输出与关掉宽度时逐位相同。
  prepare_step  只用此前已观测的信息（相当于起草前）给每个候选验证宽度 W（每请求验证行数 = 块长 + 1，
                现网 W0 = P + 1）算 E(W) = sum_r (1 + 期望接受数)，T(W) = W0 实测步时 + 行数变化 × 每行成本，
                选 E/T 最大的宽度（比当前宽度好 switch 以上才换）；另有只在 W <= W0 里选的 le0 选择器。
  observe       用本步实际的 acc / k 结算"假如本步用 W"的 token 数和步时（W <= W0 精确，同一份草稿截短；
                W > W0 越过当前块长的部分按延续率估：召回行用该请求 q_r，模型行用全局 rho_acc * rho_k），
                按 bs 档累计，stats()["width"] 输出窗口和：act（实际）、sel、sel_lo（W > W0 时不计估算部分的
                保守下限）、le0、oracle（每步事后最优）、static[W]（固定宽度）、pick、pred（预测与结算 token）。
每行成本 width.cost 默认取 r650 块长 3/5/7 实测斜率（含起草开销变化），某 bs 档还没有 W0 实测步时 T(W0)
用 r653 回归 width.base。宽度代码出错只关掉宽度影子（stats 里出现 width_error），不影响门槛策略。
"""
import collections
import math
import time

import numpy as np

DEFAULTS = {
    "mode": "legacy",
    "tag": "",
    "delta_ms": 0.87,
    "thr": 0.1,
    "thr_min": 0.02,
    "thr_max": 0.6,
    "lam_min": 0.02,
    "lam_max": 0.6,
    "kmin": 1,
    "kmax": None,
    "explore": 0.03,
    "explore_add": 2,
    "ema": 0.05,
    "logics": {"load": True, "req": True, "pos": True, "recall": True, "temp": True},
    "stats_s": 10.0,
    "width": {},
}
LOGICS = ("load", "req", "pos", "recall", "temp")
N_BUCKETS = 4
GC_STEPS = 64
BETA_LR = 0.2
BETA_MIN, BETA_MAX = 0.25, 4.0
S_MIN, S_MAX = 0.25, 4.0
S_PRIOR = 5.0
S_DECAY = 0.999
Q_MIN, Q_MAX = 0.05, 0.995
Q_PRIOR = 4.0
Q_DECAY = 0.98
Q0 = 0.85
BUCKET_LR = 0.01
MAX_STEP_GAP_MS = 1500.0

# r703 选宽度器（影子）默认值；cost 为 r650 块长 3/5/7 实测：每请求每多验证一行的毫秒数（copy≈活行，new≈死行）。
# 13 不在默认候选里：r703 用 r653 的压测矩阵对照 r653 实测的块长 12，W > W0 的估算偏高 5～30 个百分点
# （越过块长后的延续率与多出来的死行成本都估低了），要看就在 width.widths 里加上，数字只作参考。
WIDTH_DEFAULTS = {
    "enabled": False,
    "widths": [4, 8],
    "max_bs": {"13": 4},
    "switch": 0.03,
    "hist": 16,
    "glob_hist": 256,
    "prior_n": 2.0,
    "t_window": 32,
    "cost": {"bs": [1, 2, 4, 8], "live": [2.8, 2.2, 1.67, 1.29], "dead": [0.58, 0.63, 0.55, 0.52]},
    "hi_scale": 1.0,
    "base": {"a": 31.6, "b": 2.70, "live": 1.33, "dead": 0.50},
    "rho_prior": [0.5, 4.0],
    "rho_decay": 0.999,
}
BS_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def bucket_of(temp):
    t = float(temp or 0.0)
    if t <= 1e-5:
        return 0
    if t <= 0.5:
        return 1
    if t <= 1.0:
        return 2
    return 3


def merge_cfg(base, upd):
    out = dict(base)
    out["logics"] = dict(base.get("logics") or DEFAULTS["logics"])
    out["width"] = dict(base.get("width") or {})
    for k, v in (upd or {}).items():
        if k == "logics" and isinstance(v, dict):
            for lk, lv in v.items():
                if lk in LOGICS:
                    out["logics"][lk] = bool(lv)
        elif k == "width":
            if isinstance(v, dict):
                out["width"].update(v)
        elif k in DEFAULTS:
            out[k] = v
    for k in ("delta_ms", "thr", "thr_min", "thr_max", "lam_min", "lam_max", "explore", "ema", "stats_s"):
        out[k] = float(out[k])
    out["kmin"] = int(out["kmin"])
    out["explore_add"] = int(out["explore_add"])
    out["kmax"] = None if out.get("kmax") in (None, "", 0) else int(out["kmax"])
    return out


# ------------------------------------------------------------------ r703 选宽度器（影子）
def width_on(cfg):
    w = cfg.get("width")
    return cfg.get("mode") == "width" or (isinstance(w, dict) and bool(w.get("enabled")))


def width_cfg(raw):
    """规范化 width 子配置（缺的取 WIDTH_DEFAULTS），配置不合法时抛 ValueError。"""
    c = dict(WIDTH_DEFAULTS)
    for k, v in (raw or {}).items():
        if k in WIDTH_DEFAULTS:
            c[k] = v
    if not isinstance(c["widths"], (list, tuple)):
        raise ValueError("width.widths must be a list")
    widths = sorted(set(int(w) for w in c["widths"]))
    if not widths or widths[0] < 2:
        raise ValueError("width.widths must be >= 2")
    cost = dict(WIDTH_DEFAULTS["cost"])
    cost.update(dict(c["cost"] or {}))
    bsl = [int(x) for x in cost["bs"]]
    live = [float(x) for x in cost["live"]]
    dead = [float(x) for x in cost["dead"]]
    if not bsl or not (len(bsl) == len(live) == len(dead)) or min(bsl) < 1:
        raise ValueError("width.cost: bs / live / dead must have the same length")
    order = sorted(range(len(bsl)), key=lambda i: bsl[i])
    base = dict(WIDTH_DEFAULTS["base"])
    base.update(dict(c["base"] or {}))
    rp = list(c["rho_prior"])
    return {
        "widths": widths,
        "max_bs": {int(k): int(v) for k, v in dict(c["max_bs"] or {}).items()},
        "switch": float(c["switch"]),
        "hist": max(1, int(c["hist"])),
        "glob_hist": max(1, int(c["glob_hist"])),
        "prior_n": max(0.0, float(c["prior_n"])),
        "t_window": max(1, int(c["t_window"])),
        "cost": {"bs": [bsl[i] for i in order], "live": [live[i] for i in order], "dead": [dead[i] for i in order]},
        "hi_scale": float(c["hi_scale"]),
        "base": {k: float(base[k]) for k in ("a", "b", "live", "dead")},
        "rho_prior": [min(max(float(rp[0]), 0.0), 1.0), max(float(rp[1]), 1e-6)],
        "rho_decay": min(max(float(c["rho_decay"]), 0.0), 1.0),
    }


def bs_bucket(bs):
    for b in BS_BUCKETS:
        if bs <= b:
            return b
    return BS_BUCKETS[-1]


def geo_sum(rho, m):
    """sum_{i=1..m} rho**i：越过当前块长后连续再接受的期望个数。"""
    if m <= 0 or rho <= 0.0:
        return 0.0
    if rho >= 1.0 - 1e-12:
        return float(m)
    return rho * (1.0 - rho ** m) / (1.0 - rho)


def pick_width(scores, allowed, cur, switch):
    """allowed 里得分最高的下标；当前宽度可选且最高分不超过它 (1 + switch) 倍时保持不变。"""
    best = allowed[0]
    for j in allowed[1:]:
        if scores[j] > scores[best]:
            best = j
    if cur in allowed and scores[best] <= scores[cur] * (1.0 + switch):
        return cur
    return best


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span <= 0:
                return ys[i + 1]
            return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / span
    return ys[-1]


def _median(q):
    s = sorted(q)
    n = len(s)
    if n % 2:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _r2(v):
    return [round(float(v[0]), 3), round(float(v[1]), 3)]


class _WRing(object):
    """最近 cap 个观测 (acc, k, rec) 的环，增量维护各宽度要用的和。"""
    __slots__ = ("buf", "cap", "sa", "sk", "sma", "smk", "nf_rec", "nf_mod", "nk_rec", "nk_mod", "last")

    def __init__(self, cap, nlo):
        self.buf = collections.deque()
        self.cap = cap
        self.sa = self.sk = 0
        self.sma = [0] * nlo
        self.smk = [0] * nlo
        self.nf_rec = self.nf_mod = self.nk_rec = self.nk_mod = 0
        self.last = 0

    def _add(self, a, k, rec, s, lo_m1, P):
        self.sa += s * a
        self.sk += s * k
        sma, smk = self.sma, self.smk
        for i, wm in enumerate(lo_m1):
            sma[i] += s * (a if a < wm else wm)
            smk[i] += s * (k if k < wm else wm)
        if a == P:
            if rec:
                self.nf_rec += s
            else:
                self.nf_mod += s
        if k == P:
            if rec:
                self.nk_rec += s
            else:
                self.nk_mod += s

    def push(self, a, k, rec, lo_m1, P):
        self.buf.append((a, k, rec))
        self._add(a, k, rec, 1, lo_m1, P)
        if len(self.buf) > self.cap:
            oa, ok, orec = self.buf.popleft()
            self._add(oa, ok, orec, -1, lo_m1, P)


class _WidthShadow(object):
    def __init__(self, P, c):
        self.c = c
        self.P = int(P)
        self.W0 = self.P + 1
        self.widths = sorted(set(c["widths"]) | {self.W0})
        self.nlo = sum(1 for w in self.widths if w <= self.W0)
        self.j0 = self.nlo - 1
        self.lo_m1 = [w - 1 for w in self.widths[:self.nlo]]
        self.hi = [(j, self.widths[j] - 1 - self.P) for j in range(self.nlo, len(self.widths))]
        self.max_bs = dict(c["max_bs"])
        self.req = {}
        self.glob = _WRing(c["glob_hist"], self.nlo)
        self.rk = [0.0, 0.0]          # 非召回行：k >= P-1 时 k == P 的（衰减）次数 / 总数
        self.ra = [0.0, 0.0]          # 非召回行：acc >= P-1 且 k == P 时 acc == P 的次数 / 总数
        self.tq = {}
        self.dec = collections.OrderedDict()
        self.pend = collections.deque()
        self.cur = self.cur_le0 = self.j0
        self.nprep = 0
        self._coef = {}
        self._allowed = {}
        self._new_window()

    def _new_window(self):
        self.win = {"steps": 0, "timed": 0, "untimed": 0, "missing": 0, "switches": 0, "switches_le0": 0,
                    "us": 0.0, "us_n": 0, "by_bs": {}}

    def _new_bucket(self):
        ws = self.widths
        return {"n": 0, "act": [0.0, 0.0], "sel": [0.0, 0.0], "sel_lo": [0.0, 0.0], "le0": [0.0, 0.0],
                "oracle": [0.0, 0.0], "static": {w: [0.0, 0.0] for w in ws}, "pick": {w: 0 for w in ws},
                "pick_le0": {w: 0 for w in ws[:self.nlo]}, "pred": [0.0, 0.0]}

    def allowed(self, bs):
        a = self._allowed.get(bs)
        if a is None:
            a = [j for j, w in enumerate(self.widths) if j == self.j0 or bs <= self.max_bs.get(w, 1 << 30)]
            self._allowed[bs] = a
        return a

    def coef(self, bs):
        v = self._coef.get(bs)
        if v is None:
            cost = self.c["cost"]
            xs = [math.log2(b) for b in cost["bs"]]
            x = math.log2(max(bs, 1))
            v = (_interp(x, xs, cost["live"]), _interp(x, xs, cost["dead"]))
            self._coef[bs] = v
        return v

    def rho(self):
        pm, ps = self.c["rho_prior"]
        ra = (self.ra[0] + pm * ps) / (self.ra[1] + ps)
        rk = (self.rk[0] + pm * ps) / (self.rk[1] + ps)
        return ra, rk

    def base_ms(self, bs, L0, D0):
        q = self.tq.get(bs_bucket(bs))
        if q:
            return _median(q) if len(q) >= 3 else float(min(q))
        b = self.c["base"]
        return b["a"] + b["b"] * bs + b["live"] * L0 + b["dead"] * D0

    # -------------------------------------------------------- 预测
    def predict(self, rids, qs):
        """每个宽度的 E = sum_r (1 + 期望接受数)、L = sum_r (1 + 期望验证草稿数)。"""
        ws, nlo, hi = self.widths, self.nlo, self.hi
        nw = len(ws)
        ra, rk = self.rho()
        rm = ra * rk
        gm = {j: geo_sum(rm, m) for j, m in hi}
        gk = {j: geo_sum(rk, m) for j, m in hi}
        g = self.glob
        ng = len(g.buf)
        pn = self.c["prior_n"]
        glo, ghi = None, None
        if ng:
            glo = [(g.sma[j] / ng, g.smk[j] / ng) for j in range(nlo)]
            ghi = {j: ((g.sa + g.nf_mod * gm[j]) / ng, g.nf_rec / ng, (g.sk + g.nk_rec * m + g.nk_mod * gk[j]) / ng)
                   for j, m in hi}
        E = [0.0] * nw
        L = [0.0] * nw
        req = self.req
        for i, rid in enumerate(rids):
            ring = req.get(rid)
            n = len(ring.buf) if ring is not None else 0
            if n and ng:
                wr = n / (n + pn)
                wg = 1.0 - wr
            elif n:
                wr, wg = 1.0, 0.0
            elif ng:
                wr, wg = 0.0, 1.0
            else:
                for j in range(nw):
                    E[j] += 1.0
                    L[j] += 1.0 + min(1, ws[j] - 1)
                continue
            for j in range(nlo):
                a = k = 0.0
                if wr:
                    a = wr * ring.sma[j] / n
                    k = wr * ring.smk[j] / n
                if wg:
                    a += wg * glo[j][0]
                    k += wg * glo[j][1]
                E[j] += 1.0 + a
                L[j] += 1.0 + k
            if hi:
                q = qs[i]
                for j, m in hi:
                    gq = geo_sum(q, m)
                    a = k = 0.0
                    if wr:
                        a = wr * (ring.sa + ring.nf_rec * gq + ring.nf_mod * gm[j]) / n
                        k = wr * (ring.sk + ring.nk_rec * m + ring.nk_mod * gk[j]) / n
                    if wg:
                        ga, grec, gkk = ghi[j]
                        a += wg * (ga + grec * gq)
                        k += wg * gkk
                    E[j] += 1.0 + a
                    L[j] += 1.0 + k
        return E, L

    def scores(self, rids, qs):
        bs = len(rids)
        E, L = self.predict(rids, qs)
        ws, j0 = self.widths, self.j0
        D = [bs * w - L[j] for j, w in enumerate(ws)]
        base = self.base_ms(bs, L[j0], D[j0])
        cl, cd = self.coef(bs)
        hs = self.c["hi_scale"]
        T = []
        for j, w in enumerate(ws):
            s = hs if w > self.W0 else 1.0
            T.append(max(base + s * (cl * (L[j] - L[j0]) + cd * (D[j] - D[j0])), 0.25 * base))
        S = [E[j] / T[j] if T[j] > 0 else 0.0 for j in range(len(ws))]
        return {"E": E, "L": L, "D": D, "T": T, "score": S, "base": base}

    def decide(self, rids, qs, now):
        self._settle_pending(now)
        self.nprep += 1
        nprep = self.nprep
        req = self.req
        for rid in rids:
            ring = req.get(rid)
            if ring is not None:
                ring.last = nprep
        if nprep % 16 == 0:
            for rid in [r for r, ring in req.items() if nprep - ring.last > GC_STEPS]:
                del req[rid]
        bs = len(rids)
        if bs == 0:
            return None
        sc = self.scores(rids, qs)
        S = sc["score"]
        allowed = self.allowed(bs)
        sw = self.c["switch"]
        j = pick_width(S, allowed, self.cur, sw)
        if j != self.cur:
            self.win["switches"] += 1
            self.cur = j
        jl = pick_width(S, [x for x in allowed if x < self.nlo], self.cur_le0, sw)
        if jl != self.cur_le0:
            self.win["switches_le0"] += 1
            self.cur_le0 = jl
        self.dec[now] = (j, jl, sc["E"][j])
        while len(self.dec) > 16:
            self.dec.popitem(last=False)
        return sc

    # -------------------------------------------------------- 观测与结算
    def observe(self, rids, k, acc, is_rec, qs, t_prep, dt_ms):
        bs = len(rids)
        if bs == 0:
            return
        A = [int(x) for x in acc[:bs]]
        K = [int(x) for x in k[:bs]]
        R = [bool(x) for x in is_rec[:bs]]
        self.win["steps"] += 1
        item = (t_prep, A, K, R, list(qs), self.rho())
        if dt_ms is None:                       # 下一步还没开头：等下一次 decide 补上步时
            self.pend.append(item)
            while len(self.pend) > 16:
                old = self.pend.popleft()
                self.dec.pop(old[0], None)
                self.win["untimed"] += 1
        else:
            self._settle(item, dt_ms)
        P, lo_m1, nlo, d = self.P, self.lo_m1, self.nlo, self.c["rho_decay"]
        hist = self.c["hist"]
        req, glob, rk, ra = self.req, self.glob, self.rk, self.ra
        for i, rid in enumerate(rids):
            a, kk, rec = A[i], K[i], R[i]
            ring = req.get(rid)
            if ring is None:
                ring = req[rid] = _WRing(hist, nlo)
                ring.last = self.nprep
            ring.push(a, kk, rec, lo_m1, P)
            glob.push(a, kk, rec, lo_m1, P)
            if not rec:
                if kk >= P - 1:
                    rk[0] = rk[0] * d + (1.0 if kk == P else 0.0)
                    rk[1] = rk[1] * d + 1.0
                if a >= P - 1 and kk == P:
                    ra[0] = ra[0] * d + (1.0 if a == P else 0.0)
                    ra[1] = ra[1] * d + 1.0

    def _settle_pending(self, now):
        while self.pend and self.pend[0][0] < now:
            item = self.pend.popleft()
            dt = (now - item[0]) * 1000.0
            if 0.0 < dt < MAX_STEP_GAP_MS:
                self._settle(item, dt)
            else:
                self.dec.pop(item[0], None)
                self.win["untimed"] += 1

    def _settle(self, item, dt):
        t_prep, A, K, R, qs, (ra, rk) = item
        w = self.win
        dec = self.dec.pop(t_prep, None)
        if dec is None:
            w["missing"] += 1
            return
        j_sel, j_le0, e_pred = dec
        bs = len(A)
        P, W0, ws = self.P, self.W0, self.widths
        A0 = sum(A) + bs
        L0 = sum(K) + bs
        D0 = bs * W0 - L0
        rm = ra * rk
        cl, cd = self.coef(bs)
        hs = self.c["hi_scale"]
        tok, ms = [], []
        for wv in ws:
            if wv <= W0:
                wm = wv - 1
                t_w = bs + sum(a if a < wm else wm for a in A)
                l_w = bs + sum(x if x < wm else wm for x in K)
                s = 1.0
            else:
                m = wv - 1 - P
                ext_a = ext_k = 0.0
                g_m = g_k = None
                for i in range(bs):
                    if A[i] == P:
                        if R[i]:
                            ext_a += geo_sum(qs[i], m)
                        else:
                            if g_m is None:
                                g_m = geo_sum(rm, m)
                            ext_a += g_m
                    if K[i] == P:
                        if R[i]:
                            ext_k += m
                        else:
                            if g_k is None:
                                g_k = geo_sum(rk, m)
                            ext_k += g_k
                t_w = A0 + ext_a
                l_w = L0 + ext_k
                s = hs
            d_w = bs * wv - l_w
            tok.append(float(t_w))
            ms.append(float(max(dt + s * (cl * (l_w - L0) + cd * (d_w - D0)), 0.25 * dt)))
        allowed = self.allowed(bs)
        j_or = allowed[0]
        for j in allowed[1:]:
            if tok[j] * ms[j_or] > tok[j_or] * ms[j]:
                j_or = j
        bk = bs_bucket(bs)
        b = w["by_bs"].get(bk)
        if b is None:
            b = w["by_bs"][bk] = self._new_bucket()
        b["n"] += 1
        for key, tv, mv in (("act", A0, dt), ("sel", tok[j_sel], ms[j_sel]),
                            ("sel_lo", A0 if ws[j_sel] > W0 else tok[j_sel], ms[j_sel]),
                            ("le0", tok[j_le0], ms[j_le0]), ("oracle", tok[j_or], ms[j_or])):
            b[key][0] += tv
            b[key][1] += mv
        st = b["static"]
        for j, wv in enumerate(ws):
            st[wv][0] += tok[j]
            st[wv][1] += ms[j]
        b["pick"][ws[j_sel]] += 1
        b["pick_le0"][ws[j_le0]] += 1
        b["pred"][0] += e_pred
        b["pred"][1] += tok[j_sel]
        w["timed"] += 1
        q = self.tq.get(bk)
        if q is None:
            q = self.tq[bk] = collections.deque(maxlen=self.c["t_window"])
        q.append(float(dt))

    def flush(self):
        w = self.win
        ra, rk = self.rho()
        out = {"W0": self.W0, "widths": list(self.widths), "steps": w["steps"], "timed": w["timed"],
               "untimed": w["untimed"], "missing": w["missing"], "switches": w["switches"],
               "switches_le0": w["switches_le0"], "rho": [round(ra, 4), round(rk, 4)],
               "us_mean": round(w["us"] / w["us_n"], 2) if w["us_n"] else None,
               "t_w0": {str(b): round(_median(q), 3) for b, q in sorted(self.tq.items()) if q},
               "by_bs": {}}
        for bk in sorted(w["by_bs"]):
            x = w["by_bs"][bk]
            out["by_bs"][str(bk)] = {
                "n": x["n"], "act": _r2(x["act"]), "sel": _r2(x["sel"]), "sel_lo": _r2(x["sel_lo"]),
                "le0": _r2(x["le0"]), "oracle": _r2(x["oracle"]),
                "static": {str(k): _r2(v) for k, v in x["static"].items()},
                "pick": {str(k): int(v) for k, v in x["pick"].items()},
                "pick_le0": {str(k): int(v) for k, v in x["pick_le0"].items()},
                "pred": _r2(x["pred"])}
        self._new_window()
        return out


class _Req(object):
    __slots__ = ("rid", "temp", "bucket", "beta", "qa", "qb", "last_seen", "steps", "n_obs", "sum_k",
                 "sum_acc", "n_rec", "sum_thr1", "n_thr1", "tag")

    def __init__(self, rid, temp, bucket, beta, tag):
        self.rid, self.temp, self.bucket, self.beta, self.tag = rid, temp, bucket, beta, tag
        self.qa = self.qb = 0.0
        self.last_seen = self.steps = self.n_obs = self.sum_k = self.sum_acc = self.n_rec = self.n_thr1 = 0
        self.sum_thr1 = 0.0


class VcapPolicy(object):
    def __init__(self, P, cfg=None, seed=0):
        self.P = int(P)
        self.cfg = merge_cfg(DEFAULTS, cfg)
        self.rng = np.random.default_rng(seed)
        self.reqs = {}
        self.step = 0
        self.theta = None
        self.step_ms = None
        self.prep_ts = collections.deque(maxlen=128)
        self.thr_by_t = collections.OrderedDict()
        self.O = np.zeros(self.P)
        self.Pd = np.zeros(self.P)
        self.beta_b = np.ones(N_BUCKETS)
        self.q_b = np.full(N_BUCKETS, Q0)
        self.finished = []
        self.tot = collections.Counter()
        self._reset_window()
        self._wd = None
        self._wd_err = None
        self._sync_width()

    # ------------------------------------------------------------ 配置与状态
    def configure(self, cfg):
        self.cfg = merge_cfg(self.cfg, cfg)
        self._sync_width()

    def _sync_width(self):
        if not width_on(self.cfg):
            self._wd = None
            self._wd_err = None
            return
        try:
            c = width_cfg(self.cfg.get("width"))
        except Exception as exc:  # noqa: BLE001
            self._wd = None
            self._wd_err = ("config: %r" % (exc,))[:300]
            return
        if self._wd is not None and self._wd.c == c:
            return
        self._wd = _WidthShadow(self.P, c)
        self._wd_err = None

    def _width_fail(self, where, exc):
        self._wd = None
        self._wd_err = ("%s: %r" % (where, exc))[:300]

    def _reset_window(self):
        P = self.P
        self.win = {"steps": 0, "req_steps": 0, "sum_bs": 0, "sum_live": 0, "sum_acc": 0,
                    "hist_k": np.zeros(P, dtype=np.int64), "thr_sum": np.zeros(P), "thr_n": 0, "rec": 0,
                    "explore_n": 0, "policy_us": 0.0, "policy_n": 0, "acc_over_k": 0, "theta_sum": 0.0,
                    "theta_n": 0, "lam_sum": 0.0, "lam_n": 0, "step_ms_sum": 0.0, "step_ms_n": 0,
                    "dropped_obs": 0, "len_mismatch": 0}

    def count(self, name, n=1):
        if name in self.win:
            self.win[name] += n
        self.tot[name] += n

    def note_cost(self, us):
        self.win["policy_us"] += float(us)
        self.win["policy_n"] += 1

    def kbounds(self):
        kmax = self.cfg["kmax"] or self.P
        kmax = max(1, min(int(kmax), self.P))
        kmin = max(1, min(int(self.cfg["kmin"]), kmax))
        return kmin, kmax

    def lam(self):
        cfg = self.cfg
        if not cfg["logics"]["load"] or self.theta is None:
            return float(min(max(cfg["thr"], 1e-6), 1.0))
        return float(min(max(cfg["delta_ms"] * self.theta, cfg["lam_min"]), cfg["lam_max"]))

    def S(self):
        if not self.cfg["logics"]["pos"]:
            return np.ones(self.P)
        s = np.clip((self.O + S_PRIOR) / (self.Pd + S_PRIOR), S_MIN, S_MAX)
        return np.cumprod(s)

    def _q_of(self, r):
        qb = self.q_b[r.bucket] if self.cfg["logics"]["temp"] else Q0
        q = (r.qa + Q_PRIOR * qb) / (r.qa + r.qb + Q_PRIOR)
        return float(min(max(q, Q_MIN), Q_MAX))

    def _new_req(self, rid, temp):
        b = bucket_of(temp)
        beta = float(self.beta_b[b]) if self.cfg["logics"]["temp"] else 1.0
        r = _Req(rid, float(temp or 0.0), b, beta, self.cfg.get("tag", ""))
        self.reqs[rid] = r
        self.tot["new_reqs"] += 1
        return r

    # ------------------------------------------------------------ 每步开头
    def prepare_step(self, rids, temps, now, need=True):
        self.step += 1
        self.prep_ts.append(float(now))
        bs = len(rids)
        lg = self.cfg["logics"]
        use_req, use_temp = lg["req"], lg["temp"]
        betas = np.empty(bs)
        qs = np.empty(bs)
        rows = []
        reqs = self.reqs
        step_no = self.step
        q_b = self.q_b
        for i, rid in enumerate(rids):
            r = reqs.get(rid)
            if r is None:
                r = self._new_req(rid, temps[i] if i < len(temps) else 0.0)
            r.last_seen = step_no
            r.steps += 1
            rows.append(r)
            betas[i] = r.beta if use_req else 1.0
            qb = q_b[r.bucket] if use_temp else Q0
            qs[i] = (r.qa + Q_PRIOR * qb) / (r.qa + r.qb + Q_PRIOR)
        if self.step % 16 == 0:
            self._gc()
        if self._wd is not None:
            self._width_prepare(rids, rows, now)
        if not need:
            return None
        P = self.P
        kmin, kmax = self.kbounds()
        if not any(lg[x] for x in LOGICS):
            thr = np.full((bs, P), self.cfg["thr"], dtype=np.float32)
            krec = np.full(bs, kmax, dtype=np.int64)
            kadd = np.zeros(bs, dtype=np.int64)
        else:
            lam = self.lam()
            base = np.clip(lam / self.S(), 1e-9, 1.0)
            thr = np.power(base[None, :], 1.0 / betas[:, None])
            thr = np.clip(thr, self.cfg["thr_min"], self.cfg["thr_max"]).astype(np.float32)
            if lg["recall"]:
                lq = np.log(np.clip(qs, Q_MIN, Q_MAX))
                krec = np.floor(math.log(max(lam, 1e-9)) / lq)
                krec = np.clip(np.nan_to_num(krec, nan=kmax, posinf=kmax, neginf=kmin), kmin, kmax).astype(np.int64)
            else:
                krec = np.full(bs, kmax, dtype=np.int64)
            ex = self.cfg["explore"]
            if ex > 0:
                kadd = (self.rng.random(bs) < ex).astype(np.int64) * int(self.cfg["explore_add"])
            else:
                kadd = np.zeros(bs, dtype=np.int64)
            self.win["lam_sum"] += lam
            self.win["lam_n"] += 1
        self.win["explore_n"] += int((kadd > 0).sum())
        for i, r in enumerate(rows):
            r.sum_thr1 += float(thr[i, 0])
            r.n_thr1 += 1
        self.thr_by_t[float(now)] = thr
        while len(self.thr_by_t) > 16:
            self.thr_by_t.popitem(last=False)
        return thr, krec, kadd

    def _width_prepare(self, rids, rows, now):
        wd = self._wd
        try:
            t0 = time.perf_counter()
            wd.decide(list(rids), [self._q_of(r) for r in rows], float(now))
            wd.win["us"] += (time.perf_counter() - t0) * 1e6
            wd.win["us_n"] += 1
        except Exception as exc:  # noqa: BLE001
            self._width_fail("prepare", exc)

    def _width_observe(self, rids, k, acc, is_rec, t_prep, dt_ms):
        wd = self._wd
        try:
            t0 = time.perf_counter()
            reqs = self.reqs
            qs = [self._q_of(reqs[rid]) if rid in reqs else Q0 for rid in rids]
            wd.observe(list(rids), k, acc, is_rec, qs, float(t_prep), dt_ms)
            wd.win["us"] += (time.perf_counter() - t0) * 1e6
        except Exception as exc:  # noqa: BLE001
            self._width_fail("observe", exc)

    # ------------------------------------------------------------ 结果到 CPU 后
    def _next_prep(self, t_prep):
        for t in self.prep_ts:
            if t > t_prep:
                return t
        return None

    def observe(self, rids, k, cum, is_rec, acc, t_prep):
        bs = len(rids)
        if bs == 0:
            return
        P = self.P
        k = np.asarray(k, dtype=np.int64)[:bs]
        acc_raw = np.asarray(acc, dtype=np.int64)[:bs]
        cum = np.asarray(cum, dtype=np.float64)[:bs, :P]
        is_rec = np.asarray(is_rec, dtype=bool)[:bs]
        over = int((acc_raw > k).sum())
        acc = np.minimum(np.maximum(acc_raw, 0), k)
        lg = self.cfg["logics"]
        a = self.cfg["ema"]
        t_next = self._next_prep(t_prep)
        w_dt = None
        if t_next is not None:
            dt_ms = (t_next - t_prep) * 1000.0
            if 0.0 < dt_ms < MAX_STEP_GAP_MS:
                w_dt = dt_ms
                inst = float((acc + 1).sum()) / dt_ms
                self.theta = inst if self.theta is None else (1 - a) * self.theta + a * inst
                self.step_ms = dt_ms if self.step_ms is None else (1 - a) * self.step_ms + a * dt_ms
                self.win["step_ms_sum"] += dt_ms
                self.win["step_ms_n"] += 1
        if self.theta is not None:
            self.win["theta_sum"] += self.theta
            self.win["theta_n"] += 1
        w = self.win
        w["steps"] += 1
        w["req_steps"] += bs
        w["sum_bs"] += bs
        w["sum_live"] += int((k + 1).sum())
        w["sum_acc"] += int(acc.sum())
        w["acc_over_k"] += over
        w["rec"] += int(is_rec.sum())
        kk = np.clip(k, 1, P)
        w["hist_k"] += np.bincount(kk - 1, minlength=P)[:P]
        thr = self.thr_by_t.get(float(t_prep))
        if thr is not None and thr.shape[0] == bs:
            m = ~is_rec
            if m.any():
                w["thr_sum"] += thr[m].sum(axis=0)
                w["thr_n"] += int(m.sum())
        if self._wd is not None:
            self._width_observe(rids, k, acc, is_rec, t_prep, w_dt)
        S_pos = self.S()
        rows = [self.reqs.get(rid) for rid in rids]
        head = []
        for i in range(bs):
            r = rows[i]
            if r is None:
                continue
            ki = int(k[i])
            ai = int(acc[i])
            r.n_obs += 1
            r.sum_k += ki
            r.sum_acc += ai
            if is_rec[i]:
                r.n_rec += 1
                trials = min(ai + 1, ki)
                if trials > 0:
                    r.qa = r.qa * Q_DECAY + ai
                    r.qb = r.qb * Q_DECAY + (trials - ai)
                    self.q_b[r.bucket] += BUCKET_LR * (ai / float(trials) - self.q_b[r.bucket])
            elif ki >= 1:
                head.append(i)
        if not head or not (lg["req"] or lg["pos"]):
            return
        idx = np.asarray(head)
        K = k[idx]
        A = acc[idx]
        C = np.clip(cum[idx], 1e-6, 1.0)
        B = np.array([rows[i].beta for i in head]) if lg["req"] else np.ones(len(head))
        pos = np.arange(P)[None, :]
        CB = np.power(C, B[:, None])
        if lg["req"]:
            pred = np.clip(S_pos[None, :] * CB, 0.0, 1.0) * (pos < K[:, None])
            err = (pred.sum(axis=1) - A) / K
            nb = np.clip(np.exp(np.log(B) + BETA_LR * err), BETA_MIN, BETA_MAX)
            for j, i in enumerate(head):
                r = rows[i]
                r.beta = float(nb[j])
                self.beta_b[r.bucket] += BUCKET_LR * (r.beta - self.beta_b[r.bucket])
        if lg["pos"]:
            prev = np.concatenate((np.ones((len(head), 1)), C[:, :-1]), axis=1)
            cond = np.power(np.clip(C / prev, 0.0, 1.0), B[:, None])
            mask_r = pos < np.minimum(A + 1, K)[:, None]
            obs = (pos < A[:, None]) & mask_r
            n = mask_r.sum(axis=0)
            dec = np.power(S_DECAY, n)
            self.O = self.O * dec + obs.sum(axis=0)
            self.Pd = self.Pd * dec + (cond * mask_r).sum(axis=0)

    # ------------------------------------------------------------ 结束请求与统计
    def _finalize(self, r):
        n = max(r.n_obs, 1)
        self.finished.append({
            "rid": r.rid, "tag": r.tag, "temp": r.temp, "bucket": r.bucket, "steps": r.steps, "obs": r.n_obs,
            "beta": round(r.beta, 4), "q": round(self._q_of(r), 4), "mean_k": round(r.sum_k / float(n), 3),
            "mean_acc": round(r.sum_acc / float(n), 3), "frac_rec": round(r.n_rec / float(n), 3),
            "thr1_mean": round(r.sum_thr1 / r.n_thr1, 4) if r.n_thr1 else None,
        })
        if len(self.finished) > 4096:
            del self.finished[:len(self.finished) - 4096]

    def _gc(self):
        for rid in [rid for rid, r in self.reqs.items() if self.step - r.last_seen > GC_STEPS]:
            self._finalize(self.reqs.pop(rid))
            self.tot["finished"] += 1

    def drain_finished(self):
        out, self.finished = self.finished, []
        return out

    def stats(self):
        w = self.win
        P = self.P
        rs = max(w["req_steps"], 1)
        res = {
            "block": P, "steps": w["steps"], "req_steps": w["req_steps"],
            "mean_bs": round(w["sum_bs"] / float(max(w["steps"], 1)), 3),
            "mean_live": round(w["sum_live"] / float(rs), 4),
            "mean_acc": round(w["sum_acc"] / float(rs), 4),
            "mean_commit": round((w["sum_acc"] + w["req_steps"]) / float(rs), 4),
            "hist_k": [int(x) for x in w["hist_k"]],
            "theta": round(w["theta_sum"] / w["theta_n"], 5) if w["theta_n"] else None,
            "lam": round(w["lam_sum"] / w["lam_n"], 5) if w["lam_n"] else None,
            "step_ms": round(w["step_ms_sum"] / w["step_ms_n"], 3) if w["step_ms_n"] else None,
            "thr_mean": [round(float(x), 4) for x in (w["thr_sum"] / w["thr_n"])] if w["thr_n"] else None,
            "frac_rec": round(w["rec"] / float(rs), 4),
            "beta_by_bucket": [round(float(x), 4) for x in self.beta_b],
            "q_by_bucket": [round(float(x), 4) for x in self.q_b],
            "s_pos": [round(float(x), 4) for x in np.clip((self.O + S_PRIOR) / (self.Pd + S_PRIOR), S_MIN, S_MAX)],
            "explore_n": w["explore_n"], "dropped_obs": w["dropped_obs"], "len_mismatch": w["len_mismatch"],
            "acc_over_k": w["acc_over_k"],
            "policy_us_mean": round(w["policy_us"] / w["policy_n"], 2) if w["policy_n"] else None,
            "active_reqs": len(self.reqs),
        }
        self._reset_window()
        if self._wd is not None:
            try:
                res["width"] = self._wd.flush()
            except Exception as exc:  # noqa: BLE001
                self._width_fail("stats", exc)
        if self._wd_err is not None:
            res["width_error"] = self._wd_err
        return res
