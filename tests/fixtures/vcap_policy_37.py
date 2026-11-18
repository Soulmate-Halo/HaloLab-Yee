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
"""
import collections
import math

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
    for k, v in (upd or {}).items():
        if k == "logics" and isinstance(v, dict):
            for lk, lv in v.items():
                if lk in LOGICS:
                    out["logics"][lk] = bool(lv)
        elif k in DEFAULTS:
            out[k] = v
    for k in ("delta_ms", "thr", "thr_min", "thr_max", "lam_min", "lam_max", "explore", "ema", "stats_s"):
        out[k] = float(out[k])
    out["kmin"] = int(out["kmin"])
    out["explore_add"] = int(out["explore_add"])
    out["kmax"] = None if out.get("kmax") in (None, "", 0) else int(out["kmax"])
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

    # ------------------------------------------------------------ 配置与状态
    def configure(self, cfg):
        self.cfg = merge_cfg(self.cfg, cfg)

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
        if t_next is not None:
            dt_ms = (t_next - t_prep) * 1000.0
            if 0.0 < dt_ms < MAX_STEP_GAP_MS:
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
        return res
