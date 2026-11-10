//! Cross-request recall for DSpark copy drafts (sglang_d_recall's record, fifo): recent prompts and replies in one
//! ring of token ids indexed by 16-id windows, and the per-tier gate that decides when a recalled continuation is
//! worth a verify. Pure: the process's instance and its hooks live in `d_recall_live.zig`.

const std = @import("std");

pub const WINDOW = 16;
pub const MAX_BACK = 128;
/// A window is indexed only when this many ids of its own segment follow it: the end of a prompt must not shadow
/// the earlier reply that followed the same ids.
pub const MIN_TAIL = 4;
/// A run the record already holds is skipped once it covers this many ids.
pub const DEDUP_RUN = 128;
/// Ids a kept segment reaches into the duplicate runs around it, so its windows still match across the seam.
pub const OVERLAP = 32;
/// Prompt ids written ahead of a reply, so the reply is found from the prompt's end.
pub const PROMPT_TAIL = 64;
/// Recalled drafts a round can verify (the module's verify ring).
pub const MAX_DRAFTS = 15;
pub const SENTINEL: u32 = std.math.maxInt(u32);
pub const DEFAULT_GIB: f32 = 0.0625;
pub const MIN_CAP = 4096;
/// Hash slots per stored id in the index: the default, and the range a raw density is clamped to.
pub const DEFAULT_INDEX_DENSITY: f32 = 0.25;
pub const MIN_INDEX_DENSITY: f32 = 1.0 / 64.0;
pub const MAX_INDEX_DENSITY: f32 = 4.0;

const PRIOR_FRAC = 0.8;
const PRIOR_W = 2.0;
const DECAY = 0.98;
const COOLDOWN = 4;
const PROBE_START = 16;
const PROBE_MAX = 4096;
const OWN_ALPHA = 0.25;
/// Ids `dropRun` compares directly beyond a run's indexed windows.
const EXTEND = 256;

const P: u64 = 0x9E3779B97F4A7C15;
const M1: u64 = 0xBF58476D1CE4E5B9;
const P15: u64 = blk: {
    var r: u64 = 1;
    for (0..WINDOW - 1) |_| r *%= P;
    break :blk r;
};

fn hashOf(t: []const u32) u64 {
    var h: u64 = 0;
    for (t) |x| h = h *% P +% (@as(u64, x) + 1);
    return h;
}

fn roll(h: u64, out: u32, in: u32) u64 {
    return (h -% (@as(u64, out) + 1) *% P15) *% P +% (@as(u64, in) + 1);
}

/// Shorter windows the lookup backs off to when the 16-id window misses, longest first: a reply that forked from
/// a recorded one is found again a few ids after the fork instead of 16, and a short prompt still finds a reply.
pub const SHORT_WINDOWS = [_]usize{ 8, 4 };
/// The tier a match shorter than WINDOW lands in, and its prior as a fraction of the long tiers' prior.
pub const SHORT_TIER: u2 = 3;
const SHORT_PRIOR = 0.5;

fn powP(n: usize) u64 {
    var r: u64 = 1;
    for (0..n) |_| r *%= P;
    return r;
}

fn rollW(h: u64, out: u32, in: u32, pw: u64) u64 {
    return (h -% (@as(u64, out) + 1) *% pw) *% P +% (@as(u64, in) + 1);
}

/// One short-window index: slot -> window hash and the position right after that window (0 = empty).
const ShortIndex = struct {
    w: usize,
    pw: u64,
    mask: u64,
    keys: []u64,
    pos: []u64,
};

fn mix(h: u64) u64 {
    var x = h ^ (h >> 31);
    x *%= M1;
    return x ^ (x >> 27);
}

/// Ids a record of `gib` GiB holds (4 bytes an id; the index adds 16 bytes a slot: about 4 x density times the id
/// bytes, rounded up to a power of two).
pub fn capFor(gib: f32) usize {
    const ids: f64 = @as(f64, gib) * (1 << 30) / 4;
    if (!(ids < 1 << 40)) return MIN_CAP;
    return @max(MIN_CAP, @as(usize, @intFromFloat(ids)));
}

/// The density in effect: the raw value clamped to its range, the default when it is not a number.
pub fn indexDensity(raw: f32) f32 {
    if (std.math.isNan(raw)) return DEFAULT_INDEX_DENSITY;
    return @max(MIN_INDEX_DENSITY, @min(MAX_INDEX_DENSITY, raw));
}

/// The index slots a record of `cap_ids` ids holds at `density` slots an id (a power of two).
pub fn slotsFor(cap_ids: usize, density: f32) usize {
    const cap = @max(cap_ids, MIN_CAP);
    const d = indexDensity(density);
    const want: usize = @max(2, @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(cap)) * @as(f64, @floatCast(d))))));
    const bits: u6 = @intCast(@max(16, 64 - @clz(@as(u64, want - 1))));
    return @as(usize, 1) << bits;
}

pub fn tierOf(len: usize) u2 {
    if (len < WINDOW) return SHORT_TIER;
    if (len >= 64) return 2;
    if (len >= 32) return 1;
    return 0;
}

/// The slots of each short-window index beside a 16-id index of `n` slots (a power of two, at least 1 << 16).
pub fn shortSlotsFor(n: usize) usize {
    return @max(@as(usize, 1) << 16, n / 4);
}

pub const Store = struct {
    gpa: std.mem.Allocator,
    /// The ring; absolute position `g` lives at `ids[g % cap]` while `g >= base()`.
    ids: []u32,
    /// Slot -> window hash and the position right after that window (0 = empty).
    keys: []u64,
    pos: []u64,
    cap: u64,
    mask: u64,
    end: u64 = 0,
    added: u64 = 0,
    deduped: u64 = 0,
    lookups: u64 = 0,
    hits: u64 = 0,
    /// Hits found only by a short window (counted in `hits` too).
    short_hits: u64 = 0,
    short: [SHORT_WINDOWS.len]ShortIndex,

    pub const Hit = struct { pos: u64, len: usize };

    pub fn init(gpa: std.mem.Allocator, cap_ids: usize, density: f32) !Store {
        const cap = @max(cap_ids, MIN_CAP);
        const n = slotsFor(cap_ids, density);
        const ids = try gpa.alloc(u32, cap);
        errdefer gpa.free(ids);
        const keys = try gpa.alloc(u64, n);
        errdefer gpa.free(keys);
        const pos = try gpa.alloc(u64, n);
        errdefer gpa.free(pos);
        @memset(keys, 0);
        @memset(pos, 0);
        var short: [SHORT_WINDOWS.len]ShortIndex = undefined;
        var made: usize = 0;
        errdefer {
            for (short[0..made]) |si| {
                gpa.free(si.keys);
                gpa.free(si.pos);
            }
        }
        // A short-window index takes a quarter of the 16-id index's slots: its newest entries are what it serves.
        const ns = shortSlotsFor(n);
        for (SHORT_WINDOWS, 0..) |w, i| {
            const sk = try gpa.alloc(u64, ns);
            const sp = gpa.alloc(u64, ns) catch |e| {
                gpa.free(sk);
                return e;
            };
            @memset(sk, 0);
            @memset(sp, 0);
            short[i] = .{ .w = w, .pw = powP(w - 1), .mask = ns - 1, .keys = sk, .pos = sp };
            made += 1;
        }
        return .{ .gpa = gpa, .ids = ids, .keys = keys, .pos = pos, .cap = cap, .mask = n - 1, .short = short };
    }

    pub fn deinit(self: *Store) void {
        self.gpa.free(self.ids);
        self.gpa.free(self.keys);
        self.gpa.free(self.pos);
        for (self.short) |si| {
            self.gpa.free(si.keys);
            self.gpa.free(si.pos);
        }
    }

    pub fn bytes(self: *const Store) usize {
        var b = self.ids.len * @sizeOf(u32) + self.keys.len * @sizeOf(u64) * 2;
        for (self.short) |si| b += si.keys.len * @sizeOf(u64) * 2;
        return b;
    }

    pub fn slots(self: *const Store) usize {
        return @intCast(self.mask + 1);
    }

    pub fn base(self: *const Store) u64 {
        return self.end -| self.cap;
    }

    pub fn held(self: *const Store) u64 {
        return self.end - self.base();
    }

    fn at(self: *const Store, g: u64) u32 {
        return self.ids[@intCast(g % self.cap)];
    }

    fn put(self: *Store, t: u32) void {
        self.ids[@intCast(self.end % self.cap)] = t;
        self.end += 1;
    }

    fn hashAt(self: *const Store, g: u64) u64 {
        var h: u64 = 0;
        var j: u64 = 0;
        while (j < WINDOW) : (j += 1) h = h *% P +% (@as(u64, self.at(g + j)) + 1);
        return h;
    }

    /// Appends the part of `a` the record does not hold yet; returns the ids written.
    pub fn add(self: *Store, a: []const u32) usize {
        if (a.len < WINDOW) return 0;
        const keep = self.gpa.alloc(bool, a.len) catch return 0;
        defer self.gpa.free(keep);
        self.novelMask(a, keep);
        var kept: usize = 0;
        for (keep) |k| kept += @intFromBool(k);
        var written: usize = 0;
        var i: usize = 0;
        while (i < a.len) {
            if (!keep[i]) {
                i += 1;
                continue;
            }
            var j = i;
            while (j < a.len and keep[j]) j += 1;
            const s = i -| OVERLAP;
            const e = @min(a.len, j + OVERLAP);
            if (e - s >= WINDOW) written += self.appendSegment(a[s..e]);
            i = j;
        }
        self.deduped += a.len - kept;
        self.added += written;
        return written;
    }

    const Run = struct { s: usize, e: usize, p_s: u64, p_e: u64 };

    /// `keep[i]` false for ids inside runs of at least DEDUP_RUN ids the record holds (or all of `a` when it holds
    /// all of it): consecutive windows whose index entries sit at consecutive positions, widened by comparing ids.
    fn novelMask(self: *const Store, a: []const u32, keep: []bool) void {
        @memset(keep, true);
        if (self.held() < WINDOW or a.len < WINDOW) return;
        const base_g = self.base();
        const need: usize = if (a.len < 2 * WINDOW) WINDOW else 2 * WINDOW;
        var h = hashOf(a[0..WINDOW]);
        var run: ?Run = null;
        var i: usize = 0;
        while (i + WINDOW <= a.len) : (i += 1) {
            if (i > 0) h = roll(h, a[i - 1], a[i + WINDOW - 1]);
            const slot: usize = @intCast(mix(h) & self.mask);
            const p = self.pos[slot];
            const valid = self.keys[slot] == h and p >= base_g + WINDOW and p <= self.end;
            if (valid) {
                if (run) |*r| {
                    if (p == r.p_e + 1) {
                        r.e = i;
                        r.p_e = p;
                        continue;
                    }
                }
            }
            if (run) |r| self.dropRun(a, r, need, keep);
            run = if (valid) .{ .s = i, .e = i, .p_s = p, .p_e = p } else null;
        }
        if (run) |r| self.dropRun(a, r, need, keep);
    }

    fn dropRun(self: *const Store, a: []const u32, r: Run, need: usize, keep: []bool) void {
        if (r.e - r.s + WINDOW < need) return;
        var ts = r.s;
        const rs = r.p_s - WINDOW;
        const kb: usize = @intCast(@min(@as(u64, @min(EXTEND, ts)), rs - self.base()));
        var j: usize = 0;
        while (j < kb and self.at(rs - 1 - @as(u64, j)) == a[ts - 1 - j]) j += 1;
        ts -= j;
        var te = r.e + WINDOW;
        const re = r.p_e;
        const kf: usize = @intCast(@min(@as(u64, @min(EXTEND, a.len - te)), self.end - re));
        j = 0;
        while (j < kf and self.at(re + @as(u64, j)) == a[te + j]) j += 1;
        te += j;
        if (te - ts >= DEDUP_RUN or (ts == 0 and te == a.len)) @memset(keep[ts..te], false);
    }

    fn appendSegment(self: *Store, seg_in: []const u32) usize {
        const room: usize = @intCast(self.cap / 2);
        const seg = if (seg_in.len > room) seg_in[seg_in.len - room ..] else seg_in;
        if (self.end > self.base() and self.at(self.end - 1) != SENTINEL) self.put(SENTINEL);
        const start = self.end;
        for (seg) |t| self.put(t);
        self.indexRange(start, self.end);
        return seg.len;
    }

    /// Indexes every window whose last id lies in [g0, g1) and that has MIN_TAIL ids of its segment after it, in
    /// the 16-id index and in each short-window index.
    fn indexRange(self: *Store, g0: u64, g1: u64) void {
        self.indexRangeW(self.keys, self.pos, self.mask, WINDOW, P15, g0, g1);
        for (self.short) |si| self.indexRangeW(si.keys, si.pos, si.mask, si.w, si.pw, g0, g1);
    }

    fn hashAtW(self: *const Store, g: u64, w: u64) u64 {
        var h: u64 = 0;
        var j: u64 = 0;
        while (j < w) : (j += 1) h = h *% P +% (@as(u64, self.at(g + j)) + 1);
        return h;
    }

    fn indexRangeW(self: *Store, keys: []u64, pos: []u64, mask: u64, w_in: usize, pw: u64, g0: u64, g1: u64) void {
        const w: u64 = w_in;
        const span: u64 = w + MIN_TAIL;
        const s0 = @max(self.base(), g0 -| (w - 1));
        const g_ext = @min(self.end, g1 + MIN_TAIL);
        if (g_ext < s0 + span or g1 < s0 + w) return;
        const m = @min(g_ext - s0 - span + 1, g1 - w - s0 + 1);
        var h = self.hashAtW(s0, w);
        var bad: usize = 0;
        var j: u64 = 0;
        while (j < span) : (j += 1) bad += @intFromBool(self.at(s0 + j) == SENTINEL);
        var k: u64 = 0;
        while (k < m) : (k += 1) {
            if (k > 0) {
                h = rollW(h, self.at(s0 + k - 1), self.at(s0 + k + w - 1), pw);
                bad -= @intFromBool(self.at(s0 + k - 1) == SENTINEL);
                bad += @intFromBool(self.at(s0 + k + span - 1) == SENTINEL);
            }
            if (bad != 0) continue;
            const slot: usize = @intCast(mix(h) & mask);
            keys[slot] = h;
            pos[slot] = s0 + k + w;
        }
    }

    /// The newest occurrence of `tail`'s last 16 ids, else of its last 8, else of its last 4: where its
    /// continuation starts and how far back it agrees with `tail` (at most MAX_BACK).
    pub fn lookup(self: *Store, tail: []const u32) ?Hit {
        if (tail.len < SHORT_WINDOWS[SHORT_WINDOWS.len - 1] or self.held() < WINDOW) return null;
        self.lookups += 1;
        if (self.lookupW(self.keys, self.pos, self.mask, WINDOW, tail)) |hit| {
            self.hits += 1;
            return hit;
        }
        for (self.short) |si| {
            if (self.lookupW(si.keys, si.pos, si.mask, si.w, tail)) |hit| {
                self.hits += 1;
                self.short_hits += 1;
                return hit;
            }
        }
        return null;
    }

    fn lookupW(self: *const Store, keys: []const u64, pos: []const u64, mask: u64, w_in: usize, tail: []const u32) ?Hit {
        const w: usize = w_in;
        if (tail.len < w) return null;
        const t = tail[tail.len - w ..];
        const h = hashOf(t);
        const slot: usize = @intCast(mix(h) & mask);
        const p = pos[slot];
        const base_g = self.base();
        const wg: u64 = w;
        if (keys[slot] != h or p < base_g + wg or p > self.end) return null;
        for (t, 0..) |x, j| if (self.at(p - wg + @as(u64, j)) != x) return null;
        const lim: usize = @min(MAX_BACK, tail.len);
        var len: usize = w;
        while (len < lim) : (len += 1) {
            const back: u64 = len - w + 1;
            if (p - wg < base_g + back) break;
            if (self.at(p - wg - back) != tail[tail.len - w - @as(usize, @intCast(back))]) break;
        }
        return .{ .pos = p, .len = len };
    }

    /// Up to `out.len` ids from `pos`, stopping at a segment's end; 0 once `pos` left the ring.
    pub fn continuation(self: *const Store, pos: u64, out: []u32) usize {
        if (pos < self.base() or pos >= self.end) return 0;
        const avail: usize = @intCast(self.end - pos);
        const n = @min(out.len, avail);
        for (0..n) |i| {
            const t = self.at(pos + @as(u64, i));
            if (t == SENTINEL) return i;
            out[i] = t;
        }
        return n;
    }
};

const Stat = struct { n: f64 = 0, s: f64 = 0 };

const Pending = struct { tier: u2, probe: bool, real: u32, own: f64, sampled: bool = false };

/// One request's side of the gate.
pub const Stream = struct {
    /// Set when the request's prompt was recorded; `end` records its reply once.
    armed: bool = false,
    /// A sampled request: its copies land p(x) of the time, so the gate learns them apart from greedy ones.
    sampled: bool = false,
    joined: bool = false,
    /// Accepted drafts per stage round, smoothed: the bar a recalled copy must beat.
    own_ema: f64 = 0,
    cooldown: u8 = 0,
    pending: ?Pending = null,
    offers: u32 = 0,
    probes: u32 = 0,
    drafted: u64 = 0,
    landed: u64 = 0,

    /// Counts down the cooldown a copy that landed nothing started; false while it runs.
    pub fn ready(self: *Stream) bool {
        if (self.cooldown == 0) return true;
        self.cooldown -= 1;
        return false;
    }
};

/// Decayed landed tokens per offer, by match-length tier, against the stream's own stage drafts.
pub const Gate = struct {
    prior: f64,
    stats: [4]Stat = .{ .{}, .{}, .{}, .{} },
    /// The same per tier for sampled requests.
    stats_sampled: [4]Stat = .{ .{}, .{}, .{}, .{} },
    probe_iv: u32 = PROBE_START,
    probe_left: u32 = PROBE_START,
    own_global: f64,

    pub fn init(gamma: usize) Gate {
        const g: f64 = @floatFromInt(gamma);
        return .{ .prior = PRIOR_FRAC * g, .own_global = 0.5 * g };
    }

    pub fn expect(self: *const Gate, tier: u2) f64 {
        return self.expectFor(false, tier);
    }

    pub fn expectFor(self: *const Gate, sampled: bool, tier: u2) f64 {
        const s = if (sampled) self.stats_sampled[tier] else self.stats[tier];
        const prior = if (tier == SHORT_TIER) self.prior * SHORT_PRIOR else self.prior;
        return (s.s + prior * PRIOR_W) / (s.n + PRIOR_W);
    }

    fn join(self: *const Gate, st: *Stream) void {
        if (st.joined) return;
        st.joined = true;
        st.own_ema = self.own_global;
    }

    /// Offers a copy of `real` ids matched over `len`: while its tier does not beat the stream's own drafts, only
    /// one ask in `probe_iv` goes through, as a probe.
    pub fn admit(self: *Gate, st: *Stream, len: usize, real: usize) bool {
        const tier = tierOf(len);
        var probe = false;
        if (self.expectFor(st.sampled, tier) <= st.own_ema) {
            const left = self.probe_left -| 1;
            if (left > 0) {
                self.probe_left = left;
                return false;
            }
            self.probe_left = self.probe_iv;
            probe = true;
            st.probes += 1;
        }
        st.pending = .{ .tier = tier, .probe = probe, .real = @intCast(real), .own = st.own_ema, .sampled = st.sampled };
        st.offers += 1;
        return true;
    }

    /// What the target accepted of the offered copy.
    pub fn feedback(self: *Gate, st: *Stream, accepted: u32) void {
        const p = st.pending orelse return;
        st.pending = null;
        const landed = @min(accepted, p.real);
        const lf: f64 = @floatFromInt(landed);
        const s = if (p.sampled) &self.stats_sampled[p.tier] else &self.stats[p.tier];
        s.s = s.s * DECAY + lf;
        s.n = s.n * DECAY + 1;
        st.landed += landed;
        if (landed == 0) st.cooldown = COOLDOWN;
        if (p.probe) {
            self.probe_iv = if (lf > p.own) PROBE_START else @min(self.probe_iv * 2, PROBE_MAX);
            self.probe_left = self.probe_iv;
        }
    }

    pub fn observeOwn(self: *Gate, st: *Stream, accepted: u32) void {
        self.join(st);
        const a: f64 = @floatFromInt(accepted);
        st.own_ema = (1 - OWN_ALPHA) * st.own_ema + OWN_ALPHA * a;
        self.own_global = 0.99 * self.own_global + 0.01 * a;
    }
};

pub const Draft = struct { n: usize, len: u32 };

/// The record and its gate. Not thread-safe: one caller at a time.
pub const Recall = struct {
    store: Store,
    gate: ?Gate = null,

    pub fn init(gpa: std.mem.Allocator, cap_ids: usize, density: f32) !Recall {
        return .{ .store = try Store.init(gpa, cap_ids, density) };
    }

    pub fn deinit(self: *Recall) void {
        self.store.deinit();
    }

    /// The gate, built on first use with the stage block as its draft length.
    pub fn gateFor(self: *Recall, gamma: usize) *Gate {
        if (self.gate == null) self.gate = Gate.init(gamma);
        return &self.gate.?;
    }

    /// A recalled continuation of `ctx ++ [t1]` in `out`, more than `block` ids of it, when the gate pays for one.
    pub fn draft(self: *Recall, st: *Stream, ctx: []const u32, t1: u32, out: []u32, block: usize) ?Draft {
        const g = self.gateFor(block);
        g.join(st);
        if (out.len <= block or !st.ready()) return null;
        var tail: [MAX_BACK]u32 = undefined;
        const k: usize = @min(ctx.len, MAX_BACK - 1);
        @memcpy(tail[0..k], ctx[ctx.len - k ..]);
        tail[k] = t1;
        const hit = self.store.lookup(tail[0 .. k + 1]) orelse return null;
        var n = self.store.continuation(hit.pos, out);
        // A short match earns a short copy: twice what it agreed over, but always more than the stage block.
        if (hit.len < WINDOW) n = @min(n, @max(block + 1, 2 * hit.len));
        if (n <= block or !g.admit(st, hit.len, n)) return null;
        st.drafted += n;
        return .{ .n = n, .len = @intCast(hit.len) };
    }

    /// `verified` 0 = the round verified none of the offered copy: it teaches nothing.
    pub fn feedback(self: *Recall, st: *Stream, accepted: u32, verified: usize) void {
        if (verified == 0) {
            st.pending = null;
            return;
        }
        if (self.gate) |*g| g.feedback(st, accepted);
    }

    pub fn observeOwn(self: *Recall, st: *Stream, accepted: u32, gamma: usize) void {
        self.gateFor(gamma).observeOwn(st, accepted);
    }

    /// Records a finished reply behind the last PROMPT_TAIL ids of its prompt.
    pub fn remember(self: *Recall, prompt: []const u32, output: []const u32) void {
        if (output.len == 0) return;
        const head = prompt[prompt.len -| PROMPT_TAIL ..];
        const seg = self.store.gpa.alloc(u32, head.len + output.len) catch return;
        defer self.store.gpa.free(seg);
        @memcpy(seg[0..head.len], head);
        @memcpy(seg[head.len..], output);
        _ = self.store.add(seg);
    }
};

// ── tests ──

const testing = std.testing;

fn iota(out: []u32, first: u32) void {
    for (out, 0..) |*x, i| x.* = first + @as(u32, @intCast(i));
}

/// The slots a quarter density gives: a quarter of the ids, rounded up to a power of two.
fn slotsLegacy(cap_ids: usize) usize {
    const cap = @max(cap_ids, MIN_CAP);
    const want: usize = @max(2, cap / 4);
    const bits: u6 = @intCast(@max(16, 64 - @clz(@as(u64, want - 1))));
    return @as(usize, 1) << bits;
}

test "d_recall: a stored run continues after its 16-id window, and the match reaches back" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var seg: [100]u32 = undefined;
    iota(&seg, 1);
    try testing.expectEqual(@as(usize, 100), s.add(&seg));
    const hit = s.lookup(seg[0..30]).?;
    try testing.expectEqual(@as(usize, 30), hit.len);
    var out: [10]u32 = undefined;
    try testing.expectEqual(@as(usize, 10), s.continuation(hit.pos, &out));
    try testing.expectEqualSlices(u32, seg[30..40], &out);
}

test "d_recall: a window too close to its segment's end is not indexed, and a continuation stops at the separator" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var a: [40]u32 = undefined;
    iota(&a, 1);
    var b: [40]u32 = undefined;
    iota(&b, 501);
    _ = s.add(&a);
    _ = s.add(&b);
    try testing.expect(s.lookup(a[24..40]) == null);
    try testing.expect(s.lookup(a[21..37]) == null);
    const hit = s.lookup(a[20..36]).?;
    var out: [10]u32 = undefined;
    try testing.expectEqual(@as(usize, 4), s.continuation(hit.pos, &out));
    try testing.expectEqualSlices(u32, a[36..40], out[0..4]);
}

test "d_recall: a run the record already holds is not written again" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var a: [300]u32 = undefined;
    iota(&a, 1000);
    try testing.expectEqual(@as(usize, 300), s.add(&a));
    try testing.expectEqual(@as(usize, 0), s.add(&a));
    try testing.expectEqual(@as(u64, 300), s.end);
    var b: [500]u32 = undefined;
    @memcpy(b[0..300], &a);
    iota(b[300..], 5000);
    try testing.expectEqual(@as(usize, 200 + OVERLAP), s.add(&b));
}

test "d_recall: past its capacity the record forgets its oldest ids" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var seg: [1000]u32 = undefined;
    var r: u32 = 0;
    while (r < 5) : (r += 1) {
        iota(&seg, r * 10000 + 1);
        _ = s.add(&seg);
        if (r == 0) try testing.expect(s.lookup(seg[0..40]) != null);
    }
    var first: [40]u32 = undefined;
    iota(&first, 1);
    try testing.expect(s.base() > 0);
    try testing.expect(s.lookup(&first) == null);
    try testing.expect(s.lookup(seg[0..40]) != null);
}

test "d_recall: the match length stops at the first disagreement and at MAX_BACK" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var seg: [300]u32 = undefined;
    iota(&seg, 1);
    _ = s.add(&seg);
    try testing.expectEqual(@as(usize, MAX_BACK), s.lookup(seg[0..200]).?.len);
    var tail: [40]u32 = undefined;
    @memcpy(&tail, seg[0..40]);
    tail[10] = 99999;
    try testing.expectEqual(@as(usize, 29), s.lookup(&tail).?.len);
}

test "d_recall: tiers split the match length at 32 and 64" {
    try testing.expectEqual(@as(u2, 0), tierOf(16));
    try testing.expectEqual(@as(u2, 0), tierOf(31));
    try testing.expectEqual(@as(u2, 1), tierOf(32));
    try testing.expectEqual(@as(u2, 1), tierOf(63));
    try testing.expectEqual(@as(u2, 2), tierOf(64));
}

test "d_recall: a copy that lands nothing cools its stream down for four asks" {
    var g = Gate.init(5);
    var st: Stream = .{};
    g.join(&st);
    try testing.expectApproxEqAbs(@as(f64, 4.0), g.expect(0), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.5), st.own_ema, 1e-9);
    try testing.expect(st.ready());
    try testing.expect(g.admit(&st, 20, 10));
    g.feedback(&st, 0);
    try testing.expectApproxEqAbs(@as(f64, 8.0 / 3.0), g.expect(0), 1e-9);
    for (0..COOLDOWN) |_| try testing.expect(!st.ready());
    try testing.expect(st.ready());
}

test "d_recall: below the stream's own drafts a copy goes as a probe, and a failed probe waits twice as long" {
    var g = Gate.init(5);
    var st: Stream = .{};
    g.join(&st);
    st.own_ema = 10;
    var refused: usize = 0;
    while (!g.admit(&st, 20, 10)) refused += 1;
    try testing.expectEqual(@as(usize, PROBE_START - 1), refused);
    try testing.expect(st.pending.?.probe);
    g.feedback(&st, 3);
    try testing.expectEqual(@as(u32, 2 * PROBE_START), g.probe_iv);
    refused = 0;
    while (!g.admit(&st, 20, 12)) refused += 1;
    try testing.expectEqual(@as(usize, 2 * PROBE_START - 1), refused);
    g.feedback(&st, 12);
    try testing.expectEqual(@as(u32, PROBE_START), g.probe_iv);
}

test "d_recall: a reply remembered behind its prompt is drafted when the prompt comes back" {
    var r = try Recall.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer r.deinit();
    var prompt: [80]u32 = undefined;
    iota(&prompt, 100);
    var reply: [30]u32 = undefined;
    iota(&reply, 900);
    r.remember(&prompt, &reply);
    var st: Stream = .{};
    var out: [MAX_DRAFTS]u32 = undefined;
    try testing.expect(r.draft(&st, prompt[0..79], prompt[79], out[0..5], 5) == null);
    const d = r.draft(&st, prompt[0..79], prompt[79], &out, 5).?;
    try testing.expectEqual(@as(usize, MAX_DRAFTS), d.n);
    try testing.expectEqualSlices(u32, reply[0..MAX_DRAFTS], out[0..d.n]);
    try testing.expectEqual(@as(u32, 64), d.len);
    r.feedback(&st, 15, d.n);
    try testing.expectEqual(@as(u64, 15), st.landed);
    try testing.expect(st.pending == null);
}

test "d_recall: the default density keeps the slots the record took" {
    const caps = [_]usize{ MIN_CAP, 4097, 65536, 65537, 262145, 1000003, 16777216, 67108864 };
    for (caps) |cap| {
        try testing.expectEqual(slotsLegacy(cap), slotsFor(cap, DEFAULT_INDEX_DENSITY));
    }
}

test "d_recall: the slot count follows the density" {
    try testing.expectEqual(@as(usize, 1) << 22, slotsFor(16777216, 0.25));
    try testing.expectEqual(@as(usize, 1) << 26, slotsFor(67108864, 1));
    try testing.expectEqual(@as(usize, 1) << 16, slotsFor(MIN_CAP, 0.25));
    try testing.expectEqual(@as(usize, 1) << 28, slotsFor(67108864, 4));
}

test "d_recall: a density outside its range is clamped" {
    const x = 1000003;
    try testing.expectEqual(slotsFor(x, 4), slotsFor(x, 100));
    try testing.expectEqual(slotsFor(x, 1.0 / 64.0), slotsFor(x, 0));
    try testing.expectEqual(slotsFor(x, 0.25), slotsFor(x, std.math.nan(f32)));
    try testing.expectEqual(@as(f32, 4), indexDensity(100));
    try testing.expectEqual(@as(f32, 1.0 / 64.0), indexDensity(0));
    try testing.expectEqual(@as(f32, 0.25), indexDensity(std.math.nan(f32)));
    try testing.expectEqual(@as(f32, 1), indexDensity(1));
}

test "d_recall: a denser index still finds a stored run" {
    var s = try Store.init(testing.allocator, MIN_CAP, 1);
    defer s.deinit();
    try testing.expectEqual(slotsFor(MIN_CAP, 1), s.slots());
    var seg: [100]u32 = undefined;
    iota(&seg, 1);
    try testing.expectEqual(@as(usize, 100), s.add(&seg));
    const hit = s.lookup(seg[0..30]).?;
    try testing.expectEqual(@as(usize, 30), hit.len);
    var out: [10]u32 = undefined;
    try testing.expectEqual(@as(usize, 10), s.continuation(hit.pos, &out));
    try testing.expectEqualSlices(u32, seg[30..40], &out);
}
test "d_recall: a tail too short for the 16-id window is found by the 8- and 4-id windows" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var seg: [100]u32 = undefined;
    iota(&seg, 1);
    _ = s.add(&seg);
    const h8 = s.lookup(seg[20..30]).?;
    try testing.expectEqual(@as(u64, 30), h8.pos);
    try testing.expectEqual(@as(usize, 10), h8.len);
    const h4 = s.lookup(seg[40..45]).?;
    try testing.expectEqual(@as(u64, 45), h4.pos);
    try testing.expectEqual(@as(usize, 5), h4.len);
    try testing.expectEqual(@as(u64, 2), s.short_hits);
    try testing.expect(s.lookup(seg[40..43]) == null);
}

test "d_recall: a reply that forked from a recorded one is found again four ids after the fork" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    var seg: [200]u32 = undefined;
    iota(&seg, 1);
    _ = s.add(&seg);
    // The live reply agrees up to id 60, says something else once, then rejoins the record at seg[61..].
    var live: [70]u32 = undefined;
    @memcpy(live[0..60], seg[0..60]);
    live[60] = 77777;
    @memcpy(live[61..65], seg[61..65]);
    try testing.expect(s.lookup(live[0..64]) == null);
    const hit = s.lookup(live[0..65]).?;
    try testing.expectEqual(@as(u64, 65), hit.pos);
    try testing.expectEqual(@as(usize, 4), hit.len);
    var out: [8]u32 = undefined;
    try testing.expectEqual(@as(usize, 8), s.continuation(hit.pos, &out));
    try testing.expectEqualSlices(u32, seg[65..73], &out);
}

test "d_recall: a short match lands in its own tier with half the prior" {
    try testing.expectEqual(SHORT_TIER, tierOf(4));
    try testing.expectEqual(SHORT_TIER, tierOf(15));
    try testing.expectEqual(@as(u2, 0), tierOf(16));
    const g = Gate.init(5);
    try testing.expectApproxEqAbs(@as(f64, 2.0), g.expect(SHORT_TIER), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), g.expect(0), 1e-9);
}

test "d_recall: a short match drafts twice its length, never the block or less" {
    var r = try Recall.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer r.deinit();
    var prompt: [8]u32 = undefined;
    iota(&prompt, 100);
    var reply: [60]u32 = undefined;
    iota(&reply, 900);
    r.remember(&prompt, &reply);
    var st: Stream = .{ .joined = true, .own_ema = 0 };
    var out: [MAX_DRAFTS]u32 = undefined;
    // ctx ++ t1 is the reply's first 5 ids: a 5-id agreement through the 4-id window, so 10 drafts.
    const d = r.draft(&st, reply[0..4], reply[4], &out, 3).?;
    try testing.expectEqual(@as(u32, 5), d.len);
    try testing.expectEqual(@as(usize, 10), d.n);
    try testing.expectEqualSlices(u32, reply[5..15], out[0..d.n]);
    try testing.expect(st.pending.?.tier == SHORT_TIER);
    r.feedback(&st, 0, d.n);
    try testing.expect(st.pending == null);
}

test "d_recall: the record's byte count includes every short-window index" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    try testing.expectEqual(s.ids.len * 4 + s.slots() * 16 + shortSlotsFor(s.slots()) * 16 * SHORT_WINDOWS.len, s.bytes());
    try testing.expectEqual(@as(usize, 1) << 24, shortSlotsFor(@as(usize, 1) << 26));
}

test "d_recall: sampled and greedy copies are learnt apart" {
    var g = Gate.init(5);
    var greedy: Stream = .{};
    var sampled: Stream = .{ .sampled = true };
    g.join(&greedy);
    g.join(&sampled);
    greedy.own_ema = 0;
    sampled.own_ema = 0;
    try testing.expect(g.admit(&sampled, 20, 10));
    g.feedback(&sampled, 0);
    try testing.expectApproxEqAbs(@as(f64, 8.0 / 3.0), g.expectFor(true, 0), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), g.expectFor(false, 0), 1e-9);
    try testing.expect(g.admit(&greedy, 20, 10));
    g.feedback(&greedy, 10);
    try testing.expectApproxEqAbs(@as(f64, 6.0), g.expect(0), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 8.0 / 3.0), g.expectFor(true, 0), 1e-9);
}