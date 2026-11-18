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
/// The shortest copy the adaptive gate drafts.
pub const MIN_COPY = 2;
/// A DSpark stage block's draft forward against one plain verify row, by default.
pub const DEFAULT_STAGE_REL: f64 = 0.3;
/// How much more per unit of verify a copy must promise than the stream's own drafts, by default.
pub const DEFAULT_MARGIN: f64 = 0.05;
/// Pseudo-trials behind the per-position acceptance prior.
const RATE_W = 4.0;
/// Per-position acceptance priors: [sampled][tier].
const RATE_PRIOR = [2][4]f64{ .{ 0.9, 0.9, 0.9, 0.8 }, .{ 0.8, 0.8, 0.8, 0.7 } };

/// Verify cost of `rows` rows over one row's (index = rows, 0..MAX_DRAFTS+1): the dflash verify ladder in ms
/// (1:17.7 .. 16:69.1), linear between its points.
pub const REL_COST: [MAX_DRAFTS + 2]f64 = blk: {
    const pts = [_][2]f64{
        .{ 1, 17.7 }, .{ 2, 25.5 }, .{ 3, 29.6 }, .{ 4, 33.6 },  .{ 5, 36.8 },  .{ 6, 42.4 },
        .{ 7, 44.8 }, .{ 8, 47.9 }, .{ 9, 51.1 }, .{ 12, 60.5 }, .{ 16, 69.1 },
    };
    var t: [MAX_DRAFTS + 2]f64 = undefined;
    t[0] = 0;
    for (1..MAX_DRAFTS + 2) |r| {
        const x: f64 = @floatFromInt(r);
        var ms: f64 = pts[pts.len - 1][1];
        for (0..pts.len - 1) |i| {
            if (x >= pts[i][0] and x <= pts[i + 1][0]) {
                ms = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0]);
                break;
            }
        }
        t[r] = ms / pts[0][1];
    }
    break :blk t;
};

/// REL_COST with `rows` clamped to 1..MAX_DRAFTS+1.
pub fn relCost(rows: usize) f64 {
    return REL_COST[std.math.clamp(rows, 1, MAX_DRAFTS + 1)];
}

/// Tokens per unit of verify a copy of `n` drafts earns when each lands with probability `alpha` given the last did:
/// the trunk token plus the expected accepted prefix, over the cost of its n+1 rows.
pub fn copyRate(alpha: f64, n: usize) f64 {
    var e: f64 = 1;
    var p: f64 = 1;
    for (0..n) |_| {
        p *= alpha;
        e += p;
    }
    return e / relCost(n + 1);
}

/// The copy length in lo..hi with the best copyRate; a longer one has to beat the shorter by more than 2%.
pub fn bestCopyLen(alpha: f64, lo: usize, hi: usize) usize {
    var best = lo;
    var rate = copyRate(alpha, lo);
    var n = lo + 1;
    while (n <= hi) : (n += 1) {
        const r = copyRate(alpha, n);
        if (r > rate * 1.02) {
            best = n;
            rate = r;
        }
    }
    return best;
}
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

/// The shortest exact suffix match the suffix automaton answers with (shorter ones fall back to the hash windows).
pub const SAM_MIN = 3;
/// Ids one suffix-automaton generation covers, by default (two generations live: the growing one and the last).
pub const DEFAULT_SAM_IDS: usize = 1 << 20;
/// Prompt ids remembered ahead of a reply once the automaton is on, by default.
pub const DEFAULT_PROMPT_IDS: usize = 2048;
/// Segments whose hit counts the record keeps (a ring), for carrying hot ones across a generation drop.
pub const SEG_TABLE = 16384;
/// Hits a segment needs to be carried into the new generation when its own is dropped.
pub const HOT_HITS = 2;
/// Share of a generation the carried hot segments may fill, by default.
pub const DEFAULT_HOT_FRAC: f64 = 0.25;
const NONE: u32 = std.math.maxInt(u32);
const SAM_LOAD = 0.8;
/// Suffix-link ancestors an extend stamps with its position as their newest end.
const RECENT_WALK = 64;

/// One generation of a suffix automaton over the ring positions [g0, g0 + n): exact longest-suffix matches in
/// O(tail) a query. SENTINEL is extended like any id, so no match crosses a segment's end. Allocated once; an
/// extend never allocates, and a full generation refuses further ids.
pub const Sam = struct {
    gpa: std.mem.Allocator,
    g0: u64,
    n: u32 = 0,
    max_ids: u32,
    /// Per state: longest string length, suffix link, its first end position and its newest one (relative to g0;
    /// the newest is kept for the RECENT_WALK longest suffixes of each new id), first outgoing edge.
    len: []u32,
    link: []u32,
    pos: []u32,
    rpos: []u32,
    edge0: []u32,
    states: u32 = 1,
    last: u32 = 0,
    /// Edges: target, id, next edge of the same state.
    e_to: []u32,
    e_tok: []u32,
    e_next: []u32,
    edges: u32 = 0,
    /// (state << 32 | id) + 1 -> edge (0 = empty), open addressing.
    hkeys: []u64,
    hval: []u32,
    hmask: u64,

    pub fn init(gpa: std.mem.Allocator, g0: u64, max_ids_in: usize) !Sam {
        const max_ids: usize = @max(16, @min(max_ids_in, 1 << 30));
        const ms = 2 * max_ids + 2;
        const me = 3 * max_ids + 4;
        var hs: usize = 1 << 4;
        while (@as(f64, @floatFromInt(hs)) * SAM_LOAD < @as(f64, @floatFromInt(me))) hs <<= 1;
        const len = try gpa.alloc(u32, ms);
        errdefer gpa.free(len);
        const link = try gpa.alloc(u32, ms);
        errdefer gpa.free(link);
        const pos = try gpa.alloc(u32, ms);
        errdefer gpa.free(pos);
        const rpos = try gpa.alloc(u32, ms);
        errdefer gpa.free(rpos);
        const edge0 = try gpa.alloc(u32, ms);
        errdefer gpa.free(edge0);
        const e_to = try gpa.alloc(u32, me);
        errdefer gpa.free(e_to);
        const e_tok = try gpa.alloc(u32, me);
        errdefer gpa.free(e_tok);
        const e_next = try gpa.alloc(u32, me);
        errdefer gpa.free(e_next);
        const hkeys = try gpa.alloc(u64, hs);
        errdefer gpa.free(hkeys);
        const hval = try gpa.alloc(u32, hs);
        var s: Sam = .{
            .gpa = gpa,
            .g0 = g0,
            .max_ids = @intCast(max_ids),
            .len = len,
            .link = link,
            .pos = pos,
            .rpos = rpos,
            .edge0 = edge0,
            .e_to = e_to,
            .e_tok = e_tok,
            .e_next = e_next,
            .hkeys = hkeys,
            .hval = hval,
            .hmask = hs - 1,
        };
        s.reset(g0);
        return s;
    }

    pub fn deinit(self: *Sam) void {
        self.gpa.free(self.len);
        self.gpa.free(self.link);
        self.gpa.free(self.pos);
        self.gpa.free(self.rpos);
        self.gpa.free(self.edge0);
        self.gpa.free(self.e_to);
        self.gpa.free(self.e_tok);
        self.gpa.free(self.e_next);
        self.gpa.free(self.hkeys);
        self.gpa.free(self.hval);
    }

    /// Empties the automaton to cover ids from `g0` on, keeping its memory.
    pub fn reset(self: *Sam, g0: u64) void {
        @memset(self.hkeys, 0);
        self.g0 = g0;
        self.n = 0;
        self.states = 1;
        self.last = 0;
        self.edges = 0;
        self.len[0] = 0;
        self.link[0] = NONE;
        self.pos[0] = 0;
        self.rpos[0] = 0;
        self.edge0[0] = NONE;
    }

    pub fn bytes(self: *const Sam) usize {
        return (self.len.len * 5 + self.e_to.len * 3) * @sizeOf(u32) + self.hkeys.len * (@sizeOf(u64) + @sizeOf(u32));
    }

    pub fn full(self: *const Sam) bool {
        return self.n >= self.max_ids;
    }

    fn find(self: *const Sam, s: u32, tok: u32) u32 {
        const key = ((@as(u64, s) << 32) | tok) + 1;
        var i: usize = @intCast(mix(key) & self.hmask);
        while (true) {
            const k = self.hkeys[i];
            if (k == 0) return NONE;
            if (k == key) return self.hval[i];
            i = @intCast((i + 1) & self.hmask);
        }
    }

    fn next(self: *const Sam, s: u32, tok: u32) u32 {
        const e = self.find(s, tok);
        return if (e == NONE) NONE else self.e_to[e];
    }

    fn addEdge(self: *Sam, s: u32, tok: u32, to: u32) void {
        const e = self.edges;
        self.edges += 1;
        self.e_to[e] = to;
        self.e_tok[e] = tok;
        self.e_next[e] = self.edge0[s];
        self.edge0[s] = e;
        const key = ((@as(u64, s) << 32) | tok) + 1;
        var i: usize = @intCast(mix(key) & self.hmask);
        while (self.hkeys[i] != 0) i = @intCast((i + 1) & self.hmask);
        self.hkeys[i] = key;
        self.hval[i] = e;
    }

    fn degree(self: *const Sam, s: u32) usize {
        var d: usize = 0;
        var e = self.edge0[s];
        while (e != NONE) : (e = self.e_next[e]) d += 1;
        return d;
    }

    /// Appends one id; false (and nothing changed) once the generation is full.
    pub fn extend(self: *Sam, tok: u32) bool {
        if (self.full() or self.states + 2 > self.len.len) return false;
        const cur = self.states;
        self.states += 1;
        self.len[cur] = self.len[self.last] + 1;
        self.pos[cur] = self.n;
        self.rpos[cur] = self.n;
        self.edge0[cur] = NONE;
        self.link[cur] = 0;
        var p = self.last;
        while (p != NONE and self.find(p, tok) == NONE) : (p = self.link[p]) {
            if (self.edges >= self.e_to.len) return self.overflow();
            self.addEdge(p, tok, cur);
        }
        if (p != NONE) {
            const q = self.next(p, tok);
            if (self.len[p] + 1 == self.len[q]) {
                self.link[cur] = q;
            } else {
                if (self.edges + self.degree(q) > self.e_to.len) return self.overflow();
                const clone = self.states;
                self.states += 1;
                self.len[clone] = self.len[p] + 1;
                self.link[clone] = self.link[q];
                self.pos[clone] = self.pos[q];
                self.rpos[clone] = self.rpos[q];
                self.edge0[clone] = NONE;
                var e = self.edge0[q];
                while (e != NONE) : (e = self.e_next[e]) self.addEdge(clone, self.e_tok[e], self.e_to[e]);
                while (p != NONE) : (p = self.link[p]) {
                    const pe = self.find(p, tok);
                    if (pe == NONE or self.e_to[pe] != q) break;
                    self.e_to[pe] = clone;
                }
                self.link[q] = clone;
                self.link[cur] = clone;
            }
        }
        // Every suffix of the new prefix now also ends at n: the longest ones (those a match uses) learn it.
        var a = self.link[cur];
        var k: usize = 0;
        while (a != NONE and a != 0 and k < RECENT_WALK and self.len[a] >= SAM_MIN) : (k += 1) {
            self.rpos[a] = self.n;
            a = self.link[a];
        }
        self.last = cur;
        self.n += 1;
        return true;
    }

    /// Edge room ran out mid-extend (beyond the 3n bound, so not expected): the generation stops taking ids. The
    /// edges already added lead to `cur`, whose end position `n` is the id just written to the ring, so what a
    /// query reads after it is still the ring's own continuation.
    fn overflow(self: *Sam) bool {
        self.max_ids = self.n;
        return false;
    }

    pub const Match = struct { state: u32, len: usize };

    /// The longest suffix of `tail` this generation holds (len 0 = none).
    pub fn longest(self: *const Sam, tail: []const u32) Match {
        var s: u32 = 0;
        var l: usize = 0;
        for (tail) |t| {
            if (t == SENTINEL) {
                s = 0;
                l = 0;
                continue;
            }
            while (true) {
                const nx = self.next(s, t);
                if (nx != NONE) {
                    s = nx;
                    l += 1;
                    break;
                }
                if (s == 0) {
                    l = 0;
                    break;
                }
                s = self.link[s];
                l = self.len[s];
            }
        }
        return .{ .state = s, .len = l };
    }

    /// The ring position right after the newest occurrence of the match's string.
    pub fn after(self: *const Sam, m: Match) u64 {
        return self.g0 + @as(u64, self.rpos[m.state]) + 1;
    }

    /// The ring position right after its first occurrence.
    pub fn afterFirst(self: *const Sam, m: Match) u64 {
        return self.g0 + @as(u64, self.pos[m.state]) + 1;
    }
};

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
    /// The suffix automaton (0 ids = off: the hash windows alone, as before): the growing generation and the last.
    sam_ids: usize = 0,
    sam_cur: ?Sam = null,
    sam_old: ?Sam = null,
    hot_frac: f64 = DEFAULT_HOT_FRAC,
    sam_hits: u64 = 0,
    sam_len_sum: u64 = 0,
    sam_rot: u64 = 0,
    sam_fail: u64 = 0,
    sam_hot_ids: u64 = 0,
    /// The dropped generation's range whose hot segments wait to be carried (after the id being put).
    hot_lo: u64 = 0,
    hot_hi: u64 = 0,
    hot_pending: bool = false,
    /// Segment ring (SEG_TABLE entries, appended in position order): start, end, hits.
    seg_s: []u64 = &.{},
    seg_e: []u64 = &.{},
    seg_h: []u32 = &.{},
    seg_n: u64 = 0,

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
        self.dropSam();
    }

    fn dropSam(self: *Store) void {
        if (self.sam_cur) |*s| s.deinit();
        if (self.sam_old) |*s| s.deinit();
        self.sam_cur = null;
        self.sam_old = null;
        self.sam_ids = 0;
        if (self.seg_s.len != 0) {
            self.gpa.free(self.seg_s);
            self.gpa.free(self.seg_e);
            self.gpa.free(self.seg_h);
        }
        self.seg_s = &.{};
        self.seg_e = &.{};
        self.seg_h = &.{};
        self.seg_n = 0;
    }

    /// Turns the suffix automaton on with `ids` ids a generation, from the record's end on (0 = off).
    pub fn enableSam(self: *Store, ids: usize) !void {
        self.dropSam();
        if (ids == 0) return;
        const ss = try self.gpa.alloc(u64, SEG_TABLE);
        errdefer self.gpa.free(ss);
        const se = try self.gpa.alloc(u64, SEG_TABLE);
        errdefer self.gpa.free(se);
        const sh = try self.gpa.alloc(u32, SEG_TABLE);
        errdefer self.gpa.free(sh);
        self.sam_cur = try Sam.init(self.gpa, self.end, ids);
        self.seg_s = ss;
        self.seg_e = se;
        self.seg_h = sh;
        self.seg_n = 0;
        self.sam_ids = ids;
    }

    fn samPut(self: *Store, g: u64, t: u32) void {
        if (self.sam_ids == 0) return;
        if (self.sam_cur) |*s| {
            if (s.extend(t)) return;
        } else return;
        self.rotate(g);
        if (self.sam_cur) |*s| _ = s.extend(t);
    }

    /// The growing generation is full: it becomes the last one, the last one's memory starts a new generation at
    /// `g`, and the dropped generation's hot segments are marked to be carried.
    fn rotate(self: *Store, g: u64) void {
        self.sam_rot += 1;
        var spare: ?Sam = null;
        if (self.sam_old) |o| {
            self.hot_lo = o.g0;
            self.hot_hi = o.g0 + o.n;
            self.hot_pending = true;
            spare = o;
        }
        self.sam_old = self.sam_cur;
        if (spare) |*sp| {
            sp.reset(g);
            self.sam_cur = sp.*;
        } else {
            self.sam_cur = Sam.init(self.gpa, g, self.sam_ids) catch blk: {
                self.sam_fail += 1;
                break :blk null;
            };
        }
    }

    fn segAdd(self: *Store, s: u64, e: u64, hits: u32) void {
        if (self.seg_s.len == 0) return;
        const k: usize = @intCast(self.seg_n % SEG_TABLE);
        self.seg_s[k] = s;
        self.seg_e[k] = e;
        self.seg_h[k] = hits;
        self.seg_n += 1;
    }

    /// The segment-ring entry holding position `g` (a continuation start: start < g <= end), if still kept.
    fn segFind(self: *const Store, g: u64) ?usize {
        if (self.seg_n == 0) return null;
        const kept: u64 = @min(self.seg_n, SEG_TABLE);
        var lo: u64 = self.seg_n - kept;
        var hi: u64 = self.seg_n;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.seg_s[@intCast(mid % SEG_TABLE)] < g) lo = mid + 1 else hi = mid;
        }
        if (lo == self.seg_n - kept) return null;
        const k: usize = @intCast((lo - 1) % SEG_TABLE);
        return if (g <= self.seg_e[k]) k else null;
    }

    fn segHit(self: *Store, g: u64) void {
        if (self.segFind(g)) |k| self.seg_h[k] +|= 1;
    }

    /// Appends the dropped generation's segments hit at least HOT_HITS times again, hottest first, up to
    /// `hot_frac` of a generation; a carried segment keeps half its hits.
    fn carryHot(self: *Store) void {
        self.hot_pending = false;
        if (self.seg_n == 0 or self.sam_ids == 0) return;
        const kept: u64 = @min(self.seg_n, SEG_TABLE);
        const picks = self.gpa.alloc(usize, @intCast(kept)) catch return;
        defer self.gpa.free(picks);
        var np: usize = 0;
        var i: u64 = self.seg_n - kept;
        while (i < self.seg_n) : (i += 1) {
            const k: usize = @intCast(i % SEG_TABLE);
            if (self.seg_h[k] >= HOT_HITS and self.seg_s[k] >= self.hot_lo and self.seg_e[k] <= self.hot_hi and
                self.seg_s[k] >= self.base())
            {
                picks[np] = k;
                np += 1;
            }
        }
        const Ctx = struct {
            h: []const u32,
            fn more(c: @This(), a: usize, b: usize) bool {
                return c.h[a] > c.h[b];
            }
        };
        std.sort.pdq(usize, picks[0..np], Ctx{ .h = self.seg_h }, Ctx.more);
        const budget: u64 = @intFromFloat(@as(f64, @floatFromInt(self.sam_ids)) * self.hot_frac);
        var used: u64 = 0;
        // Copy every pick out first: appending moves the segment ring under the picked entries.
        const Pick = struct { s: u64, e: u64, h: u32 };
        const todo = self.gpa.alloc(Pick, np) catch return;
        defer self.gpa.free(todo);
        var nt: usize = 0;
        for (picks[0..np]) |k| {
            const len = self.seg_e[k] - self.seg_s[k];
            if (len == 0 or used + len > budget) continue;
            used += len;
            todo[nt] = .{ .s = self.seg_s[k], .e = self.seg_e[k], .h = self.seg_h[k] };
            nt += 1;
        }
        for (todo[0..nt]) |p| {
            if (p.s < self.base()) continue;
            const n: usize = @intCast(p.e - p.s);
            const buf = self.gpa.alloc(u32, n) catch return;
            defer self.gpa.free(buf);
            for (buf, 0..) |*x, j| x.* = self.at(p.s + @as(u64, j));
            const w = self.appendSegmentRaw(buf, p.h / 2);
            self.sam_hot_ids += w;
            self.added += w;
        }
    }

    /// Exact longest suffix of `tail` in either generation (the growing one on a tie), at least SAM_MIN ids.
    fn samLookup(self: *const Store, tail: []const u32) ?Hit {
        var best: ?Hit = null;
        const b = self.base();
        for ([_]?Sam{ self.sam_cur, self.sam_old }) |opt| {
            const s = opt orelse continue;
            if (s.g0 < b or s.n == 0) continue;
            const m = s.longest(tail);
            if (m.len < SAM_MIN) continue;
            if (best) |h| {
                if (m.len <= h.len) continue;
            }
            var p = s.after(m);
            // The newest occurrence may end a segment (its continuation is empty): take the first one then.
            if (p >= self.end or self.at(p) == SENTINEL) p = s.afterFirst(m);
            if (p < b or p > self.end) continue;
            best = .{ .pos = p, .len = @min(m.len, MAX_BACK) };
        }
        return best;
    }

    pub fn bytes(self: *const Store) usize {
        var b = self.ids.len * @sizeOf(u32) + self.keys.len * @sizeOf(u64) * 2;
        for (self.short) |si| b += si.keys.len * @sizeOf(u64) * 2;
        if (self.sam_cur) |s| b += s.bytes();
        if (self.sam_old) |s| b += s.bytes();
        b += self.seg_s.len * (2 * @sizeOf(u64) + @sizeOf(u32));
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
        const g = self.end;
        self.ids[@intCast(g % self.cap)] = t;
        self.end += 1;
        self.samPut(g, t);
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
        const w = self.appendSegmentRaw(seg_in, 0);
        if (self.hot_pending) self.carryHot();
        return w;
    }

    fn appendSegmentRaw(self: *Store, seg_in: []const u32, hits: u32) usize {
        const room: usize = @intCast(self.cap / 2);
        const seg = if (seg_in.len > room) seg_in[seg_in.len - room ..] else seg_in;
        if (self.end > self.base() and self.at(self.end - 1) != SENTINEL) self.put(SENTINEL);
        const start = self.end;
        for (seg) |t| self.put(t);
        self.indexRange(start, self.end);
        self.segAdd(start, self.end, hits);
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
        const min_tail: usize = if (self.sam_ids != 0) SAM_MIN else SHORT_WINDOWS[SHORT_WINDOWS.len - 1];
        if (tail.len < min_tail or self.held() < WINDOW) return null;
        self.lookups += 1;
        if (self.sam_ids != 0) {
            if (self.samLookup(tail)) |hit| {
                self.hits += 1;
                self.sam_hits += 1;
                self.sam_len_sum += hit.len;
                if (hit.len < WINDOW) self.short_hits += 1;
                self.segHit(hit.pos);
                return hit;
            }
        }
        if (self.lookupW(self.keys, self.pos, self.mask, WINDOW, tail)) |hit| {
            self.hits += 1;
            self.segHit(hit.pos);
            return hit;
        }
        for (self.short) |si| {
            if (self.lookupW(si.keys, si.pos, si.mask, si.w, tail)) |hit| {
                self.hits += 1;
                self.short_hits += 1;
                self.segHit(hit.pos);
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

    /// Every distinct place the record continues `tail` from, best first, no counter moved: the automata's newest
    /// and first occurrences of their longest match, then the 16-, 8- and 4-id windows'.
    pub fn candidates(self: *const Store, tail: []const u32, out: []Hit) usize {
        @setRuntimeSafety(true);
        const min_tail: usize = if (self.sam_ids != 0) SAM_MIN else SHORT_WINDOWS[SHORT_WINDOWS.len - 1];
        if (tail.len < min_tail or self.held() < WINDOW or out.len == 0) return 0;
        var n: usize = 0;
        const b = self.base();
        if (self.sam_ids != 0) for ([_]?Sam{ self.sam_cur, self.sam_old }) |opt| {
            const s = opt orelse continue;
            if (s.g0 < b or s.n == 0) continue;
            const m = s.longest(tail);
            if (m.len < SAM_MIN) continue;
            const len: usize = @min(m.len, MAX_BACK);
            for ([_]u64{ s.after(m), s.afterFirst(m) }) |p| {
                if (p < b or p >= self.end or self.at(p) == SENTINEL) continue;
                n = pushHit(out, n, .{ .pos = p, .len = len });
            }
        };
        if (self.lookupW(self.keys, self.pos, self.mask, WINDOW, tail)) |h| n = pushHit(out, n, h);
        for (self.short) |si| {
            if (self.lookupW(si.keys, si.pos, si.mask, si.w, tail)) |h| n = pushHit(out, n, h);
        }
        return n;
    }
};

fn pushHit(out: []Store.Hit, n: usize, h: Store.Hit) usize {
    if (n >= out.len) return n;
    for (out[0..n]) |o| if (o.pos == h.pos) return n;
    out[n] = h;
    return n + 1;
}

const Stat = struct { n: f64 = 0, s: f64 = 0 };

const Pending = struct { tier: u2, probe: bool, real: u32, own: f64, sampled: bool = false, own_rate: f64 = 0 };

const Rate = struct { acc: f64 = 0, tri: f64 = 0 };

/// One request's side of the gate.
pub const Stream = struct {
    /// Set when the request's prompt was recorded; `end` records its reply once.
    armed: bool = false,
    /// A sampled request: its copies land p(x) of the time, so the gate learns them apart from greedy ones.
    sampled: bool = false,
    joined: bool = false,
    /// Accepted drafts per stage round, smoothed: the bar a recalled copy must beat.
    own_ema: f64 = 0,
    /// Drafts the stream's own stage rounds verify, smoothed (adaptive gate).
    own_rows_ema: f64 = 0,
    cooldown: u8 = 0,
    pending: ?Pending = null,
    offers: u32 = 0,
    probes: u32 = 0,
    drafted: u64 = 0,
    landed: u64 = 0,
    /// Tree rounds (`choosePlan`): rounds, verify rows, branches, rounds a recalled branch won past the stage draft,
    /// tokens it won past it, and rounds since a tree with a recalled branch last ran.
    tree_rounds: u32 = 0,
    tree_rows: u64 = 0,
    tree_branches: u64 = 0,
    tree_wins: u32 = 0,
    tree_extra: u64 = 0,
    tree_idle: u32 = 0,
    /// Cold layer (`d_recall_cold.zig`): the stream's id there, whether the copy on offer came from it, and the
    /// context length its last lookups were posted at.
    cold_sid: u32 = 0,
    cold_from: bool = false,
    cold_posted: usize = 0,
    /// Yee KAD (kad_live.zig): this request's ladder state; null until its first verified round.
    kad: ?@import("kad.zig").State = null,

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
    /// own_global for sampled streams (adaptive gate).
    own_global_s: f64 = 0,
    gamma: f64 = 0,
    /// Per-position acceptance of copied drafts, [sampled][tier] (adaptive gate).
    rates: [2][4]Rate = .{ .{ .{}, .{}, .{}, .{} }, .{ .{}, .{}, .{}, .{} } },
    /// Off: the gate compares landed tokens per offer with the stream's own drafts, as it did before.
    adaptive: bool = false,
    stage_rel: f64 = DEFAULT_STAGE_REL,
    margin: f64 = DEFAULT_MARGIN,

    pub fn init(gamma: usize) Gate {
        const g: f64 = @floatFromInt(gamma);
        return .{ .prior = PRIOR_FRAC * g, .own_global = 0.5 * g, .own_global_s = 0.5 * g, .gamma = g };
    }

    pub fn expect(self: *const Gate, tier: u2) f64 {
        return self.expectFor(false, tier);
    }

    pub fn expectFor(self: *const Gate, sampled: bool, tier: u2) f64 {
        const s = if (sampled) self.stats_sampled[tier] else self.stats[tier];
        const prior = if (tier == SHORT_TIER) self.prior * SHORT_PRIOR else self.prior;
        return (s.s + prior * PRIOR_W) / (s.n + PRIOR_W);
    }

    /// The chance a copied draft of `tier` lands given the one before it did.
    pub fn alpha(self: *const Gate, sampled: bool, tier: u2) f64 {
        const i: usize = @intFromBool(sampled);
        const r = self.rates[i][tier];
        const a = (r.acc + RATE_PRIOR[i][tier] * RATE_W) / (r.tri + RATE_W);
        return std.math.clamp(a, 0.01, 0.995);
    }

    /// Tokens per unit of verify the stream's own stage rounds earn, their draft forward included.
    pub fn ownRate(self: *const Gate, st: *const Stream) f64 {
        const rows: usize = @intFromFloat(@round(std.math.clamp(st.own_rows_ema, 0, MAX_DRAFTS)));
        return (1 + st.own_ema) / (relCost(1 + rows) + self.stage_rel);
    }

    fn join(self: *const Gate, st: *Stream) void {
        if (st.joined) return;
        st.joined = true;
        st.own_ema = if (self.adaptive and st.sampled) self.own_global_s else self.own_global;
        st.own_rows_ema = self.gamma;
    }

    /// Lets a copy through when it pays, else one ask in `probe_iv` as a probe.
    fn pass(self: *Gate, st: *Stream, tier: u2, real: usize, pays: bool, own_rate: f64) bool {
        var probe = false;
        if (!pays) {
            const left = self.probe_left -| 1;
            if (left > 0) {
                self.probe_left = left;
                return false;
            }
            self.probe_left = self.probe_iv;
            probe = true;
            st.probes += 1;
        }
        st.pending = .{ .tier = tier, .probe = probe, .real = @intCast(real), .own = st.own_ema, .sampled = st.sampled, .own_rate = own_rate };
        st.offers += 1;
        return true;
    }

    /// Offers a copy of `real` ids matched over `len`: while its tier does not beat the stream's own drafts, only
    /// one ask in `probe_iv` goes through, as a probe.
    pub fn admit(self: *Gate, st: *Stream, len: usize, real: usize) bool {
        const tier = tierOf(len);
        return self.pass(st, tier, real, self.expectFor(st.sampled, tier) > st.own_ema, 0);
    }

    /// The adaptive admit: a copy of `n` ids matched over `len` pays when its tokens per unit of verify beat the
    /// stream's own by `margin`.
    pub fn admitRate(self: *Gate, st: *Stream, len: usize, n: usize) bool {
        const tier = tierOf(len);
        const own = self.ownRate(st);
        const pays = copyRate(self.alpha(st.sampled, tier), n) > own * (1 + self.margin);
        return self.pass(st, tier, n, pays, own);
    }

    /// What the target accepted of the offered copy, all of it verified.
    pub fn feedback(self: *Gate, st: *Stream, accepted: u32) void {
        const p = st.pending orelse return;
        self.feedbackRows(st, accepted, p.real);
    }

    /// What the target accepted of the offered copy, `verified` ids of it verified.
    pub fn feedbackRows(self: *Gate, st: *Stream, accepted: u32, verified: usize) void {
        const p = st.pending orelse return;
        st.pending = null;
        const landed = @min(accepted, p.real);
        const lf: f64 = @floatFromInt(landed);
        const s = if (p.sampled) &self.stats_sampled[p.tier] else &self.stats[p.tier];
        s.s = s.s * DECAY + lf;
        s.n = s.n * DECAY + 1;
        const lv: usize = @min(@as(usize, landed), verified);
        const r = &self.rates[@intFromBool(p.sampled)][p.tier];
        const lvf: f64 = @floatFromInt(lv);
        r.acc = r.acc * DECAY + lvf;
        r.tri = r.tri * DECAY + lvf + @as(f64, if (lv < verified) 1 else 0);
        st.landed += landed;
        if (landed == 0) st.cooldown = COOLDOWN;
        if (p.probe) {
            const won = if (self.adaptive) (1 + lf) / relCost(1 + verified) > p.own_rate else lf > p.own;
            self.probe_iv = if (won) PROBE_START else @min(self.probe_iv * 2, PROBE_MAX);
            self.probe_left = self.probe_iv;
        }
    }

    /// A stage round of the stream: `accepted` of its `rows` drafts. Sampled rounds count only in the adaptive gate.
    pub fn observeOwn(self: *Gate, st: *Stream, accepted: u32, rows: usize) void {
        if (st.sampled and !self.adaptive) return;
        self.join(st);
        const a: f64 = @floatFromInt(accepted);
        st.own_ema = (1 - OWN_ALPHA) * st.own_ema + OWN_ALPHA * a;
        const rf: f64 = @floatFromInt(rows);
        st.own_rows_ema = (1 - OWN_ALPHA) * st.own_rows_ema + OWN_ALPHA * rf;
        if (st.sampled) {
            self.own_global_s = 0.99 * self.own_global_s + 0.01 * a;
        } else {
            self.own_global = 0.99 * self.own_global + 0.01 * a;
        }
    }

    /// A recalled branch of a tree round: `landed` of its ids kept while it stayed on the path, `verified` of them
    /// tested (the first miss included). Teaches its tier as an offered copy does, without the probe schedule.
    pub fn learnBranch(self: *Gate, sampled: bool, tier: u2, landed: u32, verified: usize) void {
        if (verified == 0) return;
        const lf: f64 = @floatFromInt(landed);
        const s = if (sampled) &self.stats_sampled[tier] else &self.stats[tier];
        s.s = s.s * DECAY + lf;
        s.n = s.n * DECAY + 1;
        const lv: usize = @min(@as(usize, landed), verified);
        const r = &self.rates[@intFromBool(sampled)][tier];
        const lvf: f64 = @floatFromInt(lv);
        r.acc = r.acc * DECAY + lvf;
        r.tri = r.tri * DECAY + lvf + @as(f64, if (lv < verified) 1 else 0);
    }
};

pub const Draft = struct { n: usize, len: u32 };

/// The record and its gate. Not thread-safe: one caller at a time.
pub const Recall = struct {
    store: Store,
    gate: ?Gate = null,
    adaptive: bool = false,
    stage_rel: f64 = DEFAULT_STAGE_REL,
    margin: f64 = DEFAULT_MARGIN,
    /// Prompt ids remembered ahead of a reply.
    prompt_ids: usize = PROMPT_TAIL,

    pub fn init(gpa: std.mem.Allocator, cap_ids: usize, density: f32) !Recall {
        return .{ .store = try Store.init(gpa, cap_ids, density) };
    }

    pub fn deinit(self: *Recall) void {
        self.store.deinit();
    }

    /// The gate, built on first use with the stage block as its draft length.
    pub fn gateFor(self: *Recall, gamma: usize) *Gate {
        if (self.gate == null) {
            self.gate = Gate.init(gamma);
            self.gate.?.adaptive = self.adaptive;
            self.gate.?.stage_rel = self.stage_rel;
            self.gate.?.margin = self.margin;
        }
        return &self.gate.?;
    }

    /// A recalled continuation of `ctx ++ [t1]` in `out`, more than `block` ids of it, when the gate pays for one.
    pub fn draft(self: *Recall, st: *Stream, ctx: []const u32, t1: u32, out: []u32, block: usize) ?Draft {
        const g = self.gateFor(block);
        g.join(st);
        if (g.adaptive) return self.draftAdaptive(g, st, ctx, t1, out);
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

    /// The adaptive draft: as long a copy as its tier's per-position acceptance pays for in tokens per unit of
    /// verify (MIN_COPY up, a short match at most twice its length), offered when that beats the stream's own drafts.
    fn draftAdaptive(self: *Recall, g: *Gate, st: *Stream, ctx: []const u32, t1: u32, out: []u32) ?Draft {
        if (out.len < MIN_COPY or !st.ready()) return null;
        var tail: [MAX_BACK]u32 = undefined;
        const k: usize = @min(ctx.len, MAX_BACK - 1);
        @memcpy(tail[0..k], ctx[ctx.len - k ..]);
        tail[k] = t1;
        const hit = self.store.lookup(tail[0 .. k + 1]) orelse return null;
        var avail = self.store.continuation(hit.pos, out);
        if (hit.len < WINDOW) avail = @min(avail, 2 * hit.len);
        if (avail < MIN_COPY) return null;
        const n = bestCopyLen(g.alpha(st.sampled, tierOf(hit.len)), MIN_COPY, avail);
        if (!g.admitRate(st, hit.len, n)) return null;
        st.drafted += n;
        return .{ .n = n, .len = @intCast(hit.len) };
    }

    /// The adaptive gate on (`stage_rel` and `margin` as given), or the old one.
    pub fn setAdaptive(self: *Recall, on: bool, stage_rel: f64, margin: f64) void {
        self.adaptive = on;
        self.stage_rel = stage_rel;
        self.margin = margin;
        if (self.gate) |*g| {
            g.adaptive = on;
            g.stage_rel = stage_rel;
            g.margin = margin;
        }
    }

    pub fn alphaOf(self: *const Recall, sampled: bool, tier: u2) f64 {
        const g = self.gate orelse return RATE_PRIOR[@intFromBool(sampled)][tier];
        return g.alpha(sampled, tier);
    }

    pub fn ownRateOf(self: *const Recall, st: *const Stream) f64 {
        const g = self.gate orelse return 0;
        return g.ownRate(st);
    }

    /// `verified` 0 = the round verified none of the offered copy: it teaches nothing.
    pub fn feedback(self: *Recall, st: *Stream, accepted: u32, verified: usize) void {
        if (verified == 0) {
            st.pending = null;
            return;
        }
        if (self.gate) |*g| g.feedbackRows(st, accepted, verified);
    }

    pub fn observeOwn(self: *Recall, st: *Stream, accepted: u32, rows: usize, gamma: usize) void {
        self.gateFor(gamma).observeOwn(st, accepted, rows);
    }

    /// The suffix automaton on with `sam_ids` ids a generation (0 = off), and `prompt_ids` prompt ids remembered
    /// ahead of each reply. On an allocation failure the automaton stays off and the prompt length is still set.
    pub fn setSam(self: *Recall, sam_ids: usize, prompt_ids: usize) !void {
        self.prompt_ids = prompt_ids;
        try self.store.enableSam(sam_ids);
    }

    /// Records a finished reply behind the last `prompt_ids` ids of its prompt.
    pub fn remember(self: *Recall, prompt: []const u32, output: []const u32) void {
        if (output.len == 0) return;
        const head = prompt[prompt.len -| self.prompt_ids..];
        const seg = self.store.gpa.alloc(u32, head.len + output.len) catch return;
        defer self.store.gpa.free(seg);
        @memcpy(seg[0..head.len], head);
        @memcpy(seg[head.len..], output);
        _ = self.store.add(seg);
    }

    /// Up to `cands.len` distinct recalled continuations of `ctx ++ [t1]` for a tree round, best match first, each in
    /// `outs[k][0..cands[k].n]` (at most `room` ids, a short match at most twice its length) and as long as its tier's
    /// per-position acceptance pays (`bestCopyLen`). One continuation that starts another counts once, the longer.
    /// Moves no gate state but the stream's join, and none of the store's counters. 0 while the stream cools down.
    pub fn drafts(self: *Recall, st: *Stream, ctx: []const u32, t1: u32, outs: [][MAX_DRAFTS]u32, room_in: usize, block: usize, cands: []Cand) usize {
        @setRuntimeSafety(true);
        const g = self.gateFor(block);
        g.join(st);
        const room: usize = @min(room_in, MAX_DRAFTS);
        if (room < MIN_COPY or st.cooldown != 0 or cands.len == 0) return 0;
        var tail: [MAX_BACK]u32 = undefined;
        const k: usize = @min(ctx.len, MAX_BACK - 1);
        @memcpy(tail[0..k], ctx[ctx.len - k ..]);
        tail[k] = t1;
        var hits: [8]Store.Hit = undefined;
        const nh = self.store.candidates(tail[0 .. k + 1], &hits);
        var n: usize = 0;
        for (hits[0..nh]) |h| {
            if (n >= @min(cands.len, outs.len)) break;
            var avail = self.store.continuation(h.pos, outs[n][0..room]);
            if (h.len < WINDOW) avail = @min(avail, 2 * h.len);
            if (avail < MIN_COPY) continue;
            const tier = tierOf(h.len);
            const a = g.alpha(st.sampled, tier);
            const len = bestCopyLen(a, MIN_COPY, avail);
            var dup = false;
            for (cands[0..n], 0..) |*c, j| {
                const m = @min(c.n, len);
                if (!std.mem.eql(u32, outs[j][0..m], outs[n][0..m])) continue;
                dup = true;
                if (len > c.n) {
                    @memcpy(outs[j][0..len], outs[n][0..len]);
                    c.n = len;
                }
                if (a > c.alpha) c.* = .{ .n = c.n, .len = @intCast(h.len), .tier = tier, .alpha = a };
                break;
            }
            if (dup) continue;
            cands[n] = .{ .n = len, .len = @intCast(h.len), .tier = tier, .alpha = a };
            n += 1;
        }
        return n;
    }

    /// Whether `c` alone, verified with no stage draft, beats the stream's own drafts per unit of verify (the adaptive
    /// gate's test): such a copy goes as the plain copy round it was before tree rounds.
    pub fn copyPays(self: *const Recall, st: *const Stream, c: Cand) bool {
        const g = self.gate orelse return false;
        return copyRate(c.alpha, c.n) > g.ownRate(st) * (1 + g.margin);
    }

    /// What a tree round's recalled branch `c` did (`Walk.landed` / `Walk.verified`); `won` = its ids are the ones
    /// the round kept past what the stage draft would have.
    pub fn learnBranch(self: *Recall, st: *Stream, c: Cand, landed: u32, verified: usize, won: u32) void {
        if (self.gate) |*g| g.learnBranch(st.sampled, c.tier, landed, verified);
        st.drafted += verified;
        st.landed += won;
    }
};

/// A recalled continuation offered to a tree round.
pub const Cand = struct { n: usize, len: u32, tier: u2, alpha: f64, cold: bool = false };

/// Recalled candidates a tree round looks at.
pub const MAX_CANDS = 4;
/// Branches one tree round verifies: the stage draft and up to three recalled ones.
pub const MAX_BRANCHES = 4;
/// Rows one tree round verifies across its branches: the batched verify's compiled width (`region_rows`).
pub const TREE_ROWS = 16;
pub const DEFAULT_TREE_STATE_COST: f64 = 0.08;
pub const DEFAULT_TREE_K = 2;
pub const DEFAULT_TREE_MARGIN: f64 = 0.05;
/// Tree rounds without a recalled branch after which one goes anyway, so its tier keeps learning.
pub const TREE_PROBE = 32;

/// One branch of a tree round: its drafts after `t1`, and the chance each is accepted given the ones before it were.
pub const Branch = struct { toks: []const u32, p: []const f64 };

/// Which branches a tree round verifies and how far: `lens[b]` drafts of branch b (0 = left out).
pub const Plan = struct {
    lens: [MAX_BRANCHES]usize = @splat(0),
    branches: usize = 0,
    rows: usize = 1,
    tokens: f64 = 1,
    rate: f64 = 0,

    pub fn has(self: *const Plan, b: usize) bool {
        return self.lens[b] > 0;
    }
};

pub const TreeCost = struct {
    /// Each branch state past the first, relative to one verify row.
    state: f64 = DEFAULT_TREE_STATE_COST,
    /// The stage draft forward, relative to one verify row (paid by every plan of a round that drafted).
    stage: f64 = DEFAULT_STAGE_REL,
    /// Rows the round verifies at most, every branch's `t1` row included.
    rows: usize = TREE_ROWS,
    /// Drafts the round may keep (the token budget).
    cap: usize = MAX_DRAFTS,
};

/// Expected tokens a round emits (the trunk token and the accepted drafts) when every branch b with `lens[b] > 0`
/// is verified to that cut. The branches merge into one token trie: a node shared by several branches is one event
/// (its chance the best branch estimate), siblings are exclusive (one target pick per row), so the chance of a node is
/// its parent's times its own, siblings scaled down when their chances sum past 1. Drafts past `cap` count nothing.
pub fn treeTokens(branches: []const Branch, lens: []const usize, cap: usize) f64 {
    @setRuntimeSafety(true);
    const Node = struct { parent: usize, tok: u32, q: f64, depth: usize };
    const ROOT = std.math.maxInt(usize);
    var nodes: [MAX_BRANCHES * MAX_DRAFTS]Node = undefined;
    var nn: usize = 0;
    for (branches, lens) |br, len_in| {
        const len: usize = @min(len_in, br.toks.len, br.p.len, MAX_DRAFTS);
        var cur: usize = ROOT;
        for (0..len) |d| {
            const q = std.math.clamp(if (std.math.isNan(br.p[d])) 0 else br.p[d], 0, 1);
            var found: ?usize = null;
            for (nodes[0..nn], 0..) |nd, i| if (nd.parent == cur and nd.tok == br.toks[d]) {
                found = i;
                break;
            };
            if (found) |i| {
                nodes[i].q = @max(nodes[i].q, q);
                cur = i;
            } else {
                if (nn == nodes.len) break;
                nodes[nn] = .{ .parent = cur, .tok = br.toks[d], .q = q, .depth = d };
                cur = nn;
                nn += 1;
            }
        }
    }
    var prob: [MAX_BRANCHES * MAX_DRAFTS]f64 = undefined;
    var e: f64 = 1;
    for (nodes[0..nn], 0..) |nd, i| {
        var sum: f64 = 0;
        for (nodes[0..nn]) |o| {
            if (o.parent == nd.parent) sum += o.q;
        }
        const scale: f64 = if (sum > 1) 1 / sum else 1;
        const pp: f64 = if (nd.parent == ROOT) 1 else prob[nd.parent];
        prob[i] = pp * nd.q * scale;
        if (nd.depth < cap) e += prob[i];
    }
    return e;
}

/// The verify cost of a plan: its rows on the verify ladder, each branch state past the first, the stage draft.
pub fn treeCostOf(rows: usize, branches: usize, cost: TreeCost) f64 {
    const extra: f64 = @floatFromInt(branches -| 1);
    return relCost(rows) + extra * cost.state + cost.stage;
}

/// Cuts a recalled branch of `n` drafts is tried at.
fn cutsOf(n: usize, out: *[4]usize) []const usize {
    var k: usize = 0;
    for ([_]usize{ n, (3 * n + 3) / 4, (n + 1) / 2, MIN_COPY }) |c| {
        if (c == 0 or c > n) continue;
        if (std.mem.indexOfScalar(usize, out[0..k], c) != null) continue;
        out[k] = c;
        k += 1;
    }
    return out[0..k];
}

fn evalPlan(branches: []const Branch, lens: []const usize, cost: TreeCost) ?Plan {
    var p: Plan = .{};
    var rows: usize = 0;
    for (lens, 0..) |l, b| {
        p.lens[b] = l;
        if (l == 0) continue;
        p.branches += 1;
        rows += l + 1;
    }
    if (p.branches == 0) rows = 1;
    if (rows > cost.rows) return null;
    p.rows = rows;
    p.tokens = treeTokens(branches, lens, cost.cap);
    p.rate = p.tokens / treeCostOf(rows, p.branches, cost);
    return p;
}

pub const Choice = struct {
    /// The stage draft alone at its best cut (0 = the trunk token alone).
    own: Plan,
    /// The best plan with at least one recalled branch (`branches == 0`: none fits).
    mixed: Plan,
};

/// Branch 0 is the stage draft (any cut, 0 = left out), branches 1.. the recalled candidates, at most `max_recall` of
/// them in a plan, each at a few cuts; every plan within `cost.rows`. Picks by expected tokens per unit of verify.
pub fn choosePlan(branches: []const Branch, max_recall: usize, cost: TreeCost) Choice {
    @setRuntimeSafety(true);
    std.debug.assert(branches.len >= 1 and branches.len <= MAX_BRANCHES);
    var ch: Choice = .{ .own = .{}, .mixed = .{} };
    var lens: [MAX_BRANCHES]usize = @splat(0);
    const nb = branches.len;
    const own_max = @min(branches[0].toks.len, branches[0].p.len, cost.rows - 1);
    for (0..own_max + 1) |l0| {
        lens[0] = l0;
        if (evalPlan(branches, lens[0..nb], cost)) |p| {
            if (p.rate > ch.own.rate) ch.own = p;
        }
    }
    if (max_recall == 0 or nb < 2) return ch;
    // Every subset of the recalled branches of size 1..max_recall, by bitmask.
    const nr = nb - 1;
    var mask: usize = 1;
    while (mask < (@as(usize, 1) << @intCast(nr))) : (mask += 1) {
        if (@popCount(mask) > max_recall) continue;
        var members: [MAX_BRANCHES]usize = undefined;
        var nm: usize = 0;
        for (0..nr) |r| if (mask & (@as(usize, 1) << @intCast(r)) != 0) {
            members[nm] = r + 1;
            nm += 1;
        };
        var cut_bufs: [MAX_BRANCHES][4]usize = undefined;
        var cuts: [MAX_BRANCHES][]const usize = undefined;
        for (members[0..nm], 0..) |b, i| cuts[i] = cutsOf(@min(branches[b].toks.len, branches[b].p.len), &cut_bufs[i]);
        var pick: [MAX_BRANCHES]usize = @splat(0);
        outer: while (true) {
            @memset(lens[0..nb], 0);
            for (members[0..nm], 0..) |b, i| lens[b] = cuts[i][pick[i]];
            for (0..own_max + 1) |l0| {
                lens[0] = l0;
                if (evalPlan(branches, lens[0..nb], cost)) |p| {
                    if (p.rate > ch.mixed.rate) ch.mixed = p;
                }
            }
            var i: usize = 0;
            while (i < nm) : (i += 1) {
                pick[i] += 1;
                if (pick[i] < cuts[i].len) continue :outer;
                pick[i] = 0;
            }
            break;
        }
    }
    return ch;
}

/// A tree round's accept over its branches' drafts `toks[b]` and the target's pick at each of their rows `ys[b]`
/// (`toks[b].len + 1` rows: after `t1`, after each draft): the argmax for a greedy request, one independent draw from
/// the request's filtered density for a sampled one. Depth by depth the lowest live branch's pick is the next token;
/// the branches whose draft there equals it live on. Every draft is a point mass, so the kept tokens are the target's
/// own picks in order: a sampled tree is lossless as one chain is, and one branch accepts what one chain does.
pub const Walk = struct {
    accepted: usize,
    next: u32,
    winner: usize,
    /// Per branch: drafts kept while it stayed live, drafts tested.
    landed: [MAX_BRANCHES]u32 = @splat(0),
    verified: [MAX_BRANCHES]usize = @splat(0),
};

pub fn walkTree(toks: []const []const u32, ys: []const []const u32, cap: usize) Walk {
    @setRuntimeSafety(true);
    std.debug.assert(toks.len >= 1 and toks.len <= MAX_BRANCHES and ys.len == toks.len);
    var w: Walk = .{ .accepted = 0, .next = 0, .winner = 0 };
    var alive: [MAX_BRANCHES]bool = @splat(false);
    for (toks, 0..) |_, b| alive[b] = true;
    var depth: usize = 0;
    while (true) {
        var rep: usize = 0;
        while (!alive[rep]) rep += 1;
        const y = ys[rep][depth];
        w.winner = rep;
        w.next = y;
        w.accepted = depth;
        if (depth >= cap) return w;
        var any = false;
        for (toks, 0..) |t, b| {
            if (!alive[b]) continue;
            if (depth >= t.len) {
                alive[b] = false;
                continue;
            }
            w.verified[b] += 1;
            if (t[depth] == y) {
                w.landed[b] += 1;
                any = true;
            } else alive[b] = false;
        }
        if (!any) return w;
        depth += 1;
    }
}

// ── tests ──

const testing = std.testing;

test {
    _ = @import("d_recall_cold.zig");
    _ = @import("kad.zig");
    _ = @import("kad_stdio.zig");
    _ = @import("kad_live.zig");
}

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

test "d_recall: the relative verify cost is 1 at one row and follows the ladder between its points" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), REL_COST[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 69.1 / 17.7), REL_COST[16], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 51.1 / 17.7), REL_COST[9], 1e-12);
    // 10 sits a third of the way from 9 (51.1) to 12 (60.5).
    try testing.expectApproxEqAbs(@as(f64, (51.1 + 9.4 / 3.0) / 17.7), REL_COST[10], 1e-12);
    try testing.expectApproxEqAbs(REL_COST[16], relCost(40), 1e-12);
    try testing.expectApproxEqAbs(REL_COST[1], relCost(0), 1e-12);
}

test "d_recall: the best copy length grows with the per-position acceptance" {
    try testing.expect(bestCopyLen(0.95, MIN_COPY, MAX_DRAFTS) >= 13);
    const mid = bestCopyLen(0.82, MIN_COPY, MAX_DRAFTS);
    try testing.expect(mid >= 3 and mid <= 5);
    try testing.expectEqual(@as(usize, 2), bestCopyLen(0.5, MIN_COPY, MAX_DRAFTS));
    try testing.expectEqual(@as(usize, 6), bestCopyLen(0.995, MIN_COPY, 6));
}

test "d_recall: copies that land four of twelve teach a per-position acceptance near 0.8" {
    var g = Gate.init(5);
    g.adaptive = true;
    var st: Stream = .{ .sampled = true };
    g.join(&st);
    for (0..40) |_| {
        st.cooldown = 0;
        st.pending = .{ .tier = 0, .probe = false, .real = 12, .own = 0, .sampled = true };
        g.feedbackRows(&st, 4, 12);
    }
    const a = g.alpha(true, 0);
    try testing.expect(a >= 0.75 and a <= 0.85);
    // Greedy keeps its prior.
    try testing.expectApproxEqAbs(@as(f64, 0.9), g.alpha(false, 0), 1e-12);
}

test "d_recall: a fully landed copy adds no failed trial; a short landing adds one" {
    var g = Gate.init(5);
    var st: Stream = .{};
    st.pending = .{ .tier = 1, .probe = false, .real = 6, .own = 0 };
    g.feedbackRows(&st, 6, 6);
    try testing.expectApproxEqAbs(@as(f64, 6), g.rates[0][1].acc, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6), g.rates[0][1].tri, 1e-12);
    st.pending = .{ .tier = 1, .probe = false, .real = 6, .own = 0 };
    g.feedbackRows(&st, 2, 6);
    try testing.expectApproxEqAbs(@as(f64, 6 * DECAY + 2), g.rates[0][1].acc, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6 * DECAY + 3), g.rates[0][1].tri, 1e-12);
    // The accepted count past what was verified does not count.
    st.pending = .{ .tier = 2, .probe = false, .real = 8, .own = 0 };
    g.feedbackRows(&st, 8, 3);
    try testing.expectApproxEqAbs(@as(f64, 3), g.rates[0][2].acc, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3), g.rates[0][2].tri, 1e-12);
}

fn adaptiveRecall(sampled: bool) !struct { r: Recall, reply: [80]u32 } {
    var r = try Recall.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    r.setAdaptive(true, DEFAULT_STAGE_REL, DEFAULT_MARGIN);
    var prompt: [8]u32 = undefined;
    iota(&prompt, 100);
    var reply: [80]u32 = undefined;
    iota(&reply, 900);
    r.remember(&prompt, &reply);
    _ = sampled;
    return .{ .r = r, .reply = reply };
}

test "d_recall: adaptive drafts of a sampled tier that lands little are short" {
    var x = try adaptiveRecall(true);
    defer x.r.deinit();
    const g = x.r.gateFor(3);
    for (0..3) |t| g.rates[1][t] = .{ .acc = 40, .tri = 80 };
    var st: Stream = .{ .sampled = true, .joined = true, .own_ema = 0, .own_rows_ema = 0 };
    var out: [MAX_DRAFTS]u32 = undefined;
    const d = x.r.draft(&st, x.reply[0..40], x.reply[40], &out, 3).?;
    try testing.expectEqual(@as(usize, 2), d.n);
    try testing.expectEqualSlices(u32, x.reply[41..43], out[0..2]);
    // Greedy, at its 0.9 prior, takes a long copy from the same match.
    var gs: Stream = .{ .joined = true, .own_ema = 0, .own_rows_ema = 0 };
    const dg = x.r.draft(&gs, x.reply[0..40], x.reply[40], &out, 3).?;
    try testing.expect(dg.n >= 6);
}

test "d_recall: an adaptive copy that does not beat the stream's own rate only goes as a probe" {
    var x = try adaptiveRecall(false);
    defer x.r.deinit();
    const g = x.r.gateFor(3);
    var st: Stream = .{ .joined = true, .own_ema = 15, .own_rows_ema = 15 };
    var out: [MAX_DRAFTS]u32 = undefined;
    var offered: usize = 0;
    for (0..PROBE_START) |_| {
        st.pending = null;
        if (x.r.draft(&st, x.reply[0..40], x.reply[40], &out, 3) != null) offered += 1;
    }
    try testing.expectEqual(@as(usize, 1), offered);
    try testing.expectEqual(@as(u32, 1), st.probes);
    try testing.expect(st.pending.?.probe);
    try testing.expect(g.ownRate(&st) > 3.0);
}

test "d_recall: with the adaptive gate off a draft is what the old gate gives" {
    var a = try Recall.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer a.deinit();
    var b = try adaptiveRecall(false);
    defer b.r.deinit();
    b.r.setAdaptive(false, DEFAULT_STAGE_REL, DEFAULT_MARGIN);
    var prompt: [8]u32 = undefined;
    iota(&prompt, 100);
    a.remember(&prompt, &b.reply);
    var out_a: [MAX_DRAFTS]u32 = undefined;
    var out_b: [MAX_DRAFTS]u32 = undefined;
    var sa: Stream = .{ .joined = true, .own_ema = 0 };
    var sb: Stream = .{ .joined = true, .own_ema = 0 };
    const da = a.draft(&sa, b.reply[0..40], b.reply[40], &out_a, 3).?;
    const db = b.r.draft(&sb, b.reply[0..40], b.reply[40], &out_b, 3).?;
    try testing.expectEqual(da.n, db.n);
    try testing.expectEqual(da.len, db.len);
    try testing.expectEqual(@as(usize, MAX_DRAFTS), da.n);
    try testing.expectEqualSlices(u32, out_a[0..da.n], out_b[0..db.n]);
    a.feedback(&sa, 5, da.n);
    b.r.feedback(&sb, 5, db.n);
    try testing.expectApproxEqAbs(a.gate.?.expect(0), b.r.gate.?.expect(0), 1e-12);
}

test "d_recall: own rounds of sampled and greedy streams move their own global" {
    var g = Gate.init(5);
    var greedy: Stream = .{};
    var sampled: Stream = .{ .sampled = true };
    // Off: a sampled round is not observed at all.
    g.observeOwn(&sampled, 4, 5);
    try testing.expect(!sampled.joined);
    try testing.expectApproxEqAbs(@as(f64, 2.5), g.own_global_s, 1e-12);
    g.adaptive = true;
    g.observeOwn(&sampled, 4, 5);
    try testing.expectApproxEqAbs(@as(f64, 0.99 * 2.5 + 0.04), g.own_global_s, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.5), g.own_global, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.75 * 5 + 0.25 * 5), sampled.own_rows_ema, 1e-12);
    g.observeOwn(&greedy, 0, 1);
    try testing.expectApproxEqAbs(@as(f64, 0.99 * 2.5), g.own_global, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.75 * 5 + 0.25), greedy.own_rows_ema, 1e-12);
}

test "d_recall: an adaptive probe wins when its tokens per unit of verify beat the stream's own" {
    var g = Gate.init(5);
    g.adaptive = true;
    var st: Stream = .{ .joined = true };
    g.probe_iv = 64;
    st.pending = .{ .tier = 0, .probe = true, .real = 4, .own = 9, .own_rate = 1.2 };
    g.feedbackRows(&st, 4, 4); // 5 / REL_COST[5] = 2.4 > 1.2
    try testing.expectEqual(@as(u32, PROBE_START), g.probe_iv);
    st.pending = .{ .tier = 0, .probe = true, .real = 4, .own = 0, .own_rate = 1.2 };
    g.feedbackRows(&st, 0, 4); // 1 / REL_COST[5] < 1.2
    try testing.expectEqual(@as(u32, 2 * PROBE_START), g.probe_iv);
}

// ── suffix automaton (P3) ──

/// The longest suffix of `q` that ends somewhere in `t` without crossing a SENTINEL, by brute force.
fn bruteLongest(t: []const u32, q: []const u32) usize {
    var best: usize = 0;
    for (0..t.len) |e| {
        var l: usize = 0;
        while (l < q.len and l <= e) : (l += 1) {
            const x = t[e - l];
            if (x == SENTINEL or x != q[q.len - 1 - l]) break;
        }
        best = @max(best, l);
    }
    return best;
}

test "d_recall: the automaton's longest suffix match agrees with brute force" {
    var prng = std.Random.DefaultPrng.init(597);
    const rnd = prng.random();
    for ([_]u32{ 3, 5 }) |alpha| {
        var text: [2000]u32 = undefined;
        for (&text) |*x| x.* = if (rnd.uintLessThan(u32, 97) == 0) SENTINEL else rnd.uintLessThan(u32, alpha);
        var s = try Sam.init(testing.allocator, 1000, text.len);
        defer s.deinit();
        for (text) |x| try testing.expect(s.extend(x));
        try testing.expect(s.full());
        try testing.expect(!s.extend(0));
        for (0..200) |_| {
            var q: [40]u32 = undefined;
            const ql = 1 + rnd.uintLessThan(usize, q.len);
            for (q[0..ql]) |*x| x.* = rnd.uintLessThan(u32, alpha);
            // Half the tails end with a real substring, so long matches get tested too.
            if (rnd.boolean()) {
                const e = rnd.uintLessThan(usize, text.len);
                const l = @min(ql, e + 1);
                for (0..l) |j| {
                    const x = text[e + 1 - l + j];
                    q[ql - l + j] = if (x == SENTINEL) 0 else x;
                }
            }
            const m = s.longest(q[0..ql]);
            try testing.expectEqual(bruteLongest(&text, q[0..ql]), m.len);
            if (m.len > 0) {
                const end_g = s.after(m) - 1 - 1000;
                for (0..m.len) |j| try testing.expectEqual(q[ql - 1 - j], text[@intCast(end_g - j)]);
            }
        }
    }
}

test "d_recall: a reset automaton forgets and takes ids again" {
    var s = try Sam.init(testing.allocator, 0, 64);
    defer s.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5 };
    for (a) |x| _ = s.extend(x);
    try testing.expectEqual(@as(usize, 3), s.longest(&[_]u32{ 9, 2, 3, 4 }).len);
    s.reset(500);
    try testing.expectEqual(@as(usize, 0), s.longest(&[_]u32{ 2, 3, 4 }).len);
    for (a) |x| _ = s.extend(x);
    const m = s.longest(&[_]u32{ 2, 3 });
    try testing.expectEqual(@as(usize, 2), m.len);
    try testing.expectEqual(@as(u64, 503), s.after(m));
}

fn samStore(ids: usize) !Store {
    var s = try Store.init(testing.allocator, 1 << 16, DEFAULT_INDEX_DENSITY);
    errdefer s.deinit();
    try s.enableSam(ids);
    return s;
}

test "d_recall: with the automaton a forked reply is found again three ids after the fork" {
    var s = try samStore(4096);
    defer s.deinit();
    var a: [200]u32 = undefined;
    iota(&a, 1000);
    _ = s.add(&a);
    // The context agrees with `a` up to 1059, forks for two ids, then rejoins at 1062..1064.
    var tail: [70]u32 = undefined;
    iota(tail[0..60], 1000);
    tail[60] = 7;
    tail[61] = 8;
    iota(tail[62..65], 1062);
    const hit = s.lookup(tail[0..65]).?;
    try testing.expectEqual(@as(usize, 3), hit.len);
    var out: [4]u32 = undefined;
    try testing.expectEqual(@as(usize, 4), s.continuation(hit.pos, &out));
    try testing.expectEqual(@as(u32, 1065), out[0]);
    try testing.expectEqual(@as(u64, 1), s.sam_hits);
    // Without the automaton the hash windows need 4 rejoined ids.
    var h = try Store.init(testing.allocator, 1 << 16, DEFAULT_INDEX_DENSITY);
    defer h.deinit();
    _ = h.add(&a);
    try testing.expect(h.lookup(tail[0..65]) == null);
}

test "d_recall: the automaton's match is exact and as long as the tail agrees" {
    var s = try samStore(4096);
    defer s.deinit();
    var a: [300]u32 = undefined;
    iota(&a, 1);
    _ = s.add(&a);
    var tail: [100]u32 = undefined;
    iota(&tail, 101);
    tail[0] = 99999;
    const hit = s.lookup(&tail).?;
    try testing.expectEqual(@as(usize, 99), hit.len);
    var out: [1]u32 = undefined;
    _ = s.continuation(hit.pos, &out);
    try testing.expectEqual(@as(u32, 201), out[0]);
}

test "d_recall: the last generation still answers, the one before it is dropped unless it was hot" {
    var s = try samStore(2048);
    defer s.deinit();
    var cold: [300]u32 = undefined;
    iota(&cold, 10_000);
    var hot: [300]u32 = undefined;
    iota(&hot, 20_000);
    _ = s.add(&cold);
    _ = s.add(&hot);
    const q_cold = [_]u32{ 10_100, 10_101, 10_102 };
    const q_hot = [_]u32{ 20_100, 20_101, 20_102 };
    // Hit the hot segment twice.
    try testing.expect(s.lookup(&q_hot) != null);
    try testing.expect(s.lookup(&q_hot) != null);
    // Fill past one generation: both are now in the last generation and still found.
    var fill: [1500]u32 = undefined;
    iota(&fill, 30_000);
    _ = s.add(&fill);
    try testing.expectEqual(@as(u64, 1), s.sam_rot);
    try testing.expect(s.sam_old != null);
    try testing.expect(s.samLookup(&q_cold) != null);
    // Fill past another: the first generation is dropped; only the hot segment is carried.
    var fill2: [2200]u32 = undefined;
    iota(&fill2, 40_000);
    _ = s.add(fill2[0..1100]);
    _ = s.add(fill2[1100..]);
    try testing.expect(s.sam_rot >= 2);
    try testing.expect(s.samLookup(&q_cold) == null);
    try testing.expect(s.samLookup(&q_hot) != null);
    try testing.expectEqual(@as(u64, 300), s.sam_hot_ids);
    var out: [2]u32 = undefined;
    const hit = s.samLookup(&q_hot).?;
    try testing.expectEqual(@as(usize, 2), s.continuation(hit.pos, &out));
    try testing.expectEqual(@as(u32, 20_103), out[0]);
}

test "d_recall: with the automaton off the lookups match the hash windows exactly" {
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    var a = try Store.init(testing.allocator, 1 << 14, DEFAULT_INDEX_DENSITY);
    defer a.deinit();
    var b = try Store.init(testing.allocator, 1 << 14, DEFAULT_INDEX_DENSITY);
    defer b.deinit();
    try b.enableSam(0);
    var seg: [400]u32 = undefined;
    for (0..20) |_| {
        for (&seg) |*x| x.* = rnd.uintLessThan(u32, 50);
        _ = a.add(&seg);
        _ = b.add(&seg);
    }
    for (0..100) |_| {
        var q: [40]u32 = undefined;
        for (&q) |*x| x.* = rnd.uintLessThan(u32, 50);
        const ha = a.lookup(&q);
        const hb = b.lookup(&q);
        try testing.expectEqual(ha == null, hb == null);
        if (ha) |x| {
            try testing.expectEqual(x.pos, hb.?.pos);
            try testing.expectEqual(x.len, hb.?.len);
        }
    }
    try testing.expectEqual(a.end, b.end);
}

test "d_recall: a reply remembered behind a long prompt is found from the prompt's middle" {
    var r = try Recall.init(testing.allocator, 1 << 16, DEFAULT_INDEX_DENSITY);
    defer r.deinit();
    try r.setSam(1 << 14, 2048);
    var prompt: [1500]u32 = undefined;
    iota(&prompt, 5000);
    var reply: [40]u32 = undefined;
    iota(&reply, 90_000);
    r.remember(&prompt, &reply);
    // An id from the prompt's first quarter (outside the old 64-id tail) continues the prompt.
    const hit = r.store.lookup(&[_]u32{ 5200, 5201, 5202, 5203 }).?;
    var out: [1]u32 = undefined;
    _ = r.store.continuation(hit.pos, &out);
    try testing.expectEqual(@as(u32, 5204), out[0]);
    const h2 = r.store.lookup(&[_]u32{ 6497, 6498, 6499 }).?;
    _ = r.store.continuation(h2.pos, &out);
    try testing.expectEqual(@as(u32, 90_000), out[0]);
}

test "d_recall: a generation the ring left behind is not answered from" {
    var s = try Store.init(testing.allocator, MIN_CAP, DEFAULT_INDEX_DENSITY);
    defer s.deinit();
    try s.enableSam(1 << 16);
    var a: [500]u32 = undefined;
    iota(&a, 70_000);
    _ = s.add(&a);
    var r: u32 = 0;
    var junk: [1000]u32 = undefined;
    while (s.base() <= 600) {
        for (&junk) |*x| {
            x.* = r;
            r += 1;
        }
        _ = s.add(&junk);
    }
    // The generation began before the ring's base: no hit from it, and no read outside the ring.
    try testing.expect(s.samLookup(&[_]u32{ 70_010, 70_011, 70_012 }) == null);
    const hit = s.lookup(&[_]u32{ 70_010, 70_011, 70_012, 70_013 });
    if (hit) |h| try testing.expect(h.pos >= s.base());
}

test "d_recall: automaton generations of the default size, and their speed" {
    const n: usize = if (@import("builtin").mode == .debug) 100_000 else DEFAULT_SAM_IDS;
    var s = try Sam.init(testing.allocator, 0, n);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    // Text-like ids: a phrase pool reused with noise, so the automaton sees real repeats.
    var pool: [4096]u32 = undefined;
    for (&pool) |*x| x.* = rnd.uintLessThan(u32, 30_000);
    const io = testing.io;
    var t0 = std.Io.Timestamp.now(io, .awake);
    var i: usize = 0;
    while (!s.full()) : (i += 1) {
        const t = if (rnd.uintLessThan(u32, 5) == 0) rnd.uintLessThan(u32, 129_000) else pool[(i * 7 + rnd.uintLessThan(usize, 3)) % pool.len];
        try testing.expect(s.extend(t));
    }
    const ext_ns: u64 = @intCast(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds());
    var tail: [128]u32 = undefined;
    t0 = std.Io.Timestamp.now(io, .awake);
    const reps = 2000;
    var tot: usize = 0;
    for (0..reps) |k| {
        for (&tail, 0..) |*x, j| x.* = pool[(k * 13 + j * 7) % pool.len];
        tot += s.longest(&tail).len;
    }
    const q_ns: u64 = @intCast(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds());
    std.debug.print("\n  [sam-bench] ids={d} states={d} edges={d} bytes={d} extend={d} ms longest(128)={d} ns avg_len={d}\n", .{
        n, s.states, s.edges, s.bytes(), ext_ns / 1_000_000, q_ns / reps, tot / reps,
    });
    try testing.expect(s.edges <= 3 * n + 4);
}

test "d_recall: the automaton continues from the newest occurrence, or the first when the newest ends its segment" {
    var s = try samStore(4096);
    defer s.deinit();
    _ = s.add(&[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 101, 102, 103 });
    _ = s.add(&[_]u32{ 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 1, 2, 3, 4, 200, 201, 202, 203 });
    var out: [1]u32 = undefined;
    const h1 = s.lookup(&[_]u32{ 2, 3, 4 }).?;
    _ = s.continuation(h1.pos, &out);
    try testing.expectEqual(@as(u32, 200), out[0]);
    // 14 15 16 last occurs at the start of the first segment only; 44 45 46 1 2 3 4 occurs once.
    const h2 = s.lookup(&[_]u32{ 14, 15, 16 }).?;
    _ = s.continuation(h2.pos, &out);
    try testing.expectEqual(@as(u32, 100), out[0]);
    // A newest occurrence right at a segment's end falls back to the first one.
    _ = s.add(&[_]u32{ 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 101, 102, 103 });
    const h3 = s.lookup(&[_]u32{ 101, 102, 103 });
    try testing.expect(h3 == null or s.continuation(h3.?.pos, &out) == 0 or out[0] != SENTINEL);
}

// ── tree rounds ──

test "d_recall tree: a context recorded twice with two continuations gives both, distinct, and moves no counter" {
    var r = try Recall.init(testing.allocator, 1 << 16, DEFAULT_INDEX_DENSITY);
    defer r.deinit();
    r.setAdaptive(true, DEFAULT_STAGE_REL, DEFAULT_MARGIN);
    try r.setSam(4096, PROMPT_TAIL);
    var a: [60]u32 = undefined;
    iota(&a, 1000);
    var b: [60]u32 = undefined;
    iota(&b, 1000);
    iota(b[30..], 5000); // b agrees with a for 30 ids, then goes its own way
    r.remember(&.{}, &a);
    r.remember(&.{}, &b);
    const lookups = r.store.lookups;
    var st: Stream = .{ .joined = true };
    var outs: [MAX_CANDS][MAX_DRAFTS]u32 = undefined;
    var cands: [MAX_CANDS]Cand = undefined;
    const n = r.drafts(&st, a[0..29], a[29], &outs, MAX_DRAFTS, 3, &cands);
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(lookups, r.store.lookups);
    var seen_a = false;
    var seen_b = false;
    for (cands[0..n], 0..) |c, k| {
        try testing.expect(c.n >= MIN_COPY);
        if (outs[k][0] == 1030) {
            seen_a = true;
            try testing.expectEqualSlices(u32, a[30 .. 30 + c.n], outs[k][0..c.n]);
        }
        if (outs[k][0] == 5000) {
            seen_b = true;
            try testing.expectEqualSlices(u32, b[30 .. 30 + c.n], outs[k][0..c.n]);
        }
    }
    try testing.expect(seen_a and seen_b);
    // A stream cooling down gets none.
    st.cooldown = 2;
    try testing.expectEqual(@as(usize, 0), r.drafts(&st, a[0..29], a[29], &outs, MAX_DRAFTS, 3, &cands));
    try testing.expectEqual(@as(u8, 2), st.cooldown);
}

test "d_recall tree: a continuation that starts another one counts once, the longer" {
    var r = try Recall.init(testing.allocator, 1 << 16, DEFAULT_INDEX_DENSITY);
    defer r.deinit();
    r.setAdaptive(true, DEFAULT_STAGE_REL, DEFAULT_MARGIN);
    try r.setSam(4096, PROMPT_TAIL);
    var a: [80]u32 = undefined;
    iota(&a, 1000);
    r.remember(&.{}, &a);
    r.remember(&.{}, a[0..34]); // the same reply again, cut short: its continuation is a prefix of the first's
    var st: Stream = .{ .joined = true };
    var outs: [MAX_CANDS][MAX_DRAFTS]u32 = undefined;
    var cands: [MAX_CANDS]Cand = undefined;
    const n = r.drafts(&st, a[0..29], a[29], &outs, MAX_DRAFTS, 3, &cands);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqualSlices(u32, a[30 .. 30 + cands[0].n], outs[0][0..cands[0].n]);
    try testing.expect(cands[0].n > 4);
}

fn constP(out: []f64, v: f64) []const f64 {
    @memset(out, v);
    return out;
}

test "d_recall tree: the trie counts a shared prefix once and scales siblings that promise more than their parent" {
    var pa: [4]f64 = undefined;
    var pb: [4]f64 = undefined;
    const ta = [_]u32{ 1, 2, 3, 4 };
    const tb = [_]u32{ 1, 2, 9, 9 };
    const br = [_]Branch{ .{ .toks = &ta, .p = constP(&pa, 0.5) }, .{ .toks = &tb, .p = constP(&pb, 0.5) } };
    // One chain: 1 + .5 + .25 + .125 + .0625.
    try testing.expectApproxEqAbs(@as(f64, 1.9375), treeTokens(&br, &.{ 4, 0 }, MAX_DRAFTS), 1e-9);
    // Two chains sharing 1, 2: the shared nodes once, the two depth-2 siblings .125 each, their children .0625 each.
    try testing.expectApproxEqAbs(@as(f64, 1 + 0.5 + 0.25 + 0.125 * 2 + 0.0625 * 2), treeTokens(&br, &.{ 4, 4 }, MAX_DRAFTS), 1e-9);
    // Siblings at 0.9 each cannot both be: scaled to sum 1.
    var pc: [1]f64 = undefined;
    var pd: [1]f64 = undefined;
    const sib = [_]Branch{ .{ .toks = &.{5}, .p = constP(&pc, 0.9) }, .{ .toks = &.{6}, .p = constP(&pd, 0.9) } };
    try testing.expectApproxEqAbs(@as(f64, 2), treeTokens(&sib, &.{ 1, 1 }, MAX_DRAFTS), 1e-9);
    // The token budget: nothing past `cap` drafts counts.
    try testing.expectApproxEqAbs(@as(f64, 1.75), treeTokens(&br, &.{ 4, 0 }, 2), 1e-9);
}

test "d_recall tree: with no recalled branch the plan is the stage draft's best cut on the ladder" {
    var p: [8]f64 = undefined;
    const t = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const br = [_]Branch{.{ .toks = &t, .p = constP(&p, 0.85) }};
    const ch = choosePlan(&br, 2, .{});
    try testing.expectEqual(@as(usize, 0), ch.mixed.branches);
    // The same cut by brute force over copyRate's model (stage cost on top of each).
    var best: usize = 0;
    var rate: f64 = 0;
    for (0..9) |l| {
        var e: f64 = 1;
        var a: f64 = 1;
        for (0..l) |_| {
            a *= 0.85;
            e += a;
        }
        const r = e / (relCost(l + 1) + DEFAULT_STAGE_REL);
        if (r > rate) {
            rate = r;
            best = l;
        }
    }
    try testing.expectEqual(best, ch.own.lens[0]);
    try testing.expectApproxEqAbs(rate, ch.own.rate, 1e-9);
}

test "d_recall tree: a confident recalled branch joins an unsure stage draft, a weak one does not" {
    var ps: [6]f64 = undefined;
    var pr: [10]f64 = undefined;
    const ts = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const tr = [_]u32{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    // The stage draft is unsure (0.4 a position), the recalled branch disagrees at its first id and lands 0.9.
    const strong = [_]Branch{ .{ .toks = &ts, .p = constP(&ps, 0.4) }, .{ .toks = &tr, .p = constP(&pr, 0.9) } };
    const a = choosePlan(&strong, 2, .{});
    try testing.expect(a.mixed.has(1));
    try testing.expect(a.mixed.rate > a.own.rate * 1.05);
    try testing.expect(a.mixed.rows <= TREE_ROWS);
    // At 0.05 the recalled branch buys less than its rows and its state cost.
    const weak = [_]Branch{ .{ .toks = &ts, .p = constP(&ps, 0.4) }, .{ .toks = &tr, .p = constP(&pr, 0.05) } };
    const b = choosePlan(&weak, 2, .{});
    try testing.expect(b.mixed.rate < b.own.rate);
    // A confident stage draft leaves a middling recalled branch out.
    var ph: [6]f64 = undefined;
    const sure = [_]Branch{ .{ .toks = &ts, .p = constP(&ph, 0.97) }, .{ .toks = &tr, .p = constP(&pr, 0.5) } };
    const c = choosePlan(&sure, 2, .{});
    try testing.expect(c.mixed.rate < c.own.rate * 1.05);
}

test "d_recall tree: plans stay within the round's rows, and max_recall bounds the recalled branches" {
    var p0: [15]f64 = undefined;
    var p1: [15]f64 = undefined;
    var p2: [15]f64 = undefined;
    var p3: [15]f64 = undefined;
    var t: [4][15]u32 = undefined;
    for (&t, 0..) |*row, b| for (row, 0..) |*x, d| {
        x.* = @intCast(100 * b + d);
    };
    const br = [_]Branch{
        .{ .toks = &t[0], .p = constP(&p0, 0.6) },
        .{ .toks = &t[1], .p = constP(&p1, 0.9) },
        .{ .toks = &t[2], .p = constP(&p2, 0.9) },
        .{ .toks = &t[3], .p = constP(&p3, 0.9) },
    };
    for ([_]usize{ 1, 2, 3 }) |k| {
        const ch = choosePlan(&br, k, .{});
        var rec: usize = 0;
        var rows: usize = 0;
        for (ch.mixed.lens, 0..) |l, b| {
            if (l == 0) continue;
            rows += l + 1;
            if (b > 0) rec += 1;
        }
        try testing.expect(rec >= 1 and rec <= k);
        try testing.expectEqual(ch.mixed.rows, rows);
        try testing.expect(rows <= TREE_ROWS);
    }
}

test "d_recall tree: the walk follows the live branches and keeps the lowest one's pick" {
    // Branch 0 agrees for 1 draft, branch 1 for 3; picks along each branch's rows.
    const t0 = [_]u32{ 10, 11, 12 };
    const t1 = [_]u32{ 10, 21, 22, 23 };
    const y0 = [_]u32{ 10, 21, 0, 0 };
    const y1 = [_]u32{ 10, 21, 22, 30, 0 };
    const w = walkTree(&.{ &t0, &t1 }, &.{ &y0, &y1 }, MAX_DRAFTS);
    try testing.expectEqual(@as(usize, 3), w.accepted);
    try testing.expectEqual(@as(u32, 30), w.next);
    try testing.expectEqual(@as(usize, 1), w.winner);
    try testing.expectEqual(@as(u32, 1), w.landed[0]);
    try testing.expectEqual(@as(usize, 2), w.verified[0]);
    try testing.expectEqual(@as(u32, 3), w.landed[1]);
    try testing.expectEqual(@as(usize, 4), w.verified[1]);
    // The token budget stops the walk at `cap` drafts, its pick the row there.
    const c = walkTree(&.{ &t0, &t1 }, &.{ &y0, &y1 }, 2);
    try testing.expectEqual(@as(usize, 2), c.accepted);
    try testing.expectEqual(@as(u32, 22), c.next);
    // One branch: the chain's accept.
    const one = walkTree(&.{&t0}, &.{&y0}, MAX_DRAFTS);
    try testing.expectEqual(@as(usize, 1), one.accepted);
    try testing.expectEqual(@as(u32, 21), one.next);
    // A branch that runs out while another lives: the other carries on.
    const short = [_]u32{10};
    const ys = [_]u32{ 10, 21 };
    const s = walkTree(&.{ &short, &t1 }, &.{ &ys, &y1 }, MAX_DRAFTS);
    try testing.expectEqual(@as(usize, 3), s.accepted);
    try testing.expectEqual(@as(usize, 1), s.winner);
}

test "d_recall tree: a sampled walk over point-mass branches emits the target's own distribution" {
    // A toy target over 4 tokens whose next-token density depends on the last token; three branches of point masses.
    const V = 4;
    const dens = [V][V]f64{
        .{ 0.1, 0.6, 0.2, 0.1 },
        .{ 0.3, 0.1, 0.5, 0.1 },
        .{ 0.25, 0.25, 0.25, 0.25 },
        .{ 0.7, 0.1, 0.1, 0.1 },
    };
    const t1: u32 = 0;
    const branches = [_][]const u32{ &.{ 1, 2, 0 }, &.{ 1, 0, 1 }, &.{ 2, 3 } };
    const L = 3; // emitted tokens compared: the accepted drafts and the next token, then serial past them
    var prng = std.Random.DefaultPrng.init(0x7ee);
    const rnd = prng.random();
    const draw = struct {
        fn f(rr: std.Random, d: []const f64) u32 {
            const u = rr.float(f64);
            var acc: f64 = 0;
            for (d, 0..) |p, i| {
                acc += p;
                if (u < acc) return @intCast(i);
            }
            return @intCast(d.len - 1);
        }
    }.f;
    const trials = 40000;
    var tree_hist: [V * V * V]u32 = @splat(0);
    var serial_hist: [V * V * V]u32 = @splat(0);
    for (0..trials) |_| {
        // Every row's pick, drawn from the target given that row's context.
        var ys_buf: [3][4]u32 = undefined;
        var ys: [3][]const u32 = undefined;
        for (branches, 0..) |br, b| {
            for (0..br.len + 1) |d| {
                const last: u32 = if (d == 0) t1 else br[d - 1];
                ys_buf[b][d] = draw(rnd, &dens[last]);
            }
            ys[b] = ys_buf[b][0 .. br.len + 1];
        }
        const w = walkTree(&branches, &ys, MAX_DRAFTS);
        var seq: [L]u32 = undefined;
        var k: usize = 0;
        while (k < @min(w.accepted, L)) : (k += 1) seq[k] = branches[w.winner][k];
        if (k < L) {
            seq[k] = w.next;
            k += 1;
        }
        while (k < L) : (k += 1) seq[k] = draw(rnd, &dens[seq[k - 1]]);
        tree_hist[(seq[0] * V + seq[1]) * V + seq[2]] += 1;
        var s: [L]u32 = undefined;
        s[0] = draw(rnd, &dens[t1]);
        for (1..L) |i| s[i] = draw(rnd, &dens[s[i - 1]]);
        serial_hist[(s[0] * V + s[1]) * V + s[2]] += 1;
    }
    // Against the exact density of three serial draws too.
    var tv_exact: f64 = 0;
    var tv_serial: f64 = 0;
    for (0..V) |a| for (0..V) |b| for (0..V) |c| {
        const i = (a * V + b) * V + c;
        const exact = dens[t1][a] * dens[a][b] * dens[b][c];
        const f_tree = @as(f64, @floatFromInt(tree_hist[i])) / trials;
        const f_serial = @as(f64, @floatFromInt(serial_hist[i])) / trials;
        tv_exact += @abs(f_tree - exact);
        tv_serial += @abs(f_tree - f_serial);
    };
    try testing.expect(tv_exact / 2 < 0.02);
    try testing.expect(tv_serial / 2 < 0.02);
}

test "d_recall tree: a branch's lesson moves its tier's acceptance as an offered copy's does" {
    var g = Gate.init(3);
    g.adaptive = true;
    const before = g.alpha(false, 0);
    for (0..20) |_| g.learnBranch(false, 0, 1, 4);
    try testing.expect(g.alpha(false, 0) < before);
    try testing.expect(g.alpha(false, 0) < 0.6);
    var h = Gate.init(3);
    for (0..20) |_| h.learnBranch(true, 1, 8, 8);
    try testing.expect(h.alpha(true, 1) > 0.95);
    try testing.expectApproxEqAbs(@as(f64, 0.8), Gate.init(3).alpha(true, 1), 1e-12); // the prior stays where it was
}
