//! The recall record's cold layer (Yee 2.4): every finished reply (behind its prompt tail, as the hot record keeps it)
//! appended to `tokens.bin` on NVMe and indexed by its 16-id windows in `index.bin`, an open-addressing table on disk.
//! One background thread owns both files: it appends what the inference thread queued and answers the lookups the
//! inference thread posted for the round after the one it drafts (the context's tail plus the likely accepted
//! branches). The inference thread never waits on it: posts, stores and takes only `tryLock` a small table, and a
//! busy lock, a full queue or an unanswered lookup is a miss. Reads and writes are `pread`/`pwrite`, no mmap.

const std = @import("std");
const c = std.c;

/// Ids a cold key spans: the hot record's window.
pub const KEY = 16;
/// Ids a lookup reads back after its key.
pub const CONT = 48;
/// Posted lookups and their answers kept at once.
pub const RESULTS = 64;
/// Finished segments waiting to be written.
pub const QUEUE = 256;
pub const MAX_QUEUED_IDS: usize = 1 << 23;
/// Slots a key probes in the index before it gives up (lookup) or evicts the oldest (insert).
pub const PROBE = 8;
pub const DEFAULT_SLOTS: u64 = 268435456;
pub const DEFAULT_GB: f32 = 64;
pub const DEFAULT_PREFETCH = 3;
pub const MAX_PREFETCH = 3;
pub const HDR = 32;
pub const VERSION: u32 = 1;
pub const MAGIC_TOK: u64 = 0x3130_4B4F_5444_4C43; // "CLDTOK01"
pub const MAGIC_IDX: u64 = 0x3130_5844_4944_4C43; // "CLDIDX01"
const SENT: u32 = std.math.maxInt(u32);
const IDLE_US = 200;

extern "c" fn usleep(us: c_uint) c_int;

const Header = extern struct { magic: u64, version: u32, pad: u32 = 0, len: u64, extra: u64 = 0 };
const Slot = extern struct { hash: u64, pos: u64 };

pub fn keyHash(ids: []const u32) u64 {
    const h = std.hash.Wyhash.hash(0x5965_6532_3400, std.mem.sliceAsBytes(ids));
    return if (h == 0) 1 else h;
}

const Lock = struct {
    v: std.atomic.Value(u32) = .init(0),

    fn tryLock(l: *Lock) bool {
        return l.v.cmpxchgStrong(0, 1, .acquire, .monotonic) == null;
    }

    fn lock(l: *Lock) void {
        while (!l.tryLock()) std.Thread.yield() catch {};
    }

    fn unlock(l: *Lock) void {
        l.v.store(0, .release);
    }
};

const State = enum(u8) { empty, pending, busy, ready };

const Result = struct {
    seq: u64 = 0,
    sid: u32 = 0,
    state: State = .empty,
    n: u8 = 0,
    /// Ids of the context the key ends at (the key is its `[at - KEY, at)`).
    at: u64 = 0,
    key: [KEY]u32 = undefined,
    cont: [CONT]u32 = undefined,
};

pub const Config = struct {
    slots: u64 = DEFAULT_SLOTS,
    max_bytes: u64 = @as(u64, 64) << 30,
    /// False in tests: `pump` then runs the background work on the caller's thread.
    thread: bool = true,
};

pub const Stats = struct { ids: u64, posts: u64, ready: u64, late: u64, offers: u64, landed: u64, drop: u64, ioerr: u64 };

pub const Cold = struct {
    gpa: std.mem.Allocator,
    tok_fd: c_int = -1,
    idx_fd: c_int = -1,
    slots: u64 = 0,
    max_bytes: u64 = 0,
    /// Ids held in `tokens.bin` (the header's length); the background thread's alone, read for the log.
    len: std.atomic.Value(u64) = .init(0),
    full: bool = false,
    on: std.atomic.Value(bool) = .init(true),
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    lock: Lock = .{},
    results: [RESULTS]Result = @splat(.{}),
    next_seq: u64 = 0,
    queue: [QUEUE][]u32 = undefined,
    q_head: usize = 0,
    q_len: usize = 0,
    q_ids: usize = 0,

    posts: u64 = 0,
    late: u64 = 0,
    offers: u64 = 0,
    landed: u64 = 0,
    ready: std.atomic.Value(u64) = .init(0),
    drop: std.atomic.Value(u64) = .init(0),
    ioerr: std.atomic.Value(u64) = .init(0),

    /// The layer over `dir` (created when missing), its files reopened where a run left them; null for an empty
    /// `dir`, which touches nothing.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, cfg: Config) !?*Cold {
        if (dir.len == 0) return null;
        var pbuf: [4096]u8 = undefined;
        if (dir.len + 1 > pbuf.len) return error.ColdPathTooLong;
        @memcpy(pbuf[0..dir.len], dir);
        pbuf[dir.len] = 0;
        _ = c.mkdir(pbuf[0..dir.len :0].ptr, 0o755);

        const self = try gpa.create(Cold);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .max_bytes = cfg.max_bytes };
        errdefer self.closeFds();

        self.tok_fd = try openFile(&pbuf, dir, "tokens.bin");
        self.idx_fd = try openFile(&pbuf, dir, "index.bin");

        var th: Header = undefined;
        if (preadAll(self.tok_fd, std.mem.asBytes(&th), 0) and th.magic == MAGIC_TOK) {
            if (th.version != VERSION) return error.ColdVersion;
            self.len.store(th.len, .monotonic);
        } else {
            if (fileBytes(self.tok_fd) > 0) return error.ColdNotOurs;
            th = .{ .magic = MAGIC_TOK, .version = VERSION, .len = 0 };
            if (!pwriteAll(self.tok_fd, std.mem.asBytes(&th), 0)) return error.ColdIo;
        }
        // A run killed between its append and its header leaves ids past the length: they go.
        if (c.ftruncate(self.tok_fd, @intCast(HDR + 4 * self.len.load(.monotonic))) != 0) return error.ColdIo;

        var ih: Header = undefined;
        if (preadAll(self.idx_fd, std.mem.asBytes(&ih), 0) and ih.magic == MAGIC_IDX) {
            if (ih.version != VERSION or ih.len == 0) return error.ColdVersion;
            self.slots = ih.len;
        } else {
            if (fileBytes(self.idx_fd) > 0) return error.ColdNotOurs;
            self.slots = @max(cfg.slots, PROBE);
            ih = .{ .magic = MAGIC_IDX, .version = VERSION, .len = self.slots };
            if (!pwriteAll(self.idx_fd, std.mem.asBytes(&ih), 0)) return error.ColdIo;
            // Sparse: the table's blocks appear as it fills.
            if (c.ftruncate(self.idx_fd, @intCast(HDR + 16 * self.slots)) != 0) return error.ColdIo;
        }
        if (cfg.thread) self.thread = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    pub fn deinit(self: *Cold) void {
        self.stop.store(true, .release);
        if (self.thread) |t| t.join();
        self.closeFds();
        while (self.q_len > 0) {
            self.gpa.free(self.queue[self.q_head]);
            self.q_head = (self.q_head + 1) % QUEUE;
            self.q_len -= 1;
        }
        self.gpa.destroy(self);
    }

    fn closeFds(self: *Cold) void {
        if (self.tok_fd >= 0) _ = c.close(self.tok_fd);
        if (self.idx_fd >= 0) _ = c.close(self.idx_fd);
        self.tok_fd = -1;
        self.idx_fd = -1;
    }

    pub fn isOn(self: *const Cold) bool {
        return self.on.load(.acquire);
    }

    pub fn stats(self: *const Cold) Stats {
        return .{
            .ids = self.len.load(.monotonic),
            .posts = self.posts,
            .ready = self.ready.load(.monotonic),
            .late = self.late,
            .offers = self.offers,
            .landed = self.landed,
            .drop = self.drop.load(.monotonic),
            .ioerr = self.ioerr.load(.monotonic),
        };
    }

    // ---- the inference thread's side: never blocks ----

    /// Queues `head ++ body` (a prompt tail and its finished reply) for the background append; dropped when the
    /// queue is full or busy.
    pub fn store(self: *Cold, head: []const u32, body: []const u32) void {
        if (!self.isOn() or body.len == 0) return;
        const n = head.len + body.len;
        if (n <= KEY) return;
        const seg = self.gpa.alloc(u32, n) catch {
            _ = self.drop.fetchAdd(1, .monotonic);
            return;
        };
        @memcpy(seg[0..head.len], head);
        @memcpy(seg[head.len..], body);
        if (self.lock.tryLock()) {
            if (self.q_len < QUEUE and self.q_ids + n <= MAX_QUEUED_IDS) {
                self.queue[(self.q_head + self.q_len) % QUEUE] = seg;
                self.q_len += 1;
                self.q_ids += n;
                self.lock.unlock();
                return;
            }
            self.lock.unlock();
        }
        self.gpa.free(seg);
        _ = self.drop.fetchAdd(1, .monotonic);
    }

    /// Posts lookups for the next round of stream `sid`, whose context is now `ctx ++ [t1]` and whose drafts are
    /// `drafted` (empty when the stage drafts them): up to `branches` keys, the tail when the target rejects the
    /// first draft, the tail past all of `drafted`, past its first. Each answer serves every later round whose
    /// context runs through the key and agrees with the continuation (`take`), so the first one covers a stage
    /// draft's accepted ids too.
    pub fn prefetch(self: *Cold, sid: u32, ctx: []const u32, t1: u32, drafted: []const u32, branches: usize) void {
        if (!self.isOn() or branches == 0) return;
        var ks: [MAX_PREFETCH]usize = undefined;
        var nk: usize = 0;
        const want = @min(branches, MAX_PREFETCH);
        for ([_]usize{ 0, drafted.len, 1 }) |k| {
            if (nk >= want) break;
            if (k > drafted.len) continue;
            if (std.mem.indexOfScalar(usize, ks[0..nk], k) != null) continue;
            ks[nk] = k;
            nk += 1;
        }
        const base = ctx.len + 1;
        if (!self.lock.tryLock()) {
            _ = self.drop.fetchAdd(nk, .monotonic);
            return;
        }
        defer self.lock.unlock();
        for (ks[0..nk]) |k| {
            const end = base + k;
            if (end < KEY) continue;
            var key: [KEY]u32 = undefined;
            for (0..KEY) |i| key[i] = idAt(ctx, t1, drafted, end - KEY + i);
            // Already asked for this stream at this place.
            var dup = false;
            for (&self.results) |*r| {
                if (r.state != .empty and r.sid == sid and r.at == end and std.mem.eql(u32, &r.key, &key)) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            const slot = self.victim() orelse {
                _ = self.drop.fetchAdd(1, .monotonic);
                continue;
            };
            if (slot.state == .pending) _ = self.drop.fetchAdd(1, .monotonic);
            self.next_seq += 1;
            slot.* = .{ .seq = self.next_seq, .sid = sid, .state = .pending, .at = end, .key = key };
            self.posts += 1;
        }
    }

    /// The table entry a new post takes: an empty one, else the oldest one the background thread is not reading for.
    fn victim(self: *Cold) ?*Result {
        var best: ?*Result = null;
        for (&self.results) |*r| {
            if (r.state == .empty) return r;
            if (r.state == .busy) continue;
            if (best == null or r.seq < best.?.seq) best = r;
        }
        return best;
    }

    /// A cold continuation of `ctx ++ [t1]` for stream `sid` in `out`: the longest answered post whose key the
    /// context runs through and whose continuation agrees with every id since; 0 when none (counted late when only
    /// an unanswered post would have served).
    pub fn take(self: *Cold, sid: u32, ctx: []const u32, t1: u32, out: []u32) usize {
        if (!self.isOn() or out.len == 0) return 0;
        if (!self.lock.tryLock()) return 0;
        defer self.lock.unlock();
        const full = ctx.len + 1;
        var best: ?*const Result = null;
        var best_j: usize = 0;
        var waiting = false;
        for (&self.results) |*r| {
            if (r.sid != sid or r.state == .empty) continue;
            if (r.at > full or r.at < KEY) continue;
            const j: usize = @intCast(full - r.at);
            if (j >= CONT) continue;
            var ok = true;
            for (0..KEY) |i| {
                if (idAt(ctx, t1, &.{}, @as(usize, @intCast(r.at)) - KEY + i) != r.key[i]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            if (r.state != .ready) {
                waiting = true;
                continue;
            }
            if (j >= r.n) continue;
            for (0..j) |i| {
                if (idAt(ctx, t1, &.{}, @as(usize, @intCast(r.at)) + i) != r.cont[i]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            if (best == null or r.n - j > best.?.n - best_j) {
                best = r;
                best_j = j;
            }
        }
        const r = best orelse {
            if (waiting) self.late += 1;
            return 0;
        };
        const n = @min(out.len, r.n - best_j);
        @memcpy(out[0..n], r.cont[best_j..][0..n]);
        return n;
    }

    // ---- the background thread ----

    fn run(self: *Cold) void {
        while (!self.stop.load(.acquire) and self.isOn()) {
            if (!self.pump()) _ = usleep(IDLE_US);
        }
    }

    /// One unit of the background work: a posted lookup first, else one queued segment. False when there was none.
    pub fn pump(self: *Cold) bool {
        if (!self.isOn()) return false;
        var key: [KEY]u32 = undefined;
        var seq: u64 = 0;
        var job: ?*Result = null;
        self.lock.lock();
        for (&self.results) |*r| {
            if (r.state != .pending) continue;
            if (job == null or r.seq > job.?.seq) job = r;
        }
        if (job) |r| {
            r.state = .busy;
            key = r.key;
            seq = r.seq;
        }
        self.lock.unlock();
        if (job) |r| {
            var cont: [CONT]u32 = undefined;
            const n = self.lookup(&key, &cont);
            self.lock.lock();
            if (r.seq == seq) {
                r.n = @intCast(n);
                @memcpy(r.cont[0..n], cont[0..n]);
                r.state = .ready;
            }
            self.lock.unlock();
            if (n > 0) _ = self.ready.fetchAdd(1, .monotonic);
            return true;
        }
        var seg: ?[]u32 = null;
        self.lock.lock();
        if (self.q_len > 0) {
            seg = self.queue[self.q_head];
            self.q_head = (self.q_head + 1) % QUEUE;
            self.q_len -= 1;
            self.q_ids -= seg.?.len;
        }
        self.lock.unlock();
        const s = seg orelse return false;
        defer self.gpa.free(s);
        self.append(s);
        return true;
    }

    fn fail(self: *Cold) void {
        _ = self.ioerr.fetchAdd(1, .monotonic);
        self.on.store(false, .release);
    }

    /// Appends `seg` and a separator to `tokens.bin`, indexes its windows, then moves the header's length.
    fn append(self: *Cold, seg: []const u32) void {
        const start = self.len.load(.monotonic);
        const after = start + seg.len + 1;
        if (self.full or HDR + 4 * after + 16 * self.slots > self.max_bytes) {
            self.full = true;
            _ = self.drop.fetchAdd(1, .monotonic);
            return;
        }
        const off = HDR + 4 * start;
        const sent = [1]u32{SENT};
        if (!pwriteAll(self.tok_fd, std.mem.sliceAsBytes(seg), off) or
            !pwriteAll(self.tok_fd, std.mem.sliceAsBytes(&sent), off + 4 * seg.len))
        {
            self.fail();
            return;
        }
        var i: usize = KEY;
        while (i < seg.len) : (i += 1) {
            if (!self.insert(keyHash(seg[i - KEY .. i]), start + i)) {
                self.fail();
                return;
            }
        }
        const len_le = after;
        if (!pwriteAll(self.tok_fd, std.mem.asBytes(&len_le), @offsetOf(Header, "len"))) {
            self.fail();
            return;
        }
        self.len.store(after, .release);
    }

    fn probeRun(self: *const Cold, h: u64, buf: *[PROBE]Slot) ?u64 {
        const first = h % self.slots;
        if (first + PROBE <= self.slots) {
            if (!preadAll(self.idx_fd, std.mem.sliceAsBytes(buf), HDR + 16 * first)) return null;
        } else {
            for (0..PROBE) |k| {
                const s = (first + k) % self.slots;
                if (!preadAll(self.idx_fd, std.mem.asBytes(&buf[k]), HDR + 16 * s)) return null;
            }
        }
        return first;
    }

    /// Points `h` at id `pos` (the id after its window): an empty slot or the key's own, else the oldest of its probe.
    fn insert(self: *Cold, h: u64, pos: u64) bool {
        var buf: [PROBE]Slot = undefined;
        const first = self.probeRun(h, &buf) orelse return false;
        var pick: usize = 0;
        var found = false;
        for (buf, 0..) |s, k| {
            if (s.pos == 0 or s.hash == h) {
                pick = k;
                found = true;
                break;
            }
        }
        if (!found) {
            for (buf, 0..) |s, k| {
                if (s.pos < buf[pick].pos) pick = k;
            }
        }
        const slot: Slot = .{ .hash = h, .pos = pos + 1 };
        const at = (first + pick) % self.slots;
        return pwriteAll(self.idx_fd, std.mem.asBytes(&slot), HDR + 16 * at);
    }

    /// Up to `CONT` ids that followed `key` when it was last stored, its 16 ids read back and compared (a hash that
    /// collides finds nothing); 0 when the key is not held or the read failed.
    pub fn lookup(self: *Cold, key: *const [KEY]u32, out: *[CONT]u32) usize {
        const h = keyHash(key);
        var buf: [PROBE]Slot = undefined;
        _ = self.probeRun(h, &buf) orelse {
            self.fail();
            return 0;
        };
        const len = self.len.load(.monotonic);
        for (buf) |s| {
            if (s.pos == 0 or s.hash != h) continue;
            const p = s.pos - 1;
            if (p < KEY or p >= len) continue;
            var back: [KEY]u32 = undefined;
            if (!preadAll(self.tok_fd, std.mem.sliceAsBytes(&back), HDR + 4 * (p - KEY))) {
                self.fail();
                return 0;
            }
            if (!std.mem.eql(u32, &back, key)) continue;
            const want: usize = @intCast(@min(@as(u64, CONT), len - p));
            if (!preadAll(self.tok_fd, std.mem.sliceAsBytes(out[0..want]), HDR + 4 * p)) {
                self.fail();
                return 0;
            }
            const n = std.mem.indexOfScalar(u32, out[0..want], SENT) orelse want;
            if (n > 0) return n;
        }
        return 0;
    }
};

/// Id `i` of `ctx ++ [t1] ++ more`.
fn idAt(ctx: []const u32, t1: u32, more: []const u32, i: usize) u32 {
    if (i < ctx.len) return ctx[i];
    if (i == ctx.len) return t1;
    return more[i - ctx.len - 1];
}

fn openFile(pbuf: *[4096]u8, dir: []const u8, name: []const u8) !c_int {
    const n = dir.len + 1 + name.len;
    if (n + 1 > pbuf.len) return error.ColdPathTooLong;
    @memcpy(pbuf[0..dir.len], dir);
    pbuf[dir.len] = '/';
    @memcpy(pbuf[dir.len + 1 ..][0..name.len], name);
    pbuf[n] = 0;
    const fd = c.open(pbuf[0..n :0].ptr, .{ .ACCMODE = .RDWR, .CREAT = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.ColdOpen;
    return fd;
}

fn fileBytes(fd: c_int) u64 {
    var one: [1]u8 = undefined;
    const r = c.pread(fd, &one, 1, 0);
    return if (r > 0) 1 else 0;
}

fn preadAll(fd: c_int, buf: []u8, off: u64) bool {
    var done: usize = 0;
    while (done < buf.len) {
        const r = c.pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (r <= 0) return false;
        done += @intCast(r);
    }
    return true;
}

fn pwriteAll(fd: c_int, buf: []const u8, off: u64) bool {
    var done: usize = 0;
    while (done < buf.len) {
        const r = c.pwrite(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (r <= 0) return false;
        done += @intCast(r);
    }
    return true;
}

// ---- tests ----

const testing = std.testing;

fn testDir(buf: *[256]u8, tag: []const u8) []const u8 {
    const id: usize = @intCast(std.c.getpid());
    return std.fmt.bufPrint(buf, "/tmp/xy_cold_test_{s}_{x}", .{ tag, id }) catch unreachable;
}

fn removeDir(dir: []const u8) void {
    var pbuf: [4096]u8 = undefined;
    for ([_][]const u8{ "tokens.bin", "index.bin" }) |name| {
        const n = dir.len + 1 + name.len;
        @memcpy(pbuf[0..dir.len], dir);
        pbuf[dir.len] = '/';
        @memcpy(pbuf[dir.len + 1 ..][0..name.len], name);
        pbuf[n] = 0;
        _ = c.unlink(pbuf[0..n :0].ptr);
    }
    @memcpy(pbuf[0..dir.len], dir);
    pbuf[dir.len] = 0;
    _ = c.rmdir(pbuf[0..dir.len :0].ptr);
}

fn drain(cold: *Cold) void {
    while (cold.pump()) {}
}

fn iotaFrom(out: []u32, from: u32) void {
    for (out, 0..) |*x, i| x.* = from + @as(u32, @intCast(i));
}

const test_cfg: Config = .{ .slots = 4096, .max_bytes = 1 << 30, .thread = false };

test "d_recall cold: a stored reply is found by the next round's prefetch, and again after a reopen" {
    var dbuf: [256]u8 = undefined;
    const dir = testDir(&dbuf, "reopen");
    removeDir(dir);
    defer removeDir(dir);
    var prompt: [40]u32 = undefined;
    iotaFrom(&prompt, 100);
    var reply: [60]u32 = undefined;
    iotaFrom(&reply, 900);
    var seg: [100]u32 = undefined;
    @memcpy(seg[0..40], &prompt);
    @memcpy(seg[40..], &reply);
    {
        const cold = (try Cold.open(testing.allocator, dir, test_cfg)).?;
        defer cold.deinit();
        cold.store(&prompt, &reply);
        drain(cold);
        try testing.expectEqual(@as(u64, 101), cold.stats().ids);
        // Round n: context seg[0..41] ++ [seg[41]]; the lookup for round n+1 is posted and answered.
        cold.prefetch(7, seg[0..41], seg[41], &.{}, 3);
        try testing.expectEqual(@as(u64, 1), cold.stats().posts);
        var out: [16]u32 = undefined;
        // Not answered yet: late.
        try testing.expectEqual(@as(usize, 0), cold.take(7, seg[0..42], seg[42], &out));
        try testing.expectEqual(@as(u64, 1), cold.stats().late);
        drain(cold);
        // Round n+1 after the target rejected the first draft: its bonus id seg[42] heads the continuation.
        try testing.expectEqual(@as(usize, 16), cold.take(7, seg[0..42], seg[42], &out));
        try testing.expectEqualSlices(u32, seg[43..59], &out);
        // After four more accepted ids the same answer still serves, shifted.
        try testing.expectEqual(@as(usize, 16), cold.take(7, seg[0..46], seg[46], &out));
        try testing.expectEqualSlices(u32, seg[47..63], &out);
        // A disagreeing id since the key: nothing.
        var other: [47]u32 = seg[0..47].*;
        other[45] = 5;
        try testing.expectEqual(@as(usize, 0), cold.take(7, other[0..46], other[46], &out));
        // Another stream: nothing.
        try testing.expectEqual(@as(usize, 0), cold.take(8, seg[0..42], seg[42], &out));
    }
    {
        const cold = (try Cold.open(testing.allocator, dir, test_cfg)).?;
        defer cold.deinit();
        try testing.expectEqual(@as(u64, 101), cold.stats().ids);
        var key: [KEY]u32 = seg[50..66].*;
        var cont: [CONT]u32 = undefined;
        const n = cold.lookup(&key, &cont);
        try testing.expectEqual(@as(usize, 34), n);
        try testing.expectEqualSlices(u32, seg[66..100], cont[0..n]);
        // A second segment appends behind the first and is found too.
        var more: [30]u32 = undefined;
        iotaFrom(&more, 5000);
        cold.store(&.{}, &more);
        drain(cold);
        try testing.expectEqual(@as(u64, 132), cold.stats().ids);
        key = more[0..16].*;
        try testing.expectEqual(@as(usize, 14), cold.lookup(&key, &cont));
        try testing.expectEqualSlices(u32, more[16..30], cont[0..14]);
    }
}

test "d_recall cold: a hash that collides is stopped by the 16-id compare" {
    var dbuf: [256]u8 = undefined;
    const dir = testDir(&dbuf, "collide");
    removeDir(dir);
    defer removeDir(dir);
    const cold = (try Cold.open(testing.allocator, dir, test_cfg)).?;
    defer cold.deinit();
    var seg: [40]u32 = undefined;
    iotaFrom(&seg, 1);
    cold.store(&.{}, &seg);
    drain(cold);
    var alien: [KEY]u32 = undefined;
    iotaFrom(&alien, 777);
    // Point the alien key's hash at a stored place whose window is not the alien key.
    try testing.expect(cold.insert(keyHash(&alien), 20));
    var cont: [CONT]u32 = undefined;
    try testing.expectEqual(@as(usize, 0), cold.lookup(&alien, &cont));
    const real: [KEY]u32 = seg[4..20].*;
    try testing.expectEqual(@as(usize, 20), cold.lookup(&real, &cont));
    try testing.expectEqualSlices(u32, seg[20..40], cont[0..20]);
}

test "d_recall cold: ids past the header's length are cut on reopen, and an empty dir opens nothing" {
    try testing.expect((try Cold.open(testing.allocator, "", test_cfg)) == null);
    var dbuf: [256]u8 = undefined;
    const dir = testDir(&dbuf, "trunc");
    removeDir(dir);
    defer removeDir(dir);
    {
        const cold = (try Cold.open(testing.allocator, dir, test_cfg)).?;
        defer cold.deinit();
        var seg: [20]u32 = undefined;
        iotaFrom(&seg, 1);
        cold.store(&.{}, &seg);
        drain(cold);
        // A crash between an append and its header: ids written, length not moved.
        var junk: [8]u32 = undefined;
        iotaFrom(&junk, 3000);
        try testing.expect(pwriteAll(cold.tok_fd, std.mem.sliceAsBytes(&junk), HDR + 4 * 21));
    }
    const cold = (try Cold.open(testing.allocator, dir, test_cfg)).?;
    defer cold.deinit();
    try testing.expectEqual(@as(u64, 21), cold.stats().ids);
    var tail: [1]u8 = undefined;
    try testing.expect(c.pread(cold.tok_fd, &tail, 1, HDR + 4 * 21) == 0);
}

test "d_recall cold: a full queue drops, and the size cap stops writing but keeps reading" {
    var dbuf: [256]u8 = undefined;
    const dir = testDir(&dbuf, "cap");
    removeDir(dir);
    defer removeDir(dir);
    const cold = (try Cold.open(testing.allocator, dir, .{ .slots = 64, .max_bytes = HDR + 16 * 64 + 4 * 30, .thread = false })).?;
    defer cold.deinit();
    var seg: [20]u32 = undefined;
    iotaFrom(&seg, 1);
    for (0..QUEUE + 3) |_| cold.store(&.{}, &seg);
    try testing.expectEqual(@as(u64, 3), cold.stats().drop);
    drain(cold);
    try testing.expectEqual(@as(u64, 21), cold.stats().ids);
    const key: [KEY]u32 = seg[0..16].*;
    var cont: [CONT]u32 = undefined;
    try testing.expectEqual(@as(usize, 4), cold.lookup(&key, &cont));
}

test "d_recall cold: the background thread appends and answers on its own" {
    var dbuf: [256]u8 = undefined;
    const dir = testDir(&dbuf, "thread");
    removeDir(dir);
    defer removeDir(dir);
    const cold = (try Cold.open(testing.allocator, dir, .{ .slots = 4096, .max_bytes = 1 << 30 })).?;
    defer cold.deinit();
    var seg: [64]u32 = undefined;
    iotaFrom(&seg, 40);
    cold.store(&.{}, &seg);
    var waited: usize = 0;
    while (cold.stats().ids == 0 and waited < 5000) : (waited += 1) _ = usleep(1000);
    try testing.expectEqual(@as(u64, 65), cold.stats().ids);
    cold.prefetch(3, seg[0..20], seg[20], &.{}, 1);
    var out: [8]u32 = undefined;
    var n: usize = 0;
    waited = 0;
    while (n == 0 and waited < 5000) : (waited += 1) {
        n = cold.take(3, seg[0..21], seg[21], &out);
        if (n == 0) _ = usleep(1000);
    }
    try testing.expectEqual(@as(usize, 8), n);
    try testing.expectEqualSlices(u32, seg[22..30], &out);
}
