//! Yee KAD core for Zig engines (TensorFold, mlx-serve): per-request adaptive draft count.
//!
//! Same rules as the Python core `yee.kad` and the vLLM adapter `vllm_kad` (parse_policy,
//! ladder_update, decide_w): the engine launches with K = 12 drafts and KAD picks, per request
//! and per step, how many of them to draft and verify (5, 7, 9, 11, 12 while everything is
//! accepted; max(kmin, min(cur, accepted + 1)) after a rejection).
//!
//! The module does no I/O and has no global state. The host passes a monotonic clock (`now_ns`)
//! to `beginStep` and an `Io` table that stats / reads the policy file and atomically writes the
//! statistics file with the engine's own std.Io code. Not thread-safe: call it from the thread
//! that schedules the engine steps, or hold a lock around every call.
const std = @import("std");

pub const tag = "[yee-kad]";
/// The launch width KAD is designed for.
pub const kmax: u32 = 12;
/// Histogram buckets: widths above this are counted in the last bucket.
pub const hist_max: usize = 64;
pub const max_errors_printed: u32 = 5;
pub const max_policy_bytes: usize = 1 << 16;

pub const Mode = enum { off, fixed, ladder };
pub const Down = enum { accepted, step };

/// A normalized policy. The field defaults are the process default (ladder 5..12) used until a
/// valid policy file was read; a policy file without "mode" means "fixed" (see `parsePolicy`).
pub const Policy = struct {
    mode: Mode = .ladder,
    fixed: u32 = 5,
    base: u32 = 5,
    kmin: u32 = 5,
    kmax: u32 = 12,
    up_after: u32 = 1,
    up_step: u32 = 2,
    down: Down = .accepted,
    down_slack: u32 = 1,
    down_step: u32 = 2,
    hot_k: u32 = 12,
    hot_ttl: u32 = 8,
};

pub const default_policy: Policy = .{};

pub const PolicyError = error{ InvalidJson, NotObject, BadMode, BadDown, BadField, OutOfMemory };

const blanks = " \t\n\r\x0b\x0c";

/// Parse a policy file (a JSON object) with the field rules of `parse_policy` in the Python core:
/// "mode" defaults to "fixed"; integer fields reject booleans and non-integers and are raised to
/// their minimum; kmin > kmax lowers kmin; base is clamped to [kmin, kmax].
pub fn parsePolicy(gpa: std.mem.Allocator, bytes: []const u8) PolicyError!Policy {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .duplicate_field_behavior = .use_last }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidJson,
    };
    defer parsed.deinit();
    return policyFromValue(parsed.value);
}

pub fn policyFromValue(value: std.json.Value) PolicyError!Policy {
    const obj = switch (value) {
        .object => |o| o,
        else => return error.NotObject,
    };
    var p: Policy = .{ .mode = .fixed };
    if (obj.get("mode")) |m| {
        const s = switch (m) {
            .string => |s| s,
            else => return error.BadMode,
        };
        p.mode = parseMode(s) orelse return error.BadMode;
    }
    if (obj.get("down")) |d| {
        const s = switch (d) {
            .string => |s| s,
            else => return error.BadDown,
        };
        p.down = parseDown(s) orelse return error.BadDown;
    }
    p.fixed = try intField(obj, "fixed", 5, 1);
    p.base = try intField(obj, "base", 5, 1);
    p.kmin = try intField(obj, "kmin", 5, 1);
    p.kmax = try intField(obj, "kmax", 12, 1);
    p.up_after = try intField(obj, "up_after", 1, 1);
    p.up_step = try intField(obj, "up_step", 2, 0);
    p.down_slack = try intField(obj, "down_slack", 1, 0);
    p.down_step = try intField(obj, "down_step", 2, 0);
    p.hot_k = try intField(obj, "hot_k", 12, 1);
    p.hot_ttl = try intField(obj, "hot_ttl", 8, 0);
    if (p.kmin > p.kmax) p.kmin = p.kmax;
    p.base = @max(p.kmin, @min(p.base, p.kmax));
    return p;
}

fn parseMode(s: []const u8) ?Mode {
    const t = std.mem.trim(u8, s, blanks);
    if (std.ascii.eqlIgnoreCase(t, "off")) return .off;
    if (std.ascii.eqlIgnoreCase(t, "fixed")) return .fixed;
    if (std.ascii.eqlIgnoreCase(t, "ladder")) return .ladder;
    return null;
}

fn parseDown(s: []const u8) ?Down {
    const t = std.mem.trim(u8, s, blanks);
    if (std.ascii.eqlIgnoreCase(t, "accepted")) return .accepted;
    if (std.ascii.eqlIgnoreCase(t, "step")) return .step;
    return null;
}

fn intField(obj: std.json.ObjectMap, name: []const u8, default: u32, minimum: u32) PolicyError!u32 {
    const v = obj.get(name) orelse return default;
    switch (v) {
        .integer => |i| {
            if (i <= minimum) return minimum;
            if (i >= std.math.maxInt(u32)) return std.math.maxInt(u32);
            const r: u32 = @intCast(i);
            return r;
        },
        .number_string => |s| return bigInteger(s, minimum),
        else => return error.BadField,
    }
}

/// Integers outside i64 arrive as number strings; anything else there is not an integer.
fn bigInteger(s: []const u8, minimum: u32) PolicyError!u32 {
    var digits = s;
    const negative = digits.len > 0 and digits[0] == '-';
    if (negative) digits = digits[1..];
    if (digits.len == 0) return error.BadField;
    for (digits) |c| {
        if (c < '0' or c > '9') return error.BadField;
    }
    return if (negative) minimum else std.math.maxInt(u32);
}

/// The policy as a JSON object with sorted keys (the log line and the statistics file).
pub fn policyJson(buf: []u8, p: Policy) []const u8 {
    return std.fmt.bufPrint(buf, "{{\"base\": {d}, \"down\": \"{s}\", \"down_slack\": {d}, \"down_step\": {d}, \"fixed\": {d}, \"hot_k\": {d}, \"hot_ttl\": {d}, \"kmax\": {d}, \"kmin\": {d}, \"mode\": \"{s}\", \"up_after\": {d}, \"up_step\": {d}}}", .{
        p.base,  @tagName(p.down), p.down_slack, p.down_step, p.fixed,     p.hot_k,
        p.hot_ttl, p.kmax,         p.kmin,       @tagName(p.mode), p.up_after, p.up_step,
    }) catch "{}";
}

/// The ladder state of one request.
pub const State = struct {
    cur: u32,
    streak: u32 = 0,
    hot_until: i64 = -1,

    pub fn init(pol: Policy) State {
        return .{ .cur = pol.base };
    }
};

/// One verification of one request: `v` drafts verified, `a` accepted (no bonus token).
pub fn ladderUpdate(st: *State, v: u32, a: u32, pol: Policy) void {
    if (v == 0) return;
    var cur: u64 = st.cur;
    const lo: u64 = pol.kmin;
    const hi: u64 = pol.kmax;
    if (a >= v) {
        st.streak +|= 1;
        if (st.streak >= pol.up_after) {
            cur = @min(cur + pol.up_step, hi);
            st.streak = 0;
        }
    } else {
        st.streak = 0;
        cur = switch (pol.down) {
            .accepted => @max(lo, @min(cur, @as(u64, a) + pol.down_slack)),
            .step => @max(lo, cur -| pol.down_step),
        };
    }
    st.cur = @intCast(@max(lo, @min(cur, hi)));
}

/// The width of one request for step `step` (`decide_w` of the vLLM adapter).
pub fn decideK(st: ?*const State, pol: Policy, step: u64, kmax_run: u32) u32 {
    switch (pol.mode) {
        .off => return @max(1, kmax_run),
        .fixed => return @max(1, @min(pol.fixed, kmax_run)),
        .ladder => {},
    }
    var cur: u32 = pol.base;
    if (st) |s| {
        cur = s.cur;
        if (s.hot_until >= 0 and @as(u64, @intCast(s.hot_until)) >= step) cur = @max(cur, pol.hot_k);
    }
    const hi = @min(pol.kmax, kmax_run);
    return @max(1, @min(cur, @max(1, hi)));
}

/// File modification stamp used to detect a changed policy file.
pub const Stamp = struct { mtime_ns: i128, size: u64 };

/// File access supplied by the host (written with the engine's own std.Io code).
pub const Io = struct {
    ctx: ?*anyopaque = null,
    /// (mtime_ns, size) of `path`; null when it does not exist or cannot be stat'ed.
    stat: *const fn (ctx: ?*anyopaque, path: []const u8) ?Stamp,
    /// The whole file allocated with `gpa` (at most `limit` bytes); null on any error.
    read: *const fn (ctx: ?*anyopaque, gpa: std.mem.Allocator, path: []const u8, limit: usize) ?[]u8,
    /// Replace `path` by `bytes` atomically (temporary file + rename, parent directory created); false on error.
    write: *const fn (ctx: ?*anyopaque, path: []const u8, bytes: []const u8) bool,
};

pub const Options = struct {
    enabled: bool = false,
    /// The launch width of the engine (`setKmaxRun` when it is only known later).
    kmax_run: u32 = kmax,
    policy_file: []const u8 = "",
    stats_file: []const u8 = "",
    log_s: f64 = 30.0,
    reload_s: f64 = 1.0,
    /// Print the startup line, policy loads and the periodic statistics line.
    verbose: bool = true,
};

fn secondsToNs(s: f64) i128 {
    if (std.math.isNan(s) or s <= 0) return 0;
    if (s >= 1.0e9) return 1_000_000_000 * 1_000_000_000;
    return @intFromFloat(s * 1.0e9);
}

fn bucket(w: u64) usize {
    return @intCast(@min(w, hist_max));
}

fn mean(sum: u64, n: u64) f64 {
    if (n == 0) return 0;
    return @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
}

fn fmtHist(buf: []u8, hist: []const u64) []const u8 {
    var n: usize = 0;
    for (hist, 0..) |c, w| {
        if (c == 0) continue;
        const piece = std.fmt.bufPrint(buf[n..], "{s}{d}:{d}", .{ if (n == 0) "" else ",", w, c }) catch break;
        n += piece.len;
    }
    return buf[0..n];
}

fn appendHist(out: *std.ArrayList(u8), gpa: std.mem.Allocator, hist: []const u64) std.mem.Allocator.Error!void {
    try out.append(gpa, '{');
    var first = true;
    for (hist, 0..) |c, w| {
        if (c == 0) continue;
        if (!first) try out.appendSlice(gpa, ", ");
        first = false;
        try out.print(gpa, "\"{d}\": {d}", .{ w, c });
    }
    try out.append(gpa, '}');
}

/// Per-request adaptive draft width for one engine process.
pub const Kad = struct {
    gpa: std.mem.Allocator,
    io: ?Io,
    /// Engine name in logs and statistics (not owned; pass a string literal).
    base: []const u8,
    enabled: bool,
    verbose: bool,
    kmax_run: u32 = kmax,
    policy_file: []u8,
    stats_file: []u8,
    log_ns: i128,
    reload_ns: i128,
    step: u64 = 0,
    policy: Policy = default_policy,
    policy_sha: ?[64]u8 = null,
    stamp: ?Stamp = null,
    last_stat_ns: ?i128 = null,
    last_log_ns: ?i128 = null,
    warned_mask: u64 = 0,
    states: std.AutoHashMapUnmanaged(u64, State) = .empty,
    noted_steps: u64 = 0,
    mixed_steps: u64 = 0,
    w_hist: [hist_max + 1]u64 = @splat(0),
    b_hist: [hist_max + 1]u64 = @splat(0),
    v_steps: u64 = 0,
    sum_v: u64 = 0,
    sum_a: u64 = 0,
    acc_n: [hist_max + 1]u64 = @splat(0),
    acc_a: [hist_max + 1]u64 = @splat(0),
    errors: u64 = 0,
    errors_printed: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, io: ?Io, base: []const u8, opts: Options) std.mem.Allocator.Error!Kad {
        const policy_file = try gpa.dupe(u8, opts.policy_file);
        errdefer gpa.free(policy_file);
        const stats_file = try gpa.dupe(u8, opts.stats_file);
        var self: Kad = .{
            .gpa = gpa,
            .io = io,
            .base = base,
            .enabled = opts.enabled,
            .verbose = opts.verbose,
            .policy_file = policy_file,
            .stats_file = stats_file,
            .log_ns = secondsToNs(opts.log_s),
            .reload_ns = secondsToNs(opts.reload_s),
        };
        self.setKmaxRun(opts.kmax_run);
        if (self.enabled and self.verbose) {
            std.debug.print("{s} {s} enabled: launch K={d}, policy {s}, stats {s}\n", .{ tag, base, self.kmax_run, policy_file, stats_file });
        }
        return self;
    }

    pub fn deinit(self: *Kad) void {
        self.states.deinit(self.gpa);
        self.gpa.free(self.policy_file);
        self.gpa.free(self.stats_file);
        self.* = undefined;
    }

    /// Set the launch width of the engine; a width other than 12 prints one warning per value.
    pub fn setKmaxRun(self: *Kad, k: u32) void {
        const kk: u32 = @max(1, k);
        self.kmax_run = kk;
        const bit: u6 = @intCast(@min(kk, 63));
        const mask = @as(u64, 1) << bit;
        if (self.enabled and kk != kmax and (self.warned_mask & mask) == 0) {
            self.warned_mask |= mask;
            std.debug.print("{s} {s} WARNING launch K={d}, KAD needs {d} (per-request width is capped at min({d}, K))\n", .{ tag, self.base, kk, kmax, kmax });
        }
    }

    /// Call once per engine step before any `width`: advances the step, reloads the policy file
    /// at most every reload_s seconds, and logs / writes the statistics every log_s seconds.
    pub fn beginStep(self: *Kad, now_ns: i128) u64 {
        self.step += 1;
        if (!self.enabled) return self.step;
        self.reload(now_ns);
        self.maybeLog(now_ns, false);
        return self.step;
    }

    fn reload(self: *Kad, now_ns: i128) void {
        if (self.last_stat_ns) |t| {
            if (now_ns - t < self.reload_ns) return;
        }
        self.last_stat_ns = now_ns;
        const io = self.io orelse return;
        if (self.policy_file.len == 0) return;
        const st = io.stat(io.ctx, self.policy_file) orelse return; // no file yet: keep the last valid policy
        if (self.stamp) |s| {
            if (s.mtime_ns == st.mtime_ns and s.size == st.size) return;
        }
        self.stamp = st;
        const bytes = io.read(io.ctx, self.gpa, self.policy_file, max_policy_bytes) orelse {
            self.noteError("policy read", error.ReadFailed);
            return;
        };
        defer self.gpa.free(bytes);
        _ = self.applyPolicy(bytes);
    }

    /// Use `bytes` (a policy file) when it is a valid policy; otherwise keep the last one (errors + 1).
    pub fn applyPolicy(self: *Kad, bytes: []const u8) bool {
        const pol = parsePolicy(self.gpa, bytes) catch |e| {
            self.noteError("policy load", e);
            return false;
        };
        var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const sha = std.fmt.bytesToHex(digest, .lower);
        self.policy = pol;
        self.policy_sha = sha;
        if (self.verbose) {
            var buf: [512]u8 = undefined;
            std.debug.print("{s} {s} policy loaded sha={s} {s}\n", .{ tag, self.base, sha[0..12], policyJson(&buf, pol) });
        }
        self.writeStats();
        return true;
    }

    /// The draft width of one request for this step (the launch width when off or structured).
    pub fn width(self: *const Kad, rid: u64, structured: bool) u32 {
        return self.widthOf(self.states.getPtr(rid), structured);
    }

    /// `width` for a request whose ladder state the host keeps itself (`st` null: none yet).
    pub fn widthOf(self: *const Kad, st: ?*const State, structured: bool) u32 {
        if (!self.enabled or structured) return self.kmax_run;
        return decideK(st, self.policy, self.step, self.kmax_run);
    }

    /// Record the widths decided for one step and, for engines that run one width per batch, the width run.
    pub fn noteBatch(self: *Kad, widths: []const u32, k_used: ?u32) void {
        if (!self.enabled) return;
        if (widths.len > 0) {
            self.noted_steps += 1;
            var mixed = false;
            for (widths) |w| {
                self.w_hist[bucket(w)] += 1;
                if (w != widths[0]) mixed = true;
            }
            if (mixed) self.mixed_steps += 1;
        }
        if (k_used) |k| self.b_hist[bucket(k)] += 1;
    }

    /// One verification of one request: `v` drafts actually verified, `a` accepted (no bonus token).
    pub fn observe(self: *Kad, rid: u64, v: u32, a: u32) void {
        if (!self.enabled or v == 0) return;
        if (self.policy.mode != .ladder) return self.record(null, v, a);
        const gop = self.states.getOrPut(self.gpa, rid) catch |e| {
            self.noteError("observe", e);
            return;
        };
        if (!gop.found_existing) gop.value_ptr.* = State.init(self.policy);
        self.record(gop.value_ptr, v, a);
    }

    /// `observe` for a request whose ladder state the host keeps (`st.*` null: created here).
    pub fn observeOn(self: *Kad, st: *?State, v: u32, a: u32) void {
        if (!self.enabled or v == 0) return;
        if (self.policy.mode != .ladder) return self.record(null, v, a);
        if (st.* == null) st.* = State.init(self.policy);
        self.record(&st.*.?, v, a);
    }

    fn record(self: *Kad, st: ?*State, v: u32, a: u32) void {
        const acc: u32 = @min(a, v);
        if (st) |s| ladderUpdate(s, v, acc, self.policy);
        self.v_steps += 1;
        self.sum_v += v;
        self.sum_a += acc;
        self.acc_n[bucket(v)] += 1;
        self.acc_a[bucket(v)] += acc;
    }

    fn hotUntil(self: *const Kad) i64 {
        return @intCast(@min(self.step + self.policy.hot_ttl, std.math.maxInt(i64)));
    }

    /// A recall hit: width max(cur, hot_k) for the next hot_ttl steps.
    pub fn markHot(self: *Kad, rid: u64) void {
        if (!self.enabled or self.policy.mode != .ladder) return;
        const gop = self.states.getOrPut(self.gpa, rid) catch |e| {
            self.noteError("hot", e);
            return;
        };
        if (!gop.found_existing) gop.value_ptr.* = State.init(self.policy);
        gop.value_ptr.hot_until = self.hotUntil();
    }

    /// `markHot` for a request whose ladder state the host keeps.
    pub fn markHotOn(self: *Kad, st: *?State) void {
        if (!self.enabled or self.policy.mode != .ladder) return;
        if (st.* == null) st.* = State.init(self.policy);
        st.*.?.hot_until = self.hotUntil();
    }

    /// The request finished: forget its state.
    pub fn release(self: *Kad, rid: u64) void {
        _ = self.states.remove(rid);
    }

    /// Keep only the states of the requests in `live` (for hosts without a reliable end event).
    pub fn retain(self: *Kad, live: []const u64) void {
        var drop: [64]u64 = undefined;
        while (true) {
            var n: usize = 0;
            var it = self.states.keyIterator();
            while (it.next()) |key| {
                if (std.mem.indexOfScalar(u64, live, key.*) != null) continue;
                drop[n] = key.*;
                n += 1;
                if (n == drop.len) break;
            }
            for (drop[0..n]) |key| _ = self.states.remove(key);
            if (n < drop.len) return;
        }
    }

    pub fn tracked(self: *const Kad) usize {
        return self.states.count();
    }

    pub fn state(self: *const Kad, rid: u64) ?State {
        return if (self.states.getPtr(rid)) |p| p.* else null;
    }

    /// Count one internal error; the first `max_errors_printed` are printed.
    pub fn noteError(self: *Kad, what: []const u8, err: anyerror) void {
        self.errors += 1;
        if (self.errors_printed < max_errors_printed) {
            self.errors_printed += 1;
            std.debug.print("{s} {s} {s} failed, the step keeps the native width: {s}\n", .{ tag, self.base, what, @errorName(err) });
        }
    }

    /// One statistics line and one statistics file write every log_s seconds (`force`: now).
    pub fn maybeLog(self: *Kad, now_ns: i128, force: bool) void {
        if (!self.enabled) return;
        const last = self.last_log_ns orelse blk: {
            self.last_log_ns = now_ns;
            break :blk now_ns;
        };
        if (!force and (self.log_ns <= 0 or now_ns - last < self.log_ns)) return;
        self.last_log_ns = now_ns;
        if (self.verbose) {
            var wb: [1024]u8 = undefined;
            var bb: [1024]u8 = undefined;
            const sha: []const u8 = if (self.policy_sha) |*s| s[0..12] else "";
            std.debug.print("{s} {s} steps={d} mixed={d} w_hist={s} b_hist={s} mean_v={d:.3} mean_a={d:.3} tracked={d} errors={d} sha={s}\n", .{
                tag,                      self.base,                self.step, self.mixed_steps,
                fmtHist(&wb, &self.w_hist), fmtHist(&bb, &self.b_hist), mean(self.sum_v, self.v_steps), mean(self.sum_a, self.v_steps),
                self.tracked(),           self.errors,              sha,
            });
        }
        self.writeStats();
    }

    /// Write the statistics JSON to the stats file (when the host gave one).
    pub fn writeStats(self: *Kad) void {
        const io = self.io orelse return;
        if (self.stats_file.len == 0) return;
        const text = self.snapshotJson(self.gpa) catch |e| {
            self.noteError("stats", e);
            return;
        };
        defer self.gpa.free(text);
        if (!io.write(io.ctx, self.stats_file, text)) self.noteError("stats write", error.WriteFailed);
    }

    /// The statistics as JSON (same keys as the Python core's snapshot).
    pub fn snapshotJson(self: *const Kad, gpa: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        var pb: [512]u8 = undefined;
        const sha: []const u8 = if (self.policy_sha) |*s| s[0..] else "";
        try out.print(gpa, "{{\"base\": \"{s}\", \"enabled\": {s}, \"mode\": \"{s}\", \"policy\": {s}, \"policy_sha256\": \"{s}\", \"kmax_run\": {d}, \"steps\": {d}, \"noted_steps\": {d}, \"mixed_steps\": {d}, \"w_hist\": ", .{
            self.base, if (self.enabled) "true" else "false", @tagName(self.policy.mode), policyJson(&pb, self.policy), sha,
            self.kmax_run, self.step, self.noted_steps, self.mixed_steps,
        });
        try appendHist(&out, gpa, &self.w_hist);
        try out.appendSlice(gpa, ", \"b_hist\": ");
        try appendHist(&out, gpa, &self.b_hist);
        try out.print(gpa, ", \"v_steps\": {d}, \"sum_v\": {d}, \"sum_a\": {d}, \"acc_by_v\": {{", .{ self.v_steps, self.sum_v, self.sum_a });
        var first = true;
        for (self.acc_n, 0..) |n, v| {
            if (n == 0) continue;
            if (!first) try out.appendSlice(gpa, ", ");
            first = false;
            try out.print(gpa, "\"{d}\": [{d}, {d}]", .{ v, n, self.acc_a[v] });
        }
        try out.print(gpa, "}}, \"tracked\": {d}, \"errors\": {d}}}", .{ self.tracked(), self.errors });
        return out.toOwnedSlice(gpa);
    }
};

// ------------------------------------------------------------------------------------------------ tests
const testing = std.testing;
const ns: i128 = 1_000_000_000;

const FakeFs = struct {
    gpa: std.mem.Allocator,
    policy: ?[]const u8 = null,
    mtime: i128 = 1,
    reads: u32 = 0,
    stats: ?[]u8 = null,
    fail_write: bool = false,

    fn io(self: *FakeFs) Io {
        return .{ .ctx = self, .stat = &FakeFs.stat, .read = &FakeFs.read, .write = &FakeFs.write };
    }

    fn stat(ctx: ?*anyopaque, path: []const u8) ?Stamp {
        _ = path;
        const self: *FakeFs = @ptrCast(@alignCast(ctx.?));
        const p = self.policy orelse return null;
        return .{ .mtime_ns = self.mtime, .size = p.len };
    }

    fn read(ctx: ?*anyopaque, gpa: std.mem.Allocator, path: []const u8, limit: usize) ?[]u8 {
        _ = path;
        const self: *FakeFs = @ptrCast(@alignCast(ctx.?));
        self.reads += 1;
        const p = self.policy orelse return null;
        if (p.len > limit) return null;
        return gpa.dupe(u8, p) catch null;
    }

    fn write(ctx: ?*anyopaque, path: []const u8, bytes: []const u8) bool {
        _ = path;
        const self: *FakeFs = @ptrCast(@alignCast(ctx.?));
        if (self.fail_write) return false;
        const copy = self.gpa.dupe(u8, bytes) catch return false;
        if (self.stats) |s| self.gpa.free(s);
        self.stats = copy;
        return true;
    }

    fn deinit(self: *FakeFs) void {
        if (self.stats) |s| self.gpa.free(s);
    }
};

test "kad: parsePolicy: defaults, words, field rules" {
    const gpa = testing.allocator;
    const p = try parsePolicy(gpa, "{}");
    try testing.expectEqual(Mode.fixed, p.mode);
    var want: Policy = .{ .mode = .fixed };
    try testing.expectEqual(want, p);
    const l = try parsePolicy(gpa, "{\"mode\": \" LADDER \", \"down\": \"Step\"}");
    try testing.expectEqual(Mode.ladder, l.mode);
    try testing.expectEqual(Down.step, l.down);
    try testing.expectError(error.BadMode, parsePolicy(gpa, "{\"mode\": \"turbo\"}"));
    try testing.expectError(error.BadMode, parsePolicy(gpa, "{\"mode\": 1}"));
    try testing.expectError(error.BadDown, parsePolicy(gpa, "{\"down\": \"half\"}"));
    try testing.expectError(error.BadField, parsePolicy(gpa, "{\"kmax\": true}"));
    try testing.expectError(error.BadField, parsePolicy(gpa, "{\"kmax\": 12.0}"));
    try testing.expectError(error.BadField, parsePolicy(gpa, "{\"kmax\": \"12\"}"));
    try testing.expectError(error.BadField, parsePolicy(gpa, "{\"kmax\": null}"));
    try testing.expectError(error.NotObject, parsePolicy(gpa, "[1, 2]"));
    try testing.expectError(error.InvalidJson, parsePolicy(gpa, "{\"mode\": "));
    const m = try parsePolicy(gpa, "{\"fixed\": -3, \"up_step\": -1, \"hot_ttl\": 99999999999999999999, \"base\": -99999999999999999999}");
    try testing.expectEqual(@as(u32, 1), m.fixed);
    try testing.expectEqual(@as(u32, 0), m.up_step);
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), m.hot_ttl);
    try testing.expectEqual(@as(u32, 5), m.base); // raised to 1, then clamped to kmin 5
    const c = try parsePolicy(gpa, "{\"kmin\": 9, \"kmax\": 7, \"base\": 3}");
    try testing.expectEqual(@as(u32, 7), c.kmin);
    try testing.expectEqual(@as(u32, 7), c.base);
    const b = try parsePolicy(gpa, "{\"base\": 40}");
    try testing.expectEqual(@as(u32, 12), b.base);
    const d = try parsePolicy(gpa, "{\"mode\": \"ladder\", \"mode\": \"off\"}");
    try testing.expectEqual(Mode.off, d.mode); // the last duplicate wins, as in Python
    want = default_policy;
    var buf: [512]u8 = undefined;
    try testing.expectEqual(want, try parsePolicy(gpa, policyJson(&buf, default_policy)));
}

test "kad: ladder: 5 7 9 11 12 while accepted, accepted + 1 after a rejection" {
    const pol = default_policy;
    var st = State.init(pol);
    var seen: [6]u32 = undefined;
    for (&seen) |*k| {
        k.* = decideK(&st, pol, 1, kmax);
        ladderUpdate(&st, k.*, k.*, pol);
    }
    try testing.expectEqualSlices(u32, &.{ 5, 7, 9, 11, 12, 12 }, &seen);
    ladderUpdate(&st, 12, 3, pol);
    try testing.expectEqual(@as(u32, 5), decideK(&st, pol, 1, kmax));
    ladderUpdate(&st, 5, 5, pol);
    ladderUpdate(&st, 7, 7, pol);
    ladderUpdate(&st, 9, 7, pol);
    try testing.expectEqual(@as(u32, 8), st.cur); // max(5, min(9, 7 + 1))
    var sp = pol;
    sp.down = .step;
    var s2 = State{ .cur = 12 };
    ladderUpdate(&s2, 12, 0, sp);
    try testing.expectEqual(@as(u32, 10), s2.cur);
    var s3 = State{ .cur = 6 };
    ladderUpdate(&s3, 6, 0, sp);
    try testing.expectEqual(@as(u32, 5), s3.cur);
    var up2 = pol;
    up2.up_after = 2;
    var s4 = State.init(up2);
    ladderUpdate(&s4, 5, 5, up2);
    try testing.expectEqual(@as(u32, 5), s4.cur);
    ladderUpdate(&s4, 5, 5, up2);
    try testing.expectEqual(@as(u32, 7), s4.cur);
    ladderUpdate(&s4, 0, 0, up2);
    try testing.expectEqual(@as(u32, 7), s4.cur);
}

test "kad: decideK: off, fixed, hot, launch cap" {
    var p = default_policy;
    p.mode = .off;
    try testing.expectEqual(@as(u32, 12), decideK(null, p, 1, 12));
    p.mode = .fixed;
    p.fixed = 7;
    try testing.expectEqual(@as(u32, 7), decideK(null, p, 1, 12));
    try testing.expectEqual(@as(u32, 5), decideK(null, p, 1, 5));
    p = default_policy;
    try testing.expectEqual(@as(u32, 5), decideK(null, p, 1, 12));
    const st = State{ .cur = 5, .hot_until = 10 };
    try testing.expectEqual(@as(u32, 12), decideK(&st, p, 10, 12));
    try testing.expectEqual(@as(u32, 5), decideK(&st, p, 11, 12));
    try testing.expectEqual(@as(u32, 7), decideK(&st, p, 10, 7));
}

test "kad: Kad: off keeps the launch width and records nothing" {
    var k = try Kad.init(testing.allocator, null, "test", .{ .enabled = false });
    defer k.deinit();
    _ = k.beginStep(0);
    try testing.expectEqual(@as(u32, 12), k.width(1, false));
    k.observe(1, 12, 3);
    k.noteBatch(&.{12}, 12);
    k.markHot(1);
    try testing.expectEqual(@as(u64, 0), k.v_steps);
    try testing.expectEqual(@as(u64, 0), k.noted_steps);
    try testing.expectEqual(@as(usize, 0), k.tracked());
}

test "kad: Kad: per-request ladders, structured, hot, release, launch K" {
    var k = try Kad.init(testing.allocator, null, "test", .{ .enabled = true, .verbose = false });
    defer k.deinit();
    var lens: [5]u32 = undefined;
    for (&lens) |*n| {
        _ = k.beginStep(0);
        n.* = k.width(7, false);
        k.observe(7, n.*, n.*);
    }
    try testing.expectEqualSlices(u32, &.{ 5, 7, 9, 11, 12 }, &lens);
    _ = k.beginStep(0);
    try testing.expectEqual(@as(u32, 12), k.width(1, true));
    try testing.expectEqual(@as(u32, 5), k.width(8, false));
    k.markHot(8); // step 6: hot until step 14
    try testing.expectEqual(@as(u32, 12), k.width(8, false));
    for (0..8) |_| _ = k.beginStep(0);
    try testing.expectEqual(@as(u32, 12), k.width(8, false));
    _ = k.beginStep(0);
    try testing.expectEqual(@as(u32, 5), k.width(8, false));
    k.observe(9, 0, 0);
    k.observe(9, 5, 99); // a is clamped to v
    try testing.expectEqual(@as(u32, 7), k.state(9).?.cur);
    k.release(7);
    try testing.expectEqual(@as(u32, 5), k.width(7, false));
    try testing.expectEqual(@as(usize, 2), k.tracked());
    for (100..180) |rid| k.observe(rid, 5, 5);
    try testing.expectEqual(@as(usize, 82), k.tracked());
    k.retain(&.{ 9, 150 });
    try testing.expectEqual(@as(usize, 2), k.tracked());
    try testing.expectEqual(@as(u32, 7), k.state(150).?.cur);
    try testing.expect(k.state(8) == null);
    k.setKmaxRun(9);
    try testing.expectEqual(@as(u32, 9), k.width(7, true));
    try testing.expectEqual(@as(u32, 7), k.width(9, false));
}

test "kad: Kad: policy hot reload, sha, bad file keeps the last policy, statistics JSON" {
    const gpa = testing.allocator;
    var fs: FakeFs = .{ .gpa = gpa };
    defer fs.deinit();
    var k = try Kad.init(gpa, fs.io(), "test", .{ .enabled = true, .verbose = false, .policy_file = "p.json", .stats_file = "s.json" });
    defer k.deinit();
    _ = k.beginStep(0);
    try testing.expectEqual(Mode.ladder, k.policy.mode);
    fs.policy = "{\"mode\": \"fixed\", \"fixed\": 7}";
    _ = k.beginStep(ns / 2); // inside the 1 s interval: not read
    try testing.expectEqual(Mode.ladder, k.policy.mode);
    _ = k.beginStep(ns);
    try testing.expectEqual(Mode.fixed, k.policy.mode);
    try testing.expectEqual(@as(u32, 7), k.width(1, false));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fs.policy.?, &digest, .{});
    try testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), &k.policy_sha.?);
    const reads = fs.reads;
    _ = k.beginStep(3 * ns); // same (mtime, size): not read again
    try testing.expectEqual(reads, fs.reads);
    fs.policy = "{\"mode\": \"warp\"}";
    fs.mtime = 2;
    _ = k.beginStep(5 * ns);
    try testing.expectEqual(Mode.fixed, k.policy.mode);
    try testing.expectEqual(@as(u64, 1), k.errors);
    fs.policy = null;
    _ = k.beginStep(7 * ns);
    try testing.expectEqual(Mode.fixed, k.policy.mode);
    fs.policy = "{\"mode\": \"ladder\"}";
    fs.mtime = 3;
    _ = k.beginStep(9 * ns);
    try testing.expectEqual(Mode.ladder, k.policy.mode);
    k.noteBatch(&.{ 5, 7 }, 7);
    k.observe(1, 7, 7);
    k.observe(2, 7, 2);
    k.maybeLog(10 * ns, true);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fs.stats.?, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("test", o.get("base").?.string);
    try testing.expectEqual(true, o.get("enabled").?.bool);
    try testing.expectEqualStrings("ladder", o.get("mode").?.string);
    try testing.expectEqualStrings("ladder", o.get("policy").?.object.get("mode").?.string);
    try testing.expectEqual(@as(usize, 64), o.get("policy_sha256").?.string.len);
    try testing.expectEqual(@as(i64, 12), o.get("kmax_run").?.integer);
    try testing.expectEqual(@as(i64, 7), o.get("steps").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("noted_steps").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("mixed_steps").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("w_hist").?.object.get("5").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("b_hist").?.object.get("7").?.integer);
    try testing.expectEqual(@as(i64, 2), o.get("v_steps").?.integer);
    try testing.expectEqual(@as(i64, 14), o.get("sum_v").?.integer);
    try testing.expectEqual(@as(i64, 9), o.get("sum_a").?.integer);
    const acc = o.get("acc_by_v").?.object.get("7").?.array.items;
    try testing.expectEqual(@as(i64, 2), acc[0].integer);
    try testing.expectEqual(@as(i64, 9), acc[1].integer);
    try testing.expectEqual(@as(i64, 2), o.get("tracked").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("errors").?.integer);
    fs.fail_write = true;
    k.maybeLog(11 * ns, true);
    try testing.expectEqual(@as(u64, 2), k.errors);
}

test "kad: host-kept states (widthOf, observeOn, markHotOn)" {
    var k = try Kad.init(testing.allocator, null, "test", .{ .enabled = true, .verbose = false });
    defer k.deinit();
    var st: ?State = null;
    _ = k.beginStep(0);
    try testing.expectEqual(@as(u32, 5), k.widthOf(if (st) |*s| s else null, false));
    k.observeOn(&st, 5, 5);
    try testing.expectEqual(@as(u32, 7), k.widthOf(if (st) |*s| s else null, false));
    k.observeOn(&st, 7, 3);
    try testing.expectEqual(@as(u32, 5), st.?.cur);
    k.markHotOn(&st);
    try testing.expectEqual(@as(u32, 12), k.widthOf(if (st) |*s| s else null, false));
    try testing.expectEqual(@as(u32, 12), k.widthOf(null, true));
    try testing.expectEqual(@as(usize, 0), k.tracked());
    try testing.expectEqual(@as(u64, 2), k.v_steps);
    var off = try Kad.init(testing.allocator, null, "test", .{ .enabled = false });
    defer off.deinit();
    var none: ?State = null;
    off.observeOn(&none, 5, 5);
    off.markHotOn(&none);
    try testing.expect(none == null);
}