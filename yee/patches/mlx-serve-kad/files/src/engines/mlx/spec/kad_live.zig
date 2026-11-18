//! Yee KAD for mlx-serve DSpark (MLX_SERVE_KAD=1 or YEE_KAD=1): per-request verify width 5..12 within the model's
//! DSpark block (`"dspark_block_size": 12` in its config.json).
//!
//! A stage round (`dsparkPolicyWith`, the stage slots of `moduleGroupRound`) verifies at most the request's KAD width;
//! the confidence policy still picks the prefix inside it. Copy and recall rounds keep their own length and teach the
//! ladder; a recall round that lands drafts gives the request the full width for `hot_ttl` rounds. The ladder state
//! of a request lives in its `d_recall.Stream` (`kad` field, dropped with the request); one process-wide controller
//! (policy file, statistics) sits behind a spin lock. Off by default; off = the 2.4 behaviour, branch for branch.
const std = @import("std");
const kad = @import("kad.zig");
const kad_stdio = @import("kad_stdio.zig");
const d_recall = @import("d_recall.zig");

pub const base = "mlx-serve";

const Phase = enum(u8) { unknown, off, on };

var lock: std.atomic.Mutex = .unlocked;
var phase: Phase = .unknown;
var files: kad_stdio.Files = undefined;
var ctl: kad.Kad = undefined;

fn acquire() void {
    while (!lock.tryLock()) std.atomic.spinLoopHint();
}

fn env(comptime name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " \t\r\n");
    return if (t.len == 0) null else t;
}

fn on(comptime name: [:0]const u8) bool {
    const t = env(name) orelse return false;
    for ([_][]const u8{ "1", "true", "yes", "on" }) |w| if (std.ascii.eqlIgnoreCase(t, w)) return true;
    return false;
}

fn seconds(comptime a: [:0]const u8, comptime b: [:0]const u8, default: f64) f64 {
    const t = env(a) orelse env(b) orelse return default;
    return std.fmt.parseFloat(f64, t) catch default;
}

/// MLX_SERVE_KAD or YEE_KAD is 1 / true / yes / on.
pub fn enabled() bool {
    return on("MLX_SERVE_KAD") or on("YEE_KAD");
}

/// Under the lock: the controller, started on first use; null when KAD is off.
fn get(io: std.Io, launch_k: usize) ?*kad.Kad {
    if (phase == .unknown) start(io);
    if (phase != .on) return null;
    const k: u32 = @intCast(@min(launch_k, std.math.maxInt(u32)));
    if (ctl.kmax_run != k) ctl.setKmaxRun(k);
    return &ctl;
}

fn start(io: std.Io) void {
    phase = .off;
    if (!enabled()) return;
    startWith(io) catch |e| std.debug.print("{s} {s} not started: {s}\n", .{ kad.tag, base, @errorName(e) });
}

fn startWith(io: std.Io) !void {
    const gpa = std.heap.page_allocator;
    const home = env("HOME") orelse ".";
    const dir = if (env("YEE_KAD_DIR")) |d|
        (if (d[0] == '~') try std.fmt.allocPrint(gpa, "{s}{s}", .{ home, d[1..] }) else try gpa.dupe(u8, d))
    else
        try std.fmt.allocPrint(gpa, "{s}/.cache/yee/kad", .{home});
    defer gpa.free(dir);
    const policy = if (env("MLX_SERVE_KAD_POLICY") orelse env("YEE_KAD_POLICY")) |p| try gpa.dupe(u8, p) else try std.fmt.allocPrint(gpa, "{s}/kad_policy.json", .{dir});
    defer gpa.free(policy);
    const stats = if (env("MLX_SERVE_KAD_STATS_FILE") orelse env("YEE_KAD_STATS_FILE")) |p| try gpa.dupe(u8, p) else try std.fmt.allocPrint(gpa, "{s}/kad_stats_{s}.json", .{ dir, base });
    defer gpa.free(stats);
    files = .{ .io = io };
    ctl = try kad.Kad.init(gpa, files.table(), base, .{
        .enabled = true,
        .kmax_run = kad.kmax,
        .policy_file = policy,
        .stats_file = stats,
        .log_s = seconds("MLX_SERVE_KAD_LOG_S", "YEE_KAD_LOG_S", 30.0),
        .reload_s = seconds("MLX_SERVE_KAD_RELOAD_S", "YEE_KAD_RELOAD_S", 1.0),
    });
    phase = .on;
}

fn stateOf(st: *const d_recall.Stream) ?*const kad.State {
    return if (st.kad) |*s| s else null;
}

/// One round begins (a single request's stage round or a group round): true when KAD is on.
pub fn beginRound(io: std.Io, launch_k: usize) bool {
    if (phase == .off) return false;
    acquire();
    defer lock.unlock();
    const k = get(io, launch_k) orelse return false;
    _ = k.beginStep(files.nowNs());
    return true;
}

/// The request's width for this round (after `beginRound` returned true).
pub fn width(st: *const d_recall.Stream) u32 {
    acquire();
    defer lock.unlock();
    return ctl.widthOf(stateOf(st), false);
}

/// The widths decided in this round (statistics).
pub fn note(widths: []const u32) void {
    acquire();
    defer lock.unlock();
    ctl.noteBatch(widths, null);
}

/// A single request's stage round: `rows` drafts picked by the confidence policy, capped at its width.
pub fn capStage(io: std.Io, launch_k: usize, st: *const d_recall.Stream, rows: usize) usize {
    if (!beginRound(io, launch_k)) return rows;
    const w = width(st);
    note(&.{w});
    return @min(rows, @as(usize, w));
}

/// One verified round of a request: `v` drafts verified, `a` accepted (no bonus token); `recalled`: the drafts
/// came from `d_recall` (a hit with `a` > 0 opens the full width for hot_ttl rounds).
pub fn observe(st: *d_recall.Stream, v: usize, a: usize, recalled: bool) void {
    if (phase != .on) return;
    acquire();
    defer lock.unlock();
    const vv: u32 = @intCast(@min(v, std.math.maxInt(u32)));
    const aa: u32 = @intCast(@min(a, std.math.maxInt(u32)));
    ctl.observeOn(&st.kad, vv, aa);
    if (recalled and aa > 0) ctl.markHotOn(&st.kad);
}

test "kad_live: off without MLX_SERVE_KAD / YEE_KAD leaves every length alone" {
    if (enabled()) return error.SkipZigTest;
    var st: d_recall.Stream = .{};
    try std.testing.expectEqual(@as(usize, 9), capStage(std.testing.io, 12, &st, 9));
    try std.testing.expect(!beginRound(std.testing.io, 12));
    observe(&st, 9, 9, false);
    try std.testing.expect(st.kad == null);
}

test "kad_live: the controller on a request's d_recall.Stream" {
    const saved = .{ phase, ctl };
    defer {
        phase = saved[0];
        ctl = saved[1];
    }
    ctl = try kad.Kad.init(std.testing.allocator, null, base, .{ .enabled = true, .verbose = false });
    defer ctl.deinit();
    phase = .on;
    files = .{ .io = std.testing.io };
    var st: d_recall.Stream = .{};
    try std.testing.expectEqual(@as(usize, 5), capStage(std.testing.io, 12, &st, 9));
    try std.testing.expectEqual(@as(usize, 3), capStage(std.testing.io, 12, &st, 3));
    observe(&st, 5, 5, false);
    try std.testing.expectEqual(@as(usize, 7), capStage(std.testing.io, 12, &st, 12));
    observe(&st, 7, 2, false);
    try std.testing.expectEqual(@as(usize, 5), capStage(std.testing.io, 12, &st, 12));
    observe(&st, 20, 4, true);
    try std.testing.expectEqual(@as(usize, 12), capStage(std.testing.io, 12, &st, 12));
    try std.testing.expectEqual(@as(u32, 12), width(&st));
    try std.testing.expectEqual(@as(usize, 9), capStage(std.testing.io, 9, &st, 12));
    try std.testing.expectEqual(@as(u64, 3), ctl.v_steps);
}
