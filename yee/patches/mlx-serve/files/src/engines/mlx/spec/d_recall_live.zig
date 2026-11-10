//! The process's one recall record (`d_recall.zig`) behind the DeepSeek-V4.1 DSpark copy rounds. Every hook sits
//! beside a forward, so every caller is the inference thread and the record takes no lock.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const d_recall = @import("d_recall.zig");
const Shared = @import("../../../sdk/util/shared.zig").Shared;

pub const Stream = d_recall.Stream;

const Live = struct {
    record: ?d_recall.Recall = null,
    failed: bool = false,
};

pub const recall_shared = Shared(Live, "d_recall", .{});

pub fn enabled() bool {
    return if (env.get(.MLX_SERVE_D_RECALL)) |v| v[0] != '0' else true;
}

/// The record, allocated on first use; null while recall is off or after its allocation failed.
fn record() ?*d_recall.Recall {
    const live = recall_shared.get();
    if (live.record) |*r| return r;
    if (live.failed or !enabled()) return null;
    const cap = d_recall.capFor(env.float(.MLX_SERVE_D_RECALL_GIB, d_recall.DEFAULT_GIB));
    const density = d_recall.indexDensity(env.float(.MLX_SERVE_D_RECALL_INDEX_DENSITY, d_recall.DEFAULT_INDEX_DENSITY));
    const slots = d_recall.slotsFor(cap, density);
    live.record = d_recall.Recall.init(std.heap.page_allocator, cap, density) catch {
        live.failed = true;
        log.info("  [d-recall] off: a record of {d} ids and {d} index slots did not allocate\n", .{ cap, slots });
        return null;
    };
    log.info("  [d-recall] record of {d} ids, index {d} slots (density {d:.4}) ({d} MiB)\n", .{ cap, slots, density, live.record.?.store.bytes() >> 20 });
    return &live.record.?;
}

/// Arms a DSpark request and records the prompt ids it forwards (the part no cache held).
pub fn begin(forwarded: []const u32) Stream {
    const r = record() orelse return .{};
    _ = r.store.add(forwarded);
    return .{ .armed = true };
}

/// Recalled drafts for the context `ctx ++ [t1]` in `out` (sized to the verify and the token budget): more than
/// `block` of them, or null.
pub fn draft(st: *Stream, ctx: []const u32, t1: u32, out: []u32, block: usize) ?d_recall.Draft {
    if (!st.armed) return null;
    const r = record() orelse return null;
    return r.draft(st, ctx, t1, out, block);
}

/// What the target kept of the last recalled copy; `verified` 0 = the round verified none of it.
pub fn feedback(st: *Stream, accepted: u32, verified: usize) void {
    if (!st.armed) return;
    const r = record() orelse return;
    r.feedback(st, accepted, verified);
}

/// A stage round's accepted drafts: the bar a recalled copy has to beat.
pub fn observeOwn(st: *Stream, accepted: u32, block: usize) void {
    if (!st.armed) return;
    const r = record() orelse return;
    r.observeOwn(st, accepted, block);
}

/// Records a finished request's prompt tail and reply, once, and logs what recall did for it.
pub fn end(st: *Stream, prompt: []const u32, output: []const u32, finished: bool) void {
    if (!st.armed) return;
    st.armed = false;
    const r = record() orelse return;
    if (finished) r.remember(prompt, output);
    const g: d_recall.Gate = r.gate orelse d_recall.Gate.init(0);
    log.info("  [d-recall] offers={d} probes={d} drafted={d} landed={d} | ids={d}/{d} lookups={d} hits={d} short_hits={d} expect={d:.2}/{d:.2}/{d:.2}/{d:.2} own={d:.2}\n", .{
        st.offers,          st.probes,     st.drafted,      st.landed,
        r.store.held(),     r.store.cap,   r.store.lookups, r.store.hits,
        r.store.short_hits, g.expect(3),   g.expect(0),     g.expect(1),
        g.expect(2),        g.own_global,
    });
}
