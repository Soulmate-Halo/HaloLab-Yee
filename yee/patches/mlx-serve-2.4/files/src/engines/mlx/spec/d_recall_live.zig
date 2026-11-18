//! The process's one recall record (`d_recall.zig`) behind the DeepSeek-V4.1 DSpark copy rounds. Every hook sits
//! beside a forward, so every caller is the inference thread and the record takes no lock. With
//! `MLX_SERVE_D_RECALL_COLD_DIR` set, a cold layer on disk (`d_recall_cold.zig`) backs the record: finished replies go
//! to it too, each round posts the next round's lookups to its thread, and a round the record misses takes an
//! answered one, gated and verified as a recalled copy.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const d_recall = @import("d_recall.zig");
const d_recall_cold = @import("d_recall_cold.zig");
const Shared = @import("../../../sdk/util/shared.zig").Shared;

pub const Stream = d_recall.Stream;

const Live = struct {
    record: ?d_recall.Recall = null,
    failed: bool = false,
    cold: ?*d_recall_cold.Cold = null,
    cold_tried: bool = false,
    prefetch: usize = d_recall_cold.DEFAULT_PREFETCH,
    next_sid: u32 = 0,
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
    const adapt = if (env.get(.MLX_SERVE_D_RECALL_ADAPT)) |v| v[0] != '0' else true;
    const stage_rel = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_STAGE_REL, @as(f32, @floatCast(d_recall.DEFAULT_STAGE_REL))), 0, 2);
    live.record.?.setAdaptive(adapt, stage_rel, d_recall.DEFAULT_MARGIN);
    const sam_ids: usize = @intFromFloat(std.math.clamp(env.float(.MLX_SERVE_D_RECALL_SAM_IDS, @as(f32, @floatFromInt(d_recall.DEFAULT_SAM_IDS))), 0, 16777216));
    const prompt_ids: usize = @intFromFloat(std.math.clamp(env.float(.MLX_SERVE_D_RECALL_PROMPT_IDS, @as(f32, @floatFromInt(d_recall.DEFAULT_PROMPT_IDS))), 0, 65536));
    live.record.?.setSam(sam_ids, prompt_ids) catch {
        log.info("  [d-recall] suffix automaton off: {d} ids a generation did not allocate\n", .{sam_ids});
    };
    log.info("  [d-recall] record of {d} ids, index {d} slots (density {d:.4}) ({d} MiB) adapt={s} stage_rel={d:.2} sam_ids={d} prompt_ids={d}\n", .{ cap, slots, density, live.record.?.store.bytes() >> 20, if (adapt) "on" else "off", stage_rel, live.record.?.store.sam_ids, prompt_ids });
    return &live.record.?;
}

/// The cold layer, opened on first use; null while `MLX_SERVE_D_RECALL_COLD_DIR` is unset or empty, after it failed
/// to open, or once its thread hit an I/O error (it turns itself off).
fn cold() ?*d_recall_cold.Cold {
    const live = recall_shared.get();
    if (live.cold) |cl| return if (cl.isOn()) cl else null;
    if (live.cold_tried) return null;
    live.cold_tried = true;
    const dir = env.span(.MLX_SERVE_D_RECALL_COLD_DIR) orelse return null;
    if (dir.len == 0) return null;
    const slots: u64 = @intFromFloat(std.math.clamp(env.float(.MLX_SERVE_D_RECALL_COLD_SLOTS, @floatFromInt(d_recall_cold.DEFAULT_SLOTS)), 1024, 17179869184));
    const gb = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_COLD_GB, d_recall_cold.DEFAULT_GB), 0.001, 1048576);
    const max_bytes: u64 = @intFromFloat(@as(f64, gb) * 1e9);
    live.prefetch = @intFromFloat(std.math.clamp(env.float(.MLX_SERVE_D_RECALL_PREFETCH, @floatFromInt(d_recall_cold.DEFAULT_PREFETCH)), 0, d_recall_cold.MAX_PREFETCH));
    const opened = d_recall_cold.Cold.open(std.heap.page_allocator, dir, .{ .slots = slots, .max_bytes = max_bytes }) catch |e| {
        log.info("  [d-recall] cold layer off: {s} did not open ({s})\n", .{ dir, @errorName(e) });
        return null;
    };
    const cl = opened orelse return null;
    live.cold = cl;
    log.info("  [d-recall] cold layer {s}: {d} ids, index {d} slots, cap {d:.1} GB, prefetch {d}\n", .{ dir, cl.stats().ids, cl.slots, gb, live.prefetch });
    return cl;
}

/// Posts the next round's cold lookups once a round (the tree and the copy step ask at the same context).
fn post(cl: *d_recall_cold.Cold, st: *Stream, ctx: []const u32, t1: u32, drafted: []const u32) void {
    if (st.cold_posted == ctx.len + 1) return;
    st.cold_posted = ctx.len + 1;
    cl.prefetch(st.cold_sid, ctx, t1, drafted, recall_shared.get().prefetch);
}

/// Arms a DSpark request and records the prompt ids it forwards (the part no cache held).
pub fn begin(forwarded: []const u32) Stream {
    const r = record() orelse return .{};
    _ = r.store.add(forwarded);
    const live = recall_shared.get();
    live.next_sid +%= 1;
    return .{ .armed = true, .cold_sid = live.next_sid };
}

/// Recalled drafts for the context `ctx ++ [t1]` in `out` (sized to the verify and the token budget): more than
/// `block` of them, or null.
pub fn draft(st: *Stream, ctx: []const u32, t1: u32, out: []u32, block: usize) ?d_recall.Draft {
    if (!st.armed) return null;
    const r = record() orelse return null;
    st.cold_from = false;
    const cooling = st.cooldown != 0;
    const hot = r.draft(st, ctx, t1, out, block);
    const cl = cold() orelse return hot;
    if (hot) |h| {
        post(cl, st, ctx, t1, out[0..h.n]);
        return hot;
    }
    const got = if (cooling) null else coldDraft(r, cl, st, ctx, t1, out, block);
    post(cl, st, ctx, t1, if (got) |g| out[0..g.n] else &.{});
    return got;
}

/// A cold continuation as a copy round, under the gate a hot one of a 16-id match meets.
fn coldDraft(r: *d_recall.Recall, cl: *d_recall_cold.Cold, st: *Stream, ctx: []const u32, t1: u32, out: []u32, block: usize) ?d_recall.Draft {
    if (out.len < d_recall.MIN_COPY) return null;
    const avail = cl.take(st.cold_sid, ctx, t1, out);
    if (avail == 0) return null;
    const g = r.gateFor(block);
    const len = d_recall.WINDOW;
    var n = avail;
    if (g.adaptive) {
        if (avail < d_recall.MIN_COPY) return null;
        n = d_recall.bestCopyLen(g.alpha(st.sampled, d_recall.tierOf(len)), d_recall.MIN_COPY, avail);
        if (!g.admitRate(st, len, n)) return null;
    } else {
        if (n <= block or !g.admit(st, len, n)) return null;
    }
    st.drafted += n;
    st.cold_from = true;
    cl.offers += 1;
    return .{ .n = n, .len = len };
}

/// What the target kept of the last recalled copy; `verified` 0 = the round verified none of it.
pub fn feedback(st: *Stream, accepted: u32, verified: usize) void {
    if (!st.armed) return;
    const r = record() orelse return;
    if (st.cold_from) {
        st.cold_from = false;
        if (verified != 0) {
            if (recall_shared.get().cold) |cl| cl.landed += accepted;
        }
    }
    r.feedback(st, accepted, verified);
}

/// A stage round's accepted drafts of the `rows` it verified: the bar a recalled copy has to beat.
pub fn observeOwn(st: *Stream, accepted: u32, rows: usize, block: usize) void {
    if (!st.armed) return;
    const r = record() orelse return;
    r.observeOwn(st, accepted, rows, block);
}

/// A tree round's recalled candidates for `ctx ++ [t1]` (`Recall.drafts`), a cold one after the record's when it
/// adds a distinct continuation; 0 while recall is off or not armed.
pub fn drafts(st: *Stream, ctx: []const u32, t1: u32, outs: [][d_recall.MAX_DRAFTS]u32, room: usize, block: usize, cands: []d_recall.Cand) usize {
    if (!st.armed) return 0;
    const r = record() orelse return 0;
    var n = r.drafts(st, ctx, t1, outs, room, block, cands);
    const cl = cold() orelse return n;
    const rm = @min(room, d_recall.MAX_DRAFTS);
    if (st.cooldown == 0 and rm >= d_recall.MIN_COPY and n < @min(cands.len, outs.len)) {
        const avail = cl.take(st.cold_sid, ctx, t1, outs[n][0..rm]);
        if (avail >= d_recall.MIN_COPY) {
            const g = r.gateFor(block);
            const tier = d_recall.tierOf(d_recall.WINDOW);
            const a = g.alpha(st.sampled, tier);
            const len = d_recall.bestCopyLen(a, d_recall.MIN_COPY, avail);
            var dup = false;
            for (cands[0..n], 0..) |cd, j| {
                const m = @min(cd.n, len);
                if (std.mem.eql(u32, outs[j][0..m], outs[n][0..m])) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                cands[n] = .{ .n = len, .len = d_recall.WINDOW, .tier = tier, .alpha = a, .cold = true };
                n += 1;
                cl.offers += 1;
            }
        }
    }
    post(cl, st, ctx, t1, if (n > 0) outs[0][0..cands[0].n] else &.{});
    return n;
}

/// Whether candidate `c` pays as a plain copy round (no stage draft).
pub fn copyPays(st: *const Stream, c: d_recall.Cand) bool {
    const r = record() orelse return false;
    return r.copyPays(st, c);
}

/// What a tree round's recalled branch did.
pub fn learnBranch(st: *Stream, c: d_recall.Cand, landed: u32, verified: usize, won: u32) void {
    if (!st.armed) return;
    const r = record() orelse return;
    if (c.cold) {
        if (recall_shared.get().cold) |cl| cl.landed += won;
    }
    r.learnBranch(st, c, landed, verified, won);
}

/// The stage draft's cost relative to a verify row, as the gate charges it.
pub fn stageRel() f64 {
    const r = record() orelse return d_recall.DEFAULT_STAGE_REL;
    return r.stage_rel;
}

const END_FMT = "  [d-recall] offers={d} probes={d} drafted={d} landed={d} | ids={d}/{d} lookups={d} hits={d} short_hits={d} expect={d:.2}/{d:.2}/{d:.2}/{d:.2} own={d:.2} alpha_g={d:.2}/{d:.2}/{d:.2}/{d:.2} alpha_s={d:.2}/{d:.2}/{d:.2}/{d:.2} own_s={d:.2} own_rate={d:.2} sampled={d} sam_hits={d} sam_len={d:.1} sam_rot={d} sam_hot={d} sam_fail={d} tree={d} tree_rows={d} tree_br={d} tree_wins={d} tree_extra={d}";
const COLD_FMT = " cold_ids={d} cold_posts={d} cold_ready={d} cold_late={d} cold_offers={d} cold_landed={d} cold_drop={d} cold_ioerr={d}";

/// Records a finished request's prompt tail and reply, once, and logs what recall did for it.
pub fn end(st: *Stream, prompt: []const u32, output: []const u32, finished: bool) void {
    if (!st.armed) return;
    st.armed = false;
    const r = record() orelse return;
    const cl = cold();
    if (finished) {
        r.remember(prompt, output);
        if (cl) |c| c.store(prompt[prompt.len -| r.prompt_ids ..], output);
    }
    const g: d_recall.Gate = r.gate orelse d_recall.Gate.init(0);
    const sam_len: f64 = if (r.store.sam_hits == 0) @as(f64, 0) else @as(f64, @floatFromInt(r.store.sam_len_sum)) / @as(f64, @floatFromInt(r.store.sam_hits));
    const args = .{
        st.offers,                st.probes,         st.drafted,        st.landed,
        r.store.held(),           r.store.cap,       r.store.lookups,   r.store.hits,
        r.store.short_hits,       g.expect(3),       g.expect(0),       g.expect(1),
        g.expect(2),              g.own_global,      g.alpha(false, 0), g.alpha(false, 1),
        g.alpha(false, 2),        g.alpha(false, 3), g.alpha(true, 0),  g.alpha(true, 1),
        g.alpha(true, 2),         g.alpha(true, 3),  g.own_global_s,    g.ownRate(st),
        @intFromBool(st.sampled), r.store.sam_hits,  sam_len,           r.store.sam_rot,
        r.store.sam_hot_ids,      r.store.sam_fail,
        st.tree_rounds,           st.tree_rows,      st.tree_branches,  st.tree_wins,
        st.tree_extra,
    };
    // The cold layer as it was opened, off or not: its counters say what it did before it stopped.
    if (recall_shared.get().cold) |c| {
        const s = c.stats();
        log.info(END_FMT ++ COLD_FMT ++ "\n", args ++ .{ s.ids, s.posts, s.ready, s.late, s.offers, s.landed, s.drop, s.ioerr });
    } else {
        log.info(END_FMT ++ "\n", args);
    }
}
