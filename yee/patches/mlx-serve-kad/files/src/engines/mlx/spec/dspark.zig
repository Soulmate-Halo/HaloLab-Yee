//! DSpark rounds: greedy copy and stochastic arms, the Markov-head policy.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const dsv4_mod = @import("../../../plugins/deepseek_v4/deepseek_v4.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const mlx_stream = @import("../modules/stream.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const dflash_policy = @import("dflash/policy.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const testing = std.testing;
const Constraint = @import("../grammar/constraint.zig").Constraint;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Generator = @import("../generate/generator.zig").Generator;
const modules = @import("../modules/call.zig");
const plugin_settle = @import("../plugin/settle.zig");
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dsparkCopyLen = @import("verify.zig").dsparkCopyLen;
const d_recall = @import("d_recall.zig");
const d_recall_live = @import("d_recall_live.zig");
const kad_live = @import("kad_live.zig");
const mtpBatchedAcceptGraph = @import("mtp/batched_graph.zig").mtpBatchedAcceptGraph;
const mtpSerialOneToken = @import("mtp/serial.zig").mtpSerialOneToken;
const probsAllPositions = @import("../sampling/spec_sampling.zig").probsAllPositions;
const seedKey = @import("../sampling/lazy.zig").seedKey;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const intree = @import("../plugin/intree.zig");
const abi = @import("../../../abi/plugin.zig");
const plugin_driver = @import("../plugin/driver.zig");
const log = @import("../../../sdk/util/log.zig");

fn treeTrace() bool {
    return if (env.get(.MLX_SERVE_D_RECALL_TREE_TRACE)) |v| v[0] == '1' else false;
}


/// Which DSpark accept rule (if any) the dsv4 chokepoint may arm for a
/// request. Pure over its inputs so every arm is unit-testable.
pub const DsparkArm = enum { off, greedy, stochastic };

/// `clean` = nothing consumes logits beyond plain sampling: penalties,
/// grammar and logprobs stay serial on BOTH arms (matching the greedy-only
/// contract this generalizes). A clean greedy request (temp < 0.01 or
/// top_k == 1) gets the raw argmax-equality accept; a clean SAMPLED
/// request gets the stochastic arm (MTP one-hot Leviathan acceptance over
/// filtered target probs).
pub fn dsparkArmFor(sampling: SamplingParams, logprobs_n: u32) DsparkArm {
    const clean = !sampling.penalized() and
        sampling.constraint == null and
        logprobs_n == 0;
    if (!clean) return .off;
    const greedy = sampling.temperature < 0.01 or sampling.top_k == 1;
    if (greedy) return .greedy;
    return .stochastic;
}

/// Shared by dispatch and /props; this disables drafting, not resident stages.
pub fn dsparkEnabled() bool {
    return if (env.get(.MLX_SERVE_DSV4_DSPARK)) |v| v[0] != '0' else true;
}

/// PLD draft+verify decode step. The draft comes from an n-gram lookup
/// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
/// what makes PLD model-agnostic and cheap.
///
/// `key_len` is the n-gram size used for matching (default 3). `draft_len`
/// is the maximum number of speculative tokens to verify per step (default
/// 5). Both are clamped to safe upper bounds internally.
///
/// Returns `null` only when generation is already done. When no n-gram
/// match exists (cold start, novel output), falls back to the regular
/// `next()` path and returns a single-token result with `used_lookup=false`.
/// DeepSeek-V4 DSpark step (the arch's OWN block-parallel spec decode).
/// Entry/exit share the v2 spec invariant: module state = prompt +
/// emitted positions, t1 = `next_token_id` NOT in state, pending empty
/// (init's spec branch establishes it; every exit restores it). The
/// heavy lifting — draft, batched verify, snapshot rollback — lives in
/// `deepseek_v4.dsparkRound`; this wrapper only keeps the Generator's
/// bookkeeping (generated_ids, step accounting, the shell cache.step
/// that forwardDsv4WithImpl keys fresh-vs-decode on) in sync.
pub fn nextDspark(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (self.done) return null;
    if (!self.dspark_enabled) {
        // Same defensive fallback as nextPld's disarmed arm: the
        // dispatching caller's flag alone must never run a draft.
        const tok_opt = try self.next(allocator);
        if (tok_opt == null) return null;
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok_opt.?;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }
    if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;
    // A skipped round is a tick of the serial pipeline; the first drafted round after it drains the pipeline.
    if (self.dspark_pol) |*pol| {
        if (!pol.drafting.shouldDraft() and self.dflashCanPipeline()) return try self.dsparkPlainPipelined(allocator);
        pol.plain_watch = null;
        if (self.has_pending_logits or self.has_pending_token) switch (try self.drainPipelineForSpec(allocator)) {
            .drained => |tok| return try mtpSerialOneToken(allocator, tok),
            .stopped => return null,
            .already_clean, .stay_disabled => return error.PipelineHalfState,
        };
    }
    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting
    if (self.xfm.mlx_stream_ext) |m| {
        const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
        const r = try mlx_stream.round(m, allocator, self.next_token_id, cap);
        return self.commitDsparkRound(allocator, .{ .tokens = r.tokens, .accepted = r.accepted, .next_token = r.next_token }, mlx_stream.position(m));
    }
    if (plugin_driver.nativeDrafter(self.xfm) != null) return nativeRound(self, allocator);
    return modules.call("dspark_round", .{ self, allocator });
}

/// A sampled round of the served plugin's drafter: the plugin drafts and verifies, the engine accepts under the
/// request's sampling, the plugin keeps what was accepted.
fn nativeSampledRound(self: *Generator, allocator: std.mem.Allocator, cap: u32) !?DrafterStepResult {
    var v = try plugin_driver.nativeVerify(self.xfm, &self.ctx, self.next_token_id, cap);
    defer _ = mlx.mlx_array_free(v.logits);
    const ids = v.round.tokens[0..v.round.n];
    // A failure below leaves the plugin's verify open, which is the model's, not this request's: close it.
    var closed = false;
    errdefer if (!closed) {
        _ = plugin_driver.nativeFinish(self.xfm, &self.ctx, 0) catch 0;
    };
    const verdict = try stochasticVerdict(self, v.logits, ids, self.xfm.config.vocab_size, cap);
    closed = true;
    const position = try plugin_driver.nativeFinish(self.xfm, &self.ctx, verdict.accepted);
    const tokens = try allocator.dupe(u32, ids[0 .. 1 + verdict.accepted]);
    errdefer allocator.free(tokens);
    self.dspark_stats.record(v.round.rows -| 1, verdict.accepted);
    return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = verdict.accepted, .next_token = verdict.next_token }, @intCast(position));
}

/// One round of the served plugin's own drafter (`NativeDrafter.step`): its tokens and the trunk token after them.
fn nativeRound(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    try plugin_settle.activate(self.xfm, &self.ctx);
    // A forced commit (`commitForcedTokens`) leaves its last token in the model's state, whatever `dspark_pol` is:
    // publish it and sample `t1` from its logits, so the drafter never takes it a second time.
    if (self.has_pending_logits or self.has_pending_token) switch (try self.drainPipelineForSpec(allocator)) {
        .drained => |tok| return try mtpSerialOneToken(allocator, tok),
        .stopped => return null,
        .already_clean, .stay_disabled => return error.PipelineHalfState,
    };
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
    if (self.dspark_stochastic) return nativeSampledRound(self, allocator, @min(cap, abi.max_round_tokens - 1));
    const r = try plugin_driver.nativeRound(self.xfm, &self.ctx, self.next_token_id, @min(cap, abi.max_round_tokens - 1));
    const tokens = try allocator.dupe(u32, r.tokens[0..r.n]);
    errdefer allocator.free(tokens);
    self.dspark_stats.record(r.rows -| 1, r.accepted);
    return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = r.accepted, .next_token = r.next_token }, @intCast(r.position));
}

/// One DSpark round on the DeepSeek-V4/V4.1 module itself (`modules.call`).
pub fn moduleRound(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (intree.dsv41(self.xfm)) |mdl| {
        try plugin_settle.activate(self.xfm, &self.ctx);
        if (try dsparkTreeStep(self, allocator, mdl)) |res| return res;
        if (try self.dsparkCopyStep(allocator, mdl)) |res| return res;
        return self.dsparkPolicyStep(allocator, mdl);
    }
    return self.dsparkStep(allocator, dsv4_mod, intree.dsv4(self.xfm).?);
}

/// A DSpark round whose drafts are the context's own continuation after `t1` (an echo, an edit), or, for a request
/// (greedy or sampled) whose context holds none, an earlier request's (`d_recall`): no stage forward, and up to
/// `VERIFY_MAX_DRAFTS` drafts where the block holds `ds_block`. Null when neither holds a copy worth verifying.
pub fn dsparkCopyStep(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model) !?DrafterStepResult {
    const t1 = self.next_token_id;
    const idx = try self.mtpLookupIndex(allocator);
    const max_drafts = dsv41_mod.maxVerifyDrafts(mdl);
    const remaining = self.max_tokens -| self.completion_tokens -| 1;
    var found = idx.matchBest(t1, @intCast(max_drafts));
    var n = dsparkCopyLen(found, remaining, mdl.ds_block);
    var recalled: [d_recall.MAX_DRAFTS]u32 = undefined;
    var recall = false;
    // Sampled requests take recalled drafts too: a recalled draft is a point-mass proposal (q one-hot), which is
    // exactly what the stochastic verify (`mtpBatchedAcceptGraph(.., null, ..)`) assumes, so the output stays lossless.
    if (n == 0) {
        self.recall.sampled = self.dspark_stochastic;
        const room: usize = @min(max_drafts, @as(usize, remaining), recalled.len);
        if (d_recall_live.draft(&self.recall, idx.toks.items, t1, recalled[0..room], mdl.ds_block)) |hit| {
            found = .{ .draft = recalled[0..hit.n], .suffix = hit.len };
            n = hit.n;
            recall = true;
        }
    }
    if (n == 0) return null;
    const ids = try allocator.alloc(u32, n + 1);
    ids[0] = t1;
    @memcpy(ids[1..], found.?.draft[0..n]);
    var copy: dsv41_mod.DsparkDraft = .{ .ids = ids, .len = n, .logits = &.{}, .confidence = &.{} };
    defer copy.deinit(allocator);
    const st = &mdl.dec_state.?;
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, dsv41_mod, mdl, t1, cap, &copy)
    else
        try dsv41_mod.dsparkRoundWith(mdl, allocator, st, t1, &copy, cap);
    kad_live.observe(&self.recall, n, round.accepted, recall);
    if (recall) {
        d_recall_live.feedback(&self.recall, round.accepted, n);
    } else {
        self.dspark_copies += 1;
        self.dspark_copy_drafted += n;
        self.dspark_copy_accepted += round.accepted;
    }
    return self.commitDsparkRound(allocator, round, st.n);
}

fn treeEnabled() bool {
    return if (env.get(.MLX_SERVE_D_RECALL_TREE)) |v| v[0] != '0' else true;
}

const TreeEnv = struct { k: usize, state: f64, max_bytes: u64, margin: f64 };

fn treeEnv() TreeEnv {
    const k = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_TREE_K, @as(f32, d_recall.DEFAULT_TREE_K)), 0, 3);
    const state = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_TREE_STATE_COST, @as(f32, @floatCast(d_recall.DEFAULT_TREE_STATE_COST))), 0, 4);
    const mib = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_TREE_MAX_MIB, 512), 0, 1 << 20);
    const margin = std.math.clamp(env.float(.MLX_SERVE_D_RECALL_TREE_MARGIN, @as(f32, @floatCast(d_recall.DEFAULT_TREE_MARGIN))), 0, 1);
    return .{ .k = @intFromFloat(k), .state = state, .max_bytes = @as(u64, @intFromFloat(mib)) << 20, .margin = margin };
}

/// A DSpark round that verifies the stage draft and recalled copies as the branches of one tree: when no recalled copy
/// pays as a plain copy round (`dsparkCopyStep` keeps those), the stages draft, `d_recall.choosePlan` prices every set
/// of branches and cuts within the batched verify's rows, and a plan with a recalled branch that beats the stage draft
/// alone (by `MLX_SERVE_D_RECALL_TREE_MARGIN`) runs. Each branch past the first is a clone of the state
/// (`dsv41.cloneState`, rows shared until written); one `extendBatched` verifies every branch's `[t1, drafts…]`, the
/// walk (`d_recall.walkTree`) keeps the target's own picks along the live branches, the winning state keeps its prefix
/// and becomes the state. Greedy picks are each row's argmax, sampled ones an independent draw a row from the request's
/// filtered density, so the round is lossless. Null when the round is not a tree round's (the copy and stage rounds
/// then run as before); a plan of the stage draft alone is the stage round itself.
pub fn dsparkTreeStep(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model) !?DrafterStepResult {
    @setRuntimeSafety(true);
    if (!self.recall.armed or !treeEnabled()) return null;
    if (mdl.pair != null or mdl.streamed != null or mdl.dspark == null) return null;
    if (self.dspark_pol == null) return null;
    const te = treeEnv();
    if (te.k == 0) return null;
    const st = &mdl.dec_state.?;
    if (st.verify != null) return null;
    const t1 = self.next_token_id;
    const idx = try self.mtpLookupIndex(allocator);
    const max_drafts = dsv41_mod.maxVerifyDrafts(mdl);
    const remaining = self.max_tokens -| self.completion_tokens -| 1;
    // The context's own continuation goes first, as the copy round takes it.
    if (dsparkCopyLen(idx.matchBest(t1, @intCast(max_drafts)), remaining, mdl.ds_block) > 0) return null;
    const room: usize = @min(max_drafts, @as(usize, remaining), d_recall.MAX_DRAFTS);
    self.recall.sampled = self.dspark_stochastic;
    var bufs: [d_recall.MAX_CANDS][d_recall.MAX_DRAFTS]u32 = undefined;
    var cands: [d_recall.MAX_CANDS]d_recall.Cand = undefined;
    const nc = d_recall_live.drafts(&self.recall, idx.toks.items, t1, &bufs, room, mdl.ds_block, &cands);
    if (nc == 0) return null;
    for (cands[0..nc]) |c| if (d_recall_live.copyPays(&self.recall, c)) return null;
    if (dsv41_mod.stateBytes(st) > te.max_bytes) return null;

    const watch = io_util.Stopwatch.init(self.timer.io);
    const pol = &self.dspark_pol.?;
    var d = try dsv41_mod.dsparkDraft(mdl, allocator, st, t1);
    defer d.deinit(allocator);
    const own_cut = dsparkVerifyDrafts(d.confidence[0..d.len], &pol.calib, pol.plain.ms);
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);

    // Branch 0 the stage draft (its calibrated confidences), then the recalled ones (their tier's acceptance).
    var probs: [d_recall.MAX_BRANCHES][d_recall.MAX_DRAFTS]f64 = undefined;
    var branches: [d_recall.MAX_BRANCHES]d_recall.Branch = undefined;
    const own_n: usize = @min(d.len, d_recall.MAX_DRAFTS);
    for (0..own_n) |j| probs[0][j] = pol.calib.probability(sigmoid(d.confidence[j]));
    branches[0] = .{ .toks = d.ids[1 .. 1 + own_n], .p = probs[0][0..own_n] };
    // Typed: a bare @min against a comptime bound narrows (to u2 here), and 1 + nr overflows at 3.
    const nr: usize = @min(nc, d_recall.MAX_BRANCHES - 1);
    for (cands[0..nr], 0..) |c, i| {
        @memset(probs[i + 1][0..c.n], c.alpha);
        branches[i + 1] = .{ .toks = bufs[i][0..c.n], .p = probs[i + 1][0..c.n] };
    }
    const nb = 1 + nr;
    const ch = d_recall.choosePlan(branches[0..nb], te.k, .{
        .state = te.state,
        .stage = d_recall_live.stageRel(),
        .rows = @min(d_recall.TREE_ROWS, dsv41_mod.region_rows),
        .cap = cap,
    });
    const plan = ch.mixed;
    var take = plan.branches > 0 and plan.rate > ch.own.rate * (1 + te.margin);
    if (!take and plan.branches > 0) {
        // A recalled tier left out keeps learning: one round in TREE_PROBE takes its best plan anyway.
        self.recall.tree_idle += 1;
        take = self.recall.tree_idle >= d_recall.TREE_PROBE;
    }
    if (!take) {
        d.len = own_cut;
        return dsparkPolicyWith(self, allocator, mdl, &d, watch);
    }
    self.recall.tree_idle = 0;

    var order: [d_recall.MAX_BRANCHES]usize = undefined;
    var nu: usize = 0;
    for (0..nb) |b| if (plan.lens[b] > 0) {
        order[nu] = b;
        nu += 1;
    };
    if (nu == 1) {
        // One recalled branch: the copy round's chain, verified on the state itself.
        const b = order[0];
        const n = plan.lens[b];
        const ids = try allocator.alloc(u32, n + 1);
        ids[0] = t1;
        @memcpy(ids[1..], branches[b].toks[0..n]);
        var copy: dsv41_mod.DsparkDraft = .{ .ids = ids, .len = n, .logits = &.{}, .confidence = &.{} };
        defer copy.deinit(allocator);
        const round = if (self.dspark_stochastic)
            try self.dsparkStochasticRound(allocator, dsv41_mod, mdl, t1, cap, &copy)
        else
            try dsv41_mod.dsparkRoundWith(mdl, allocator, st, t1, &copy, cap);
        d_recall_live.learnBranch(&self.recall, cands[b - 1], round.accepted, @min(n, @as(usize, round.accepted) + 1), round.accepted);
        self.recall.tree_rounds += 1;
        self.recall.tree_rows += n + 1;
        self.recall.tree_branches += 1;
        if (round.accepted > 0) self.recall.tree_wins += 1;
        self.recall.tree_extra += round.accepted;
        return self.commitDsparkRound(allocator, round, st.n);
    }

    // Branch i of the plan on states[i]: the state itself first, a clone for each other.
    const trace = treeTrace();
    if (trace) log.info("  [d-tree] plan n={d} branches={d} rows={d} lens={d}/{d}/{d}/{d} sampled={d} cap={d} nc={d}\n", .{ st.n, nu, plan.rows, plan.lens[0], plan.lens[1], plan.lens[2], plan.lens[3], @intFromBool(self.dspark_stochastic), cap, nc });
    var clones: [d_recall.MAX_BRANCHES]dsv41_mod.State = undefined;
    var n_clones: usize = 0;
    defer for (clones[0..n_clones]) |*c| c.deinit();
    var states: [d_recall.MAX_BRANCHES]*dsv41_mod.State = undefined;
    var gens: [d_recall.MAX_BRANCHES]u64 = undefined;
    var chunk_bufs: [d_recall.MAX_BRANCHES][d_recall.MAX_DRAFTS + 1]u32 = undefined;
    var chunks: [d_recall.MAX_BRANCHES][]const u32 = undefined;
    var toks: [d_recall.MAX_BRANCHES][]const u32 = undefined;
    var rows: usize = 0;
    for (order[0..nu], 0..) |b, i| {
        if (i == 0) {
            states[0] = st;
            gens[0] = mdl.state_gen;
        } else {
            clones[n_clones] = try dsv41_mod.cloneState(st);
            n_clones += 1;
            states[i] = &clones[n_clones - 1];
            gens[i] = 0;
        }
        const n = plan.lens[b];
        chunk_bufs[i][0] = t1;
        @memcpy(chunk_bufs[i][1 .. n + 1], branches[b].toks[0..n]);
        chunks[i] = chunk_bufs[i][0 .. n + 1];
        toks[i] = chunk_bufs[i][1 .. n + 1];
        rows += n + 1;
    }
    if (trace) log.info("  [d-tree] clones={d} rows={d}\n", .{ n_clones, rows });
    const logits = try dsv41_mod.extendBatched(mdl, allocator, states[0..nu], gens[0..nu], chunks[0..nu], true);
    defer _ = mlx.mlx_array_free(logits);
    if (trace) log.info("  [d-tree] extended n={d}\n", .{st.n});
    // Every branch's verify is open until the winner keeps its prefix; on a failure the state's is dropped (its clones'
    // go with them) and the request fails, as a chain round's failure leaves it.
    var open = true;
    errdefer if (open) dsv41_mod.abortVerify(st);
    var picks: [d_recall.TREE_ROWS]u32 = undefined;
    if (self.dspark_stochastic) try sampleEveryRow(self, logits, rows, mdl.vocab, picks[0..rows]) else try argmaxEveryRow(logits, mdl.s, picks[0..rows]);
    var ys: [d_recall.MAX_BRANCHES][]const u32 = undefined;
    var off: usize = 0;
    for (chunks[0..nu], 0..) |c, i| {
        ys[i] = picks[off .. off + c.len];
        off += c.len;
    }
    const w = d_recall.walkTree(toks[0..nu], ys[0..nu], cap);
    if (trace) log.info("  [d-tree] walk accepted={d} winner={d} next={d}\n", .{ w.accepted, w.winner, w.next });
    try dsv41_mod.finishVerify(mdl, allocator, states[w.winner], w.accepted + 1);
    if (w.winner != 0) dsv41_mod.adoptBranch(st, states[w.winner]);
    open = false;
    if (trace) log.info("  [d-tree] kept n={d}\n", .{st.n});
    const tokens = try allocator.dupe(u32, chunks[w.winner][0 .. w.accepted + 1]);
    errdefer allocator.free(tokens);

    // The stage draft learns what it did on the path, each recalled branch what it did; the recalled winner is credited
    // the tokens it kept past the stage draft's own.
    var own_landed: u32 = 0;
    var own_in = false;
    for (order[0..nu], 0..) |b, i| if (b == 0) {
        own_in = true;
        own_landed = w.landed[i];
        for (d.confidence[0..plan.lens[0]], 0..) |c, k| {
            if (k >= w.verified[i]) break;
            pol.calib.observe(sigmoid(c), k < w.landed[i]);
        }
        self.dspark_stats.record(@intCast(plan.lens[0]), w.landed[i]);
        d_recall_live.observeOwn(&self.recall, w.landed[i], plan.lens[0], mdl.ds_block);
    };
    const extra: u32 = if (order[w.winner] != 0 and w.accepted > own_landed) @intCast(w.accepted - own_landed) else 0;
    for (order[0..nu], 0..) |b, i| if (b > 0) {
        d_recall_live.learnBranch(&self.recall, cands[b - 1], w.landed[i], w.verified[i], if (i == w.winner) extra else 0);
    };
    self.recall.tree_rounds += 1;
    self.recall.tree_rows += rows;
    self.recall.tree_branches += nu;
    if (extra > 0) self.recall.tree_wins += 1;
    self.recall.tree_extra += extra;
    if (own_in) {
        const wall_ms = @as(f32, @floatFromInt(watch.read())) / @as(f32, std.time.ns_per_ms);
        pol.drafting.record(@as(f32, @floatFromInt(tokens.len)) * pol.plain.ms, wall_ms);
    }
    return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = @intCast(w.accepted), .next_token = w.next }, st.n);
}

/// Every row's argmax of `logits` `[R, V]` into `out` (R of them).
fn argmaxEveryRow(logits: mlx.mlx_array, s: mlx.mlx_stream, out: []u32) !void {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const data = mlx.mlx_array_data_int32(am32) orelse return error.MlxArrayDataNull;
    if (mlx.mlx_array_size(am32) != out.len) return error.TreeRowsMismatch;
    for (out, 0..) |*o, i| o.* = @intCast(data[i]);
}

/// One independent draw a row of `logits` `[rows, vocab]` from the request's filtered density (one sampling draw of
/// the request, `nextDraw`), into `out`.
fn sampleEveryRow(self: *Generator, logits: mlx.mlx_array, rows: usize, vocab: usize, out: []u32) !void {
    const s = self.xfm.s;
    var vl3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vl3);
    try mlx.check(mlx.mlx_reshape(&vl3, logits, &[_]c_int{ 1, @intCast(rows), @intCast(vocab) }, 3, s));
    const sp = self.nextDraw();
    const probs = try probsAllPositions(vl3, sp, s);
    defer _ = mlx.mlx_array_free(probs);
    var log_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_p);
    try mlx.check(mlx.mlx_log(&log_p, probs, s));
    const key = seedKey(sp);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_p, -1, key, s));
    var si = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(si);
    try mlx.check(mlx.mlx_astype(&si, sampled, .int32, s));
    try mlx.check(mlx.mlx_array_eval(si));
    const data = mlx.mlx_array_data_int32(si) orelse return error.MlxArrayDataNull;
    if (mlx.mlx_array_size(si) != out.len) return error.TreeRowsMismatch;
    for (out, 0..) |*o, i| o.* = @intCast(data[i]);
}

/// One DSpark round on module `mod` (deepseek_v4 or deepseek_v41, which
/// share the round API) and its bookkeeping on the Generator.
pub fn dsparkStep(self: *Generator, allocator: std.mem.Allocator, comptime mod: type, mdl: anytype) !?DrafterStepResult {
    const t1 = self.next_token_id;
    const accepted_cap = capAcceptedForTokenBudget(
        std.math.maxInt(u32),
        self.completion_tokens,
        self.max_tokens,
    );
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, mod, mdl, t1, accepted_cap, null)
    else
        try mod.dsparkRound(mdl, allocator, &mdl.dec_state.?, t1, accepted_cap);
    return self.commitDsparkRound(allocator, round, mdl.dec_state.?.n);
}

/// A round the drafting backoff skipped: one tick of the serial pipeline. Consecutive ticks time the plain step.
pub fn dsparkPlainPipelined(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    const tok = (try self.next(allocator)) orelse return null;
    self.dspark_plain += 1;
    const pol = &self.dspark_pol.?;
    if (pol.plain_watch) |*w| pol.plain.sample(@as(f32, @floatFromInt(w.read())) / @as(f32, std.time.ns_per_ms));
    pol.plain_watch = io_util.Stopwatch.init(self.timer.io);
    return try mtpSerialOneToken(allocator, tok);
}

/// One DeepSeek-V4.1 stage round under `dspark_pol`: draft, verify the prefix the calibrated confidence pays
/// for (none = the trunk token alone), and teach the calibration and the drafting backoff what it bought.
pub fn dsparkPolicyStep(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model) !?DrafterStepResult {
    const watch = io_util.Stopwatch.init(self.timer.io);
    const pol = &self.dspark_pol.?;
    const st = &mdl.dec_state.?;
    const t1 = self.next_token_id;
    var d = try dsv41_mod.dsparkDraft(mdl, allocator, st, t1);
    defer d.deinit(allocator);
    d.len = dsparkVerifyDrafts(d.confidence[0..d.len], &pol.calib, pol.plain.ms);
    return dsparkPolicyWith(self, allocator, mdl, &d, watch);
}

/// `dsparkPolicyStep` past its draft: `d` cut to the drafts it verifies, `watch` started before the draft.
fn dsparkPolicyWith(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model, d: *dsv41_mod.DsparkDraft, watch: io_util.Stopwatch) !?DrafterStepResult {
    d.len = kad_live.capStage(self.timer.io, mdl.ds_block, &self.recall, d.len); // Yee KAD (off: unchanged)
    const pol = &self.dspark_pol.?;
    const st = &mdl.dec_state.?;
    const t1 = self.next_token_id;
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, dsv41_mod, mdl, t1, cap, d)
    else
        try dsv41_mod.dsparkRoundWith(mdl, allocator, st, t1, d, cap);
    for (d.confidence[0..d.len], 0..) |c, k| {
        if (k > round.accepted) break;
        pol.calib.observe(sigmoid(c), k < round.accepted);
    }
    self.dspark_stats.record(@intCast(d.len), round.accepted);
    kad_live.observe(&self.recall, d.len, round.accepted, false);
    self.recall.sampled = self.dspark_stochastic;
    d_recall_live.observeOwn(&self.recall, round.accepted, d.len, mdl.ds_block);
    const wall_ms = @as(f32, @floatFromInt(watch.read())) / @as(f32, std.time.ns_per_ms);
    pol.drafting.record(@as(f32, @floatFromInt(round.tokens.len)) * pol.plain.ms, wall_ms);
    return self.commitDsparkRound(allocator, round, st.n);
}

pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// Drafts a DSpark round verifies: the prefix whose calibrated acceptance (the confidence head's logits through
/// `calib`) buys the most tokens per ms of verify, on the verify ladder scaled to this request's plain step.
pub fn dsparkVerifyDrafts(conf_logits: []const f32, calib: *const dflash_policy.Calibration, plain_ms: f32) usize {
    var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
    const n = @min(conf_logits.len, p.len);
    for (conf_logits[0..n], p[0..n]) |c, *o| o.* = calib.probability(sigmoid(c));
    const costs = dflash_policy.priorCosts(plain_ms);
    return dflash_policy.chooseRows(p[0..n], &costs).rows - 1;
}

/// A draft round's bookkeeping; the module stands at `position`.
pub fn commitDsparkRound(self: *Generator, allocator: std.mem.Allocator, round_in: dsv4_mod.DsparkRound, position: usize) !?DrafterStepResult {
    var round = round_in;
    errdefer round.deinit(allocator);
    // The round advanced the module state — mirror it on the shell
    // cache verbatim so a later serial fallback (or the fresh-request
    // check keying on step==0) sees a consistent position. Generator.step
    // itself moves through advanceStep below (the clear-cadence clock).
    self.ctx.cache.step = position;
    self.dspark_attempted += 1;
    self.dspark_accepted_tokens += round.accepted;
    try self.generated_ids.appendSlice(allocator, round.tokens);
    self.advanceStep(@intCast(round.tokens.len));
    self.next_token_id = round.next_token;
    // tokens ownership transfers to the caller (scheduler frees).
    return DrafterStepResult{ .tokens = round.tokens, .accepted_tokens = round.accepted };
}

/// This request's sampling for one more draw: a seeded request keys each draw apart (`seedKey`).
pub fn nextDraw(self: *Generator) SamplingParams {
    const sp = self.sampling;
    self.sampling.draw +%= 1;
    return sp;
}

/// One stochastic DSpark round: dsv4's own greedy stage draft (a one-hot
/// proposal) verified with the MTP acceptance machinery — filtered target
/// probs at EVERY verify position (`probsAllPositions`, the request's own
/// temperature/top-k/top-p), accept draft k with prob `min(1, p_k)`,
/// first reject at `a` corrected from `normalize(max(p_a − onehot, 0))`,
/// full accept sampled from the bonus row — corrections pre-sampled in
/// ONE batched graph (`mtpBatchedAcceptGraph`, one-hot arm) so the round
/// pays ONE bounded sync. The output distribution equals serial sampling
/// (the toy-vocab exactness test's invariant), and the correction always
/// derives from the ORIGINAL verify logits at the acceptance point — the
/// house partial-accept invariant in sampled form.
/// `copy`, when set, is the draft instead of the stages' (a one-hot proposal too).
pub fn dsparkStochasticRound(self: *Generator, allocator: std.mem.Allocator, comptime mod: type, mdl: anytype, t1: u32, accepted_cap: u32, copy: ?*const mod.DsparkDraft) !mod.DsparkRound {
    var pending = if (copy) |d| try mod.dsparkBeginWith(mdl, allocator, &mdl.dec_state.?, t1, d) else try mod.dsparkBegin(mdl, allocator, &mdl.dec_state.?, t1);
    defer pending.deinit();
    const verdict = try stochasticVerdict(self, pending.vl_g, pending.verify[0 .. pending.b + 1], mdl.vocab, accepted_cap);
    pending.lapVerify(mdl);

    const round = try mod.dsparkFinish(mdl, allocator, &mdl.dec_state.?, &pending, verdict.accepted, verdict.next_token);
    mod.dsparkObserve(mdl, round.phases);
    return round;
}

/// The sampled accept of a verify chunk: `verify` is `[t1, drafts...]` and `vl_g` its logits `[rows, vocab]`. Keeps each
/// draft (a point mass) with the probability the request's filtered density gives it; the token pending after the kept
/// prefix is the residual sample at the first reject, or the bonus draw from the last row on a full accept.
pub fn stochasticVerdict(self: *Generator, vl_g: mlx.mlx_array, verify: []const u32, vocab: usize, accepted_cap: u32) !struct { accepted: u32, next_token: u32 } {
    const s = self.xfm.s;
    const b: u32 = @intCast(verify.len - 1);

    // Filtered target probs over every verify row: [b+1, V] → [1, b+1, V].
    const vshape = [_]c_int{ 1, @intCast(verify.len), @intCast(vocab) };
    var vl3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vl3);
    try mlx.check(mlx.mlx_reshape(&vl3, vl_g, &vshape, 3, s));
    const sp = self.nextDraw();
    const probs_all = try probsAllPositions(vl3, sp, s);
    defer _ = mlx.mlx_array_free(probs_all);

    var accepted: u32 = 0;
    var next_token: u32 = undefined;
    if (b == 0) {
        // The confidence gate submitted nothing: this round verifies t1
        // alone and row 0 IS the bonus row — sample the next trunk token
        // from it directly.
        var log_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_p);
        try mlx.check(mlx.mlx_log(&log_p, probs_all, s));
        const key = seedKey(sp);
        defer _ = mlx.mlx_array_free(key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, log_p, -1, key, s));
        var samp_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(samp_i);
        try mlx.check(mlx.mlx_astype(&samp_i, sampled, .int32, s));
        try mlx.check(mlx.mlx_array_eval(samp_i));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, samp_i));
        next_token = @intCast(v);
    } else {
        // [1] int32 arrays of the draft ids (already realized host values
        // — dsv4 drafts synchronously, unlike the MTP head's lazy chain).
        var draft_arrs: [16]mlx.mlx_array = undefined;
        var n_arrs: usize = 0;
        defer for (draft_arrs[0..n_arrs]) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        const idshape = [_]c_int{1};
        for (0..b) |k| {
            const idv = [_]i32{@intCast(verify[k + 1])};
            draft_arrs[k] = mlx.mlx_array_new_data(&idv, &idshape, 1, .int32);
            n_arrs += 1;
        }
        var bg = try mtpBatchedAcceptGraph(probs_all, draft_arrs[0..b], null, b, sp, s);
        defer bg.deinit();

        // ONE bounded sync: the accept vector + pre-sampled corrections
        // (and the whole verify graph beneath them) in one batched eval.
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, bg.accept_p);
            _ = mlx.mlx_vector_array_append_value(ev, bg.corr_samples);
            try mlx.check(mlx.mlx_async_eval(ev));
        }
        try mlx.check(mlx.mlx_array_eval(bg.accept_p));
        const p_data = mlx.mlx_array_data_float32(bg.accept_p) orelse return error.MlxArrayDataNull;
        while (accepted < b) {
            const accept_prob: f32 = @min(1.0, p_data[accepted]);
            const u: f32 = self.prng.random().float(f32);
            if (u >= accept_prob) break;
            accepted += 1;
        }
        // Pick the correction at the request-budget boundary and let
        // dsparkFinish roll module-owned state back to that same point.
        accepted = @min(accepted, accepted_cap);
        try mlx.check(mlx.mlx_array_eval(bg.corr_samples));
        const corr = mlx.mlx_array_data_int32(bg.corr_samples) orelse return error.MlxArrayDataNull;
        next_token = @intCast(corr[accepted]);
    }
    return .{ .accepted = accepted, .next_token = next_token };
}

// ── tests ──


test "native speculation: shared DSpark environment policy" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    try testing.expect(Generator.dsparkEnabled());
    for ([_][:0]const u8{ "", "1", "0", "0disabled" }) |value| {
        env.set(.MLX_SERVE_DSV4_DSPARK, value);
        try testing.expectEqual(value.len == 0 or value[0] != '0', Generator.dsparkEnabled());
    }
}

/// The DSV4_MINI pack at `path` served as the scheduler loads it: its plugin behind the driver's shell, `cfg` the
/// engine's parse (freed after the shell).
fn servedMini(io: std.Io, a: std.mem.Allocator, path: []const u8, cfg: *model_mod.ModelConfig) !Transformer {
    cfg.* = try model_mod.parseConfig(io, a, path);
    errdefer cfg.deinit(a);
    return plugin_driver.load(io, a, cfg, path, mlx.gpuStream());
}

test "dsv4: nextPld on a chokepoint-disabled generator stays serial (DSV4_MINI)" {
    // DSpark is opt-in at load; the nextDspark arm below needs it armed.
    env.set(.MLX_SERVE_DSV4_DSPARK, "1");
    // The live corruption path (log 166348-166361): the scheduler
    // decode tick dispatched on `slot.enable_pld` alone, so it called
    // `nextPld` on a generator whose init the dsv4 guard had already flipped
    // to pld_enabled=false — and nextPld trusted its caller, ran lookup +
    // verify forwards, and the rejected drafts left dsv4's module-owned
    // state (rings + kv/comp caches) permanently ahead of the rolled-back
    // KVCache shell. This test reproduces the bypassing caller directly:
    // nextPld on such a generator must (a) report the chokepoint flip via
    // `gen.pld_enabled == false`, (b) never run a verify forward
    // (`pld_attempted == 0`), and (c) emit the exact serial-decode tokens.
    //
    // Fabricate the mini with:
    //   python3 tests/dsv4_mlx_ref.py --fabricate /tmp/dsv4-mini
    //   DSV4_MINI=/tmp/dsv4-mini zig build test -Dtest-filter=DSV4_MINI
    const path_z = env.get(.DSV4_MINI) orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();


    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    // Prompt = every vocab id once. With key_len=1 any sampled t1 < V has an
    // earlier occurrence, so PLD's lookup ALWAYS proposes a draft — on the
    // pre-fix code that guarantees a verify forward (and the corruption);
    // random-content prompts can idle in the cold path and mask the bug
    // (exactly how the first two live requests read as "guards held").
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 10;

    // Serial baseline: the regular scheduler shape (skip_lazy_preforward).
    var serial: [want]u32 = undefined;
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .skip_lazy_preforward = true,
        });
        defer gen.deinit(allocator);
        var n: usize = 0;
        while (n < want) {
            const t = (try gen.next(allocator)) orelse break;
            serial[n] = t;
            n += 1;
        }
        try testing.expectEqual(want, n);
    }

    // Bypass arm: init asks for PLD (the app's always-on flags), the dsv4
    // chokepoint flips it off, and the caller drives nextPld anyway — the
    // scheduler tick's exact live behavior before specTickMode grew the
    // generator-state conjunct.
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);

        var pld_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextPld(allocator, 4, 1)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    pld_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        try testing.expectEqual(@as(u64, 0), gen.pld_attempted);
        try testing.expectEqualSlices(u32, serial[0..], pld_toks[0..]);
    }

    // DSpark arm: the SAME app-shaped init (spec flags on, greedy) now arms
    // dsv4's own draft mode on a stage-bearing checkpoint. The tick driver
    // is nextDspark; the sequence must match serial (the batch-verify vs
    // single-token kernel-choice class allows a late near-tie flip on the
    // random mini — the module-level dsparkRound gate pins the loop itself,
    // this pins the GENERATOR wiring: engagement, step accounting, and the
    // shell-cache mirror).
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .mtp_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);
        try testing.expect(gen.mtp == null);
        try testing.expect(gen.dspark_enabled);

        var ds_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextDspark(allocator)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    ds_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        // ENGAGEMENT: silent serial fallback is output-identical — count rounds.
        try testing.expect(gen.dspark_attempted >= 1);
        // Shell cache mirrors the module state exactly.
        try testing.expectEqual(intree.dsv4(&xfm).?.dec_state.?.n, gen.ctx.cache.step);
        // Sequence agreement: exact up to the sanctioned near-tie window.
        var first_div: usize = want;
        for (0..want) |k| {
            if (serial[k] != ds_toks[k]) {
                first_div = k;
                break;
            }
        }
        if (first_div < 4) {
            std.debug.print("nextDspark serial={any} dspark={any}\n", .{ serial, ds_toks });
            try testing.expect(false);
        }
        std.debug.print("dsv4 nextDspark (generator): first_div={d}/{d}, rounds={d}, accepts={d}\n", .{ first_div, want, gen.dspark_attempted, gen.dspark_accepted_tokens });
    }
}

test "nextDraw: each stochastic DSpark round takes a fresh draw of the request's seed" {
    var gen: Generator = undefined;
    gen.sampling = .{ .temperature = 0.6, .seed = 7, .draw = 5 };
    const a = gen.nextDraw();
    const b = gen.nextDraw();
    try testing.expectEqual(@as(u64, 5), a.draw);
    try testing.expectEqual(@as(u64, 6), b.draw);
    try testing.expectEqual(@as(?u64, 7), b.seed);
    try testing.expectEqual(@as(u64, 7), gen.sampling.draw);
}

test "dsparkArmFor: greedy and stochastic arms gate on clean sampling" {
    // Greedy clean → the argmax-equality arm (temp 0 or top_k 1).
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.0 }, 0));
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_k = 1 }, 0));
    // Sampled clean → stochastic (the checkpoint-default temp 0.6 agent
    // shape: pi and friends omit temperature, generation_config fills 0.6).
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_p = 0.95 }, 0));
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 1.0, .top_k = 40 }, 0));
    // Penalties / grammar / logprobs stay serial on BOTH arms — the
    // pre-stochastic contract, unchanged.
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.0, .repeat_penalty = 1.1 }, 0));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .presence_penalty = 0.5 }, 0));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6 }, 5));
    var c: Constraint = undefined;
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .constraint = &c }, 0));
}

test "dsv4: stochastic dspark engages at sampled temperature and keeps the exit invariant (DSV4_MINI)" {
    // The motivating traffic shape: the checkpoint ships generation_config
    // temp 0.6 and agent CLIs omit temperature, so every real agent request
    // ran serial while only pinned `--temp 0` earned DSpark. The stochastic
    // arm ports the MTP probsAllPositions acceptance (one-hot Leviathan over
    // filtered target probs) onto dsv4's own draft stages. Engagement is
    // COUNTED (dspark_attempted) — a silent serial fallback emits perfectly
    // plausible tokens.
    env.set(.MLX_SERVE_DSV4_DSPARK, "1");
    const path_z = env.get(.DSV4_MINI) orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();


    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    // The agent-default request shape, seeded so the run is reproducible.
    const sampled = SamplingParams{ .temperature = 0.6, .top_p = 0.95, .seed = 0xD54A };

    var xfm_cfg: model_mod.ModelConfig = undefined;
    var xfm = try servedMini(io, allocator, path, &xfm_cfg);
    defer xfm_cfg.deinit(allocator);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen.deinit(allocator);

    try testing.expect(gen.dspark_enabled);
    try testing.expect(gen.dspark_stochastic);

    const mdl = intree.dsv4(&xfm).?;
    const want: usize = 10;
    var n: usize = 0;
    while (n < want) {
        const t1 = gen.next_token_id;
        const r = (try gen.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expect(r.tokens.len >= 1);
        try testing.expectEqual(t1, r.tokens[0]);
        try testing.expect(r.accepted_tokens <= mdl.ds_block);
        for (r.tokens) |tok| {
            try testing.expect(tok < mdl.vocab);
            n += 1;
        }
        // Module state and the shell-cache mirror stay in lockstep with
        // exactly what was committed (the v2 spec exit invariant).
        try testing.expectEqual(prompt.len + n, mdl.dec_state.?.n);
        try testing.expectEqual(mdl.dec_state.?.n, gen.ctx.cache.step);
    }
    try testing.expect(n >= want);
    // ENGAGEMENT: count rounds, never output shape.
    try testing.expect(gen.dspark_attempted >= 1);
    // Exit invariant survives a hand-off: a following serial step decodes a
    // valid token (finite logits) from the state the rounds left behind.
    const t = (try gen.next(allocator)) orelse return error.TestExpectedResult;
    try testing.expect(t < mdl.vocab);
    std.debug.print("dsv4 stochastic dspark (generator): {d} tokens over {d} rounds, {d} drafts accepted\n", .{ n, gen.dspark_attempted, gen.dspark_accepted_tokens });

    // b==0 arm: an over-threshold confidence gate submits NOTHING — the round
    // verifies t1 alone, commits it, and samples the next trunk token from
    // row 0's filtered probs (the stochastic sibling of the greedy
    // confidence-gate test), on a fresh Transformer whose gate is set past any confidence.
    var xfm2_cfg: model_mod.ModelConfig = undefined;
    var xfm2 = try servedMini(io, allocator, path, &xfm2_cfg);
    defer xfm2_cfg.deinit(allocator);
    defer xfm2.deinit();
    intree.dsv4(&xfm2).?.ds_conf_thr = 999999;
    var gen2 = try Generator.initWithOptions(io, allocator, &xfm2, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen2.deinit(allocator);
    try testing.expect(gen2.dspark_enabled);
    var n2: usize = 0;
    while (n2 < 4) {
        const r = (try gen2.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expectEqual(@as(u32, 0), r.accepted_tokens);
        try testing.expectEqual(@as(usize, 1), r.tokens.len);
        try testing.expect(r.tokens[0] < mdl.vocab);
        n2 += 1;
    }
    try testing.expectEqual(@as(usize, 4), n2);
    try testing.expect(gen2.dspark_attempted >= 4);
    try testing.expectEqual(intree.dsv4(&xfm2).?.dec_state.?.n, gen2.ctx.cache.step);
    std.debug.print("dsv4 stochastic dspark (b==0 gate): {d} single-token rounds\n", .{n2});
}

test "dsparkVerifyDrafts: a sure head verifies the block, an unsure one t1 alone, a doubt mid-block cuts there" {
    const G = Generator;
    var calib = dflash_policy.Calibration{};
    try testing.expectEqual(@as(usize, 5), G.dsparkVerifyDrafts(&.{ 9, 9, 9, 9, 9 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{ 0, 0, 0, 0, 0 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 2), G.dsparkVerifyDrafts(&.{ 9, 9, -4, 9, 9 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{}, &calib, 17.6));
    // A head that keeps claiming certainty the trunk rejects stops buying rows.
    for (0..40) |_| calib.observe(G.sigmoid(9), false);
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{ 9, 9, 9, 9, 9 }, &calib, 17.6));
}

test "dspark drafting backoff: prose rounds that lose to the plain step are skipped, copy-like rounds are not" {
    // A round that commits few tokens for its cost loses to the plain step (costs in ms).
    var prose = dflash_policy.Drafting{};
    try testing.expect(prose.shouldDraft());
    prose.record(3.2 * 17.6, 66);
    try testing.expect(!prose.shouldDraft());
    var echo = dflash_policy.Drafting{};
    try testing.expect(echo.shouldDraft());
    echo.record(5.5 * 17.6, 70);
    try testing.expect(echo.shouldDraft());
}
