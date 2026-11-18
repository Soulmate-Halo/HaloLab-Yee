"""KAD worker kernel: DSpark draft inputs with a per-request block width.

Imported lazily by ``vllm_kad_worker`` (and by ``image_tests/test_varlen_kernel.py``); importing it
needs torch, triton and vllm.

``_prepare_dspark_varlen_inputs_kernel`` is a copy of ``_prepare_dflash_inputs_kernel``
(``vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`` :521-718, vLLM 0.31.0) for the DSpark
layout (``sample_from_anchor=True``, ``num_query_per_req == num_speculative_steps``).  Changes:

* request ``i`` owns ``w[i]`` query tokens starting at ``qstart[i]`` instead of
  ``num_query_per_req`` tokens starting at ``i * num_query_per_req``;
* ``seq_lens[i] = min(last_valid_pos + 1 + w[i], max_model_len)``;
* sample rows are laid out ``[num_reqs, max_w]``: row ``i * max_w + off`` samples query
  ``qstart[i] + off`` for ``off < w[i]``; rows ``w[i] <= off < max_w`` are inert
  (``sample_indices = qstart[i]``, ``sample_pos = 0``, ``sample_idx_mapping = -1``);
* tail padding starts at ``total_query_tokens`` (query side) and ``num_reqs * max_w``
  (sample side).

Context positions / slots, the bonus token, ``cp_local_slot``, the null-block rule and the
clamp of the query positions are unchanged.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.cp_utils import cp_local_slot


@triton.jit(do_not_specialize=["max_w", "total_query_tokens"])
def _prepare_dspark_varlen_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # KAD: per-request block width and exclusive prefix sum of it.
    w_ptr,
    qstart_ptr,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_speculative_steps,
    max_w,
    total_query_tokens,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    # KAD: this request's query block is [query_base, query_base + w_i).
    query_base = tl.load(qstart_ptr + req_idx)
    w_i = tl.load(w_ptr + req_idx)

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + w_i)
    query_off = j - num_valid_ctx

    # --- Context positions / slots (unchanged) ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    local_ctx_slot = cp_local_slot(
        ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // (block_size * CP_SIZE)
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_resident = is_query & (q_block_id != 0)
    local_q_slot = cp_local_slot(
        query_pos,
        q_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    q_slot = tl.where(
        q_resident,
        local_q_slot,
        PAD_SLOT_ID,
    )

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping (DSpark: sample from the anchor) ---
    # Row req_idx * max_w + off samples query off; rows off in [w_i, max_w) are inert.
    sample_idx = req_idx * max_w + query_off
    sample_pos = query_pos + 1
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_query)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_query)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_query)
    is_pad_sample = (j >= num_valid_ctx + w_i) & (j < num_valid_ctx + max_w)
    # Inert rows gather this request's anchor (a valid row of the draft hidden states, so the
    # gather never reads out of bounds) and map to -1 (sampling writes no draft logits).
    tl.store(out_sample_indices_ptr + sample_idx, query_base, mask=is_pad_sample)
    tl.store(out_sample_pos_ptr + sample_idx, 0, mask=is_pad_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, -1, mask=is_pad_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + w_i, max_model_len),
        )
        # Copy sampling state.
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs.
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, total_query_tokens, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * max_w
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            for i in range(total_query_tokens, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def grid_for(max_target_query_len: int, max_w: int) -> tuple[int, int]:
    """(BLOCK_SIZE, num_blocks) covering ctx + the widest query block of the batch.

    Same rule as ``prepare_dflash_inputs`` with ``num_query_per_req`` replaced by ``max_w``.
    """
    max_tokens_per_req = int(max_target_query_len) + int(max_w)
    block = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    return block, triton.cdiv(max_tokens_per_req, block)


def launch_varlen_kernel(
    *,
    out_input_ids: torch.Tensor,
    out_positions: torch.Tensor,
    out_query_start_loc: torch.Tensor,
    out_seq_lens: torch.Tensor,
    out_query_slot_mapping: torch.Tensor,
    out_context_positions: torch.Tensor,
    out_context_slot_mapping: torch.Tensor,
    out_sample_indices: torch.Tensor,
    out_sample_pos: torch.Tensor,
    out_sample_idx_mapping: torch.Tensor,
    out_temperature: torch.Tensor,
    out_seeds: torch.Tensor,
    target_positions: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    w: torch.Tensor,
    qstart: torch.Tensor,
    num_reqs: int,
    max_target_query_len: int,
    max_w: int,
    total_query_tokens: int,
    parallel_drafting_token_id: int,
    block_size: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
    pad_slot_id: int = PAD_SLOT_ID,
) -> None:
    """Launch the varlen kernel on plain tensors (``w`` and ``qstart`` int32, on the device)."""
    assert num_reqs > 0
    assert 1 <= int(max_w) <= int(num_speculative_steps)
    block, num_blocks = grid_for(max_target_query_len, max_w)
    _prepare_dspark_varlen_inputs_kernel[(int(num_reqs), num_blocks)](
        out_input_ids,
        out_positions,
        out_query_start_loc,
        out_seq_lens,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_sample_indices,
        out_sample_pos,
        out_sample_idx_mapping,
        out_temperature,
        out_seeds,
        target_positions,
        target_query_start_loc,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        temperature,
        seeds,
        block_table,
        block_table.stride(0),
        w,
        qstart,
        int(parallel_drafting_token_id),
        int(block_size),
        int(num_speculative_steps),
        int(max_w),
        int(total_query_tokens),
        int(max_num_reqs),
        int(max_num_tokens),
        int(max_model_len),
        int(cp_rank),
        PAD_SLOT_ID=pad_slot_id,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=block,
    )


def prepare_dspark_varlen_inputs(spec, gid_index: int, gid: int, input_batch, num_sampled,
                                 num_rejected, last_sampled, next_prefill_tokens, temperature,
                                 seeds, w_gpu: torch.Tensor, q_gpu: torch.Tensor, max_w: int,
                                 total_query_tokens: int) -> None:
    """The varlen counterpart of the ``prepare_dflash_inputs`` call in ``DFlashSpeculator.propose``
    (same buffers, same argument sources) for one draft KV cache group."""
    num_reqs = input_batch.num_reqs
    max_target_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
    bt = spec.block_tables
    launch_varlen_kernel(
        out_input_ids=spec.input_buffers.input_ids,
        out_positions=spec.input_buffers.positions,
        out_query_start_loc=spec.input_buffers.query_start_loc,
        out_seq_lens=spec.input_buffers.seq_lens,
        out_query_slot_mapping=bt.slot_mappings[gid],
        out_context_positions=spec.context_positions,
        out_context_slot_mapping=spec._context_slot_mappings[gid_index],
        out_sample_indices=spec.sample_indices,
        out_sample_pos=spec.sample_pos,
        out_sample_idx_mapping=spec.sample_idx_mapping,
        out_temperature=spec.temperature,
        out_seeds=spec.seeds,
        target_positions=input_batch.positions,
        target_query_start_loc=input_batch.query_start_loc,
        idx_mapping=input_batch.idx_mapping,
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill_tokens,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        temperature=temperature,
        seeds=seeds,
        block_table=bt.input_block_tables[gid],
        w=w_gpu,
        qstart=q_gpu,
        num_reqs=num_reqs,
        max_target_query_len=max_target_query_len,
        max_w=max_w,
        total_query_tokens=total_query_tokens,
        parallel_drafting_token_id=spec.parallel_drafting_token_id,
        block_size=bt.kernel_block_sizes[gid],
        num_speculative_steps=spec.num_speculative_steps,
        max_num_reqs=spec.max_num_reqs,
        max_num_tokens=spec.max_num_tokens,
        max_model_len=spec.max_model_len,
        cp_rank=bt.cp_rank,
        cp_size=spec.dcp_size,
        cp_interleave=bt.cp_interleave,
    )
