"""
Main entry point for ReCrit training.

Usage (normally launched through run.sh, which auto-detects single-GPU vs.
multi-GPU mode from CUDA_VISIBLE_DEVICES):
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python -m train \
        --model_path /path/to/model --train_dataset data.jsonl

    # Multi GPU (run.sh launches one process per GPU)
    # Each process only sees one GPU, so LOCAL_RANK always stays 0.
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 python -m train ... &
    CUDA_VISIBLE_DEVICES=1 RANK=1 LOCAL_RANK=0 WORLD_SIZE=2 python -m train ... &

Architecture (distributed rollout + DDP training):
    Each GPU independently runs the full pipeline, and DDP only communicates
    during gradient synchronization:
        1. vLLM wake_up + sync weights  -> rollout
        2. vLLM sleep                   -> release GPU memory
        3. compute rewards              -> Judge API per rank
        4. compute advantages           -> GRPO group normalization per rank
        5. build training batch         -> per rank
        6. train_step                   -> micro-batch accumulation + DDP all-reduce
"""

import os

# Fixed environment defaults. Users do not need to set them in launch scripts.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_NET_GDR_DISABLE", "1")
os.environ.setdefault("NCCL_COLLNET_ENABLE", "0")
# Note: expandable_segments is enabled later via
# torch._C._accelerator_setAllocatorSettings() after vLLM initialization.
import dataclasses
import gc
import hashlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from config import parse_args
from dataset import QADataset, collate_fn
from reward import compute_all_rewards, compute_grpo_advantages, recompute_advantages_on_kept, quadrant_stats
from rollout import run_rollout, build_training_batch, vllm_sleep, vllm_wake_and_sync, verify_vllm_weights
from trainer import train_step
from utils import (
    setup_file_logging, make_run_dir, setup_ddp, is_main_process,
    broadcast_string, load_training_model, load_reference_model, load_vllm_engine,
    build_optimizer_scheduler, save_checkpoint, log_metrics, fmt_duration,
)

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config):
    # vLLM must be created before any other CUDA initialization.
    # Remove distributed env vars temporarily so they are not inherited by
    # vLLM EngineCore worker processes and accidentally reused for NCCL init.
    # run.sh already pins each rank to a single visible GPU.
    _DIST_VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    _dist_backup = {k: os.environ.pop(k) for k in _DIST_VARS if k in os.environ}
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

    llm = load_vllm_engine(config)

    # CUDA allocator is already initialized during vLLM startup, so setting
    # the env var now is too late. Use the programmatic API instead to reduce
    # fragmentation during subsequent training allocations.
    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")

    os.environ.update(_dist_backup)

    rank, local_rank, world_size = setup_ddp()

    # Sanity-check the configured vLLM context length.
    _PROMPT_OVERHEAD = 2048
    _BRIDGE_OVERHEAD = 200
    _worst_case = (
        _PROMPT_OVERHEAD
        + config.num_turns * config.max_new_tokens
        + (config.num_turns - 1) * _BRIDGE_OVERHEAD
    )
    if _worst_case >= config.vllm_max_model_len:
        raise ValueError(
            f"vllm_max_model_len={config.vllm_max_model_len} is too small.\n"
            f"Worst-case estimate: prompt_overhead({_PROMPT_OVERHEAD}) + "
            f"num_turns({config.num_turns}) × max_new_tokens({config.max_new_tokens}) + "
            f"(num_turns-1) × bridge({_BRIDGE_OVERHEAD}) = {_worst_case} tokens.\n"
            f"Please set --vllm_max_model_len to at least {_worst_case + 1024}."
        )
    if _worst_case > config.vllm_max_model_len * 0.8:
        logger.warning(
            f"[Train] vllm_max_model_len={config.vllm_max_model_len} may be too small: "
            f"worst-case estimate {_worst_case} tokens "
            f"({ _worst_case / config.vllm_max_model_len:.0%} of the limit)."
        )

    # Create a versioned run directory on rank 0 and broadcast it.
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        run_dir = make_run_dir(config.output_dir)
    else:
        run_dir = ""
    run_dir = broadcast_string(run_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.output_dir = run_dir
    setup_file_logging(os.path.join(run_dir, "train.log"))
    logger.info(f"[Train] Run directory: {run_dir}")

    # Save the resolved training arguments for reproducibility.
    if is_main_process():
        args_record = dataclasses.asdict(config)
        args_record["world_size"] = world_size
        args_record["python"] = sys.executable
        args_path = os.path.join(run_dir, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args_record, f, indent=2, ensure_ascii=False)
        logger.info(f"[Train] Args saved to {args_path}")

    # Seed each rank independently while keeping the run reproducible.
    random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)

    # Tokenizer and dataset.
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token






    dataset = QADataset(
        config.train_dataset,
        judge_mode=config.judge_mode,
        add_format_prompt=config.add_format_prompt,
    )
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=config.seed, drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
    steps_per_epoch = len(dataloader)


    model = load_training_model(config, local_rank, world_size)


    ref_model = None
    if config.kl_beta > 0:
        ref_model = load_reference_model(config, local_rank)

        ref_model.cpu()
        torch.cuda.empty_cache()


    accum_steps = config.gradient_accumulation_steps
    accum_scale = 1.0 / accum_steps

    last_group_size = steps_per_epoch % accum_steps
    last_group_accum_scale = 1.0 / last_group_size if last_group_size > 0 else accum_scale
    last_group_start = steps_per_epoch - last_group_size if last_group_size > 0 else steps_per_epoch
    total_data_steps = steps_per_epoch * config.num_train_epochs
    total_steps = math.ceil(total_data_steps / accum_steps)
    optimizer, scheduler = build_optimizer_scheduler(model, config, total_steps)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    accum_counter = 0
    accum_metrics_list = []
    accum_results_all = []
    accum_n_samples = 0
    accum_n_dropped = 0
    accum_seq_lengths = []
    accum_total_tokens = 0
    _accum_start_time = time.monotonic()
    _train_start_time = time.monotonic()
    logger.info(f"[Train] Starting: {config.num_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch, total {total_steps} optimizer steps "
                f"(gradient_accumulation={accum_steps})")
    logger.info(f"[Train] Data parallel: world_size={world_size}, "
                f"per_device_prompts={config.per_device_train_batch_size}, "
                f"per_device_samples={config.per_device_train_batch_size * config.num_generations}, "
                f"global_samples={config.per_device_train_batch_size * config.num_generations * world_size}, "
                f"effective_batch_prompts={config.per_device_train_batch_size * accum_steps * world_size}")
    if config.turn_loss_weights:
        logger.info(
            "[Train] Turn-wise loss weighting enabled: "
            f"turn_loss_weights={list(config.turn_loss_weights)} "
            "(token_weights extends the original completion mask; later turns receive larger RL weight)"
        )
    else:
        logger.info(
            "[Train] Turn-wise loss weighting disabled: "
            "all assistant turns use the default equal loss weight."
        )

    for epoch in range(config.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, batch_prompts in enumerate(dataloader):


            if config.debug:
                qs = "|".join(p["question"][:50] for p in batch_prompts)
                h = hashlib.md5(qs.encode()).hexdigest()[:8]
                logger.info(
                    f"[DEBUG rank={rank}] opt_step={global_step+1} "
                    f"rollout={accum_counter+1}/{accum_steps} "
                    f"data_hash={h} n_prompts={len(batch_prompts)} "
                    f"first_q={batch_prompts[0]['question'][:60]!r}"
                )


            gc.collect()
            torch.cuda.empty_cache()

            if config.debug:
                alloc_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"[DEBUG rank={rank}] pre-wake CUDA memory: "
                    f"allocated={alloc_gb:.2f}GiB reserved={reserved_gb:.2f}GiB "
                    f"(after gc+empty_cache)"
                )

            raw_model = model.module if hasattr(model, "module") else model
            need_sync = (accum_counter == 0)
            vllm_wake_and_sync(llm, raw_model, sync_weights=need_sync)


            if config.debug and need_sync:
                try:
                    torch.cuda.empty_cache()
                    verify_vllm_weights(llm, raw_model)

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "[DEBUG] verify_vllm_weights skipped due to CUDA OOM. "
                        "This only affects the debug check, not training correctness. "
                        "Consider increasing vllm_gpu_memory_utilization or reducing model size."
                    )
                    torch.cuda.empty_cache()

            results = run_rollout(
                llm, tokenizer, batch_prompts, config.num_generations,
                config, num_turns=config.num_turns,
                debug=config.debug, rank=rank, world_size=world_size,
                step_label=f"opt_step={global_step+1}/{total_steps}, accum_step={accum_counter+1}/{accum_steps}",
            )

            vllm_sleep(llm, level=1)


            if ref_model is not None:
                ref_model.cuda(local_rank)


            if config.debug:
                expected_samples = len(batch_prompts) * config.num_generations
                logger.info(
                    f"[DEBUG rank={rank}] opt_step={global_step+1} "
                    f"rollout={accum_counter+1}/{accum_steps} "
                    f"got {len(results)} samples (expected {expected_samples}), "
                    f"sync_weights={need_sync}"
                )

                r0 = results[0]
                for t_idx, resp in enumerate(r0["responses"]):
                    has_think_open  = "<think>" in resp
                    has_think_close = "</think>" in resp
                    escaped = resp.replace("\n", "\\n")
                    logger.info(
                        f"[DEBUG rank={rank}] sample[0] turn{t_idx} "
                        f"({len(r0['token_ids'][t_idx])} tokens, "
                        f"<think>={has_think_open}, </think>={has_think_close}):\\n"
                        f"{escaped}"
                    )


            results = compute_all_rewards(results, config)


            results = compute_grpo_advantages(results, config.num_generations)


            batch, kept_indices = build_training_batch(
                results,
                tokenizer,
                config.max_seq_length,
                turn_loss_weights=list(config.turn_loss_weights),
            )

            _batch_n_samples = len(results)
            _batch_n_dropped = 0
            _batch_seq_lengths = []
            _batch_total_tokens = 0
            if batch is not None:
                n_dropped = len(results) - len(kept_indices)
                _batch_n_dropped = n_dropped
                _batch_seq_lengths = batch["attention_mask"].sum(dim=1).tolist()
                _batch_total_tokens = int(batch["attention_mask"].sum().item())
                if n_dropped > 0:
                    kept_results = [results[i] for i in kept_indices]

                    all_prompt_ids = {r["prompt_idx"] for r in results}
                    kept_prompt_ids = {r["prompt_idx"] for r in kept_results}
                    lost_prompts = all_prompt_ids - kept_prompt_ids
                    if lost_prompts:
                        logger.warning(
                            f"[Train] All samples for prompt_idx {lost_prompts} were "
                            "dropped for overlength, so those prompts will not "
                            "participate in this training step."
                        )

                    logger.warning(
                        f"[Train] Dropped {n_dropped}/{len(results)} sample(s) for "
                        f"overlength. Recomputing advantages on the remaining "
                        f"{len(kept_indices)} sample(s)."
                    )
                    recompute_advantages_on_kept(kept_results)
                    batch["advantages"] = torch.tensor(
                        [r["advantage"] for r in kept_results], dtype=torch.float32
                    )
            else:

                _batch_n_dropped = _batch_n_samples



            has_batch_t = torch.tensor(
                [1 if batch is not None else 0], dtype=torch.long,
            ).cuda()
            if world_size > 1:
                dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
            all_have_batch = has_batch_t.item() != 0
            del has_batch_t
            if not all_have_batch:
                logger.warning(
                    "[Train] At least one rank has an empty batch. Skipping this step on all ranks."
                )
                continue



            accum_n_samples += _batch_n_samples
            accum_n_dropped += _batch_n_dropped
            accum_seq_lengths.extend(_batch_seq_lengths)
            accum_total_tokens += _batch_total_tokens

            accum_counter += 1
            is_last_batch_in_epoch = (batch_idx == steps_per_epoch - 1)




            is_opt_step = (accum_counter >= accum_steps) or is_last_batch_in_epoch

            actual_scale = last_group_accum_scale if batch_idx >= last_group_start else accum_scale

            gc.collect()
            torch.cuda.empty_cache()
            step_metrics = train_step(
                model, batch, optimizer, config,
                is_opt_step=is_opt_step,
                accumulation_scale=actual_scale,
                ref_model=ref_model,
            )
            del batch


            if ref_model is not None:
                ref_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()


            accum_metrics_list.append(step_metrics)
            accum_results_all.extend(results)

            if not is_opt_step:
                continue


            scheduler.step()
            global_step += 1
            accum_counter = 0


            n_accum = len(accum_metrics_list)
            agg_metrics = {
                k: sum(m[k] for m in accum_metrics_list) / n_accum
                for k in accum_metrics_list[0]
            }

            agg_metrics["grad_norm"] = accum_metrics_list[-1]["grad_norm"]


            if config.debug and world_size > 1:

                loss_t = torch.tensor([agg_metrics["loss"]], dtype=torch.float64).cuda()
                all_losses = [torch.zeros_like(loss_t) for _ in range(world_size)]
                dist.all_gather(all_losses, loss_t)
                losses_str = ", ".join(
                    f"rank{i}={l.item():.6f}" for i, l in enumerate(all_losses)
                )
                logger.info(f"[DEBUG rank={rank}] per-rank loss: {losses_str}")
                del loss_t, all_losses

                raw = model.module if hasattr(model, "module") else model
                param_sum = sum(p.data.sum().item() for p in raw.parameters())
                param_t = torch.tensor([param_sum], dtype=torch.float64).cuda()
                all_params = [torch.zeros_like(param_t) for _ in range(world_size)]
                dist.all_gather(all_params, param_t)
                params_str = ", ".join(
                    f"rank{i}={p.item():.4f}" for i, p in enumerate(all_params)
                )
                synced = all(
                    abs(p.item() - all_params[0].item()) < 1e-2 for p in all_params
                )
                if synced:
                    logger.info(f"[DEBUG rank={rank}] param_checksum: {params_str} SYNCED")
                else:
                    raise RuntimeError(
                        f"[FATAL] DDP parameters are out of sync! param_checksum: "
                        f"{params_str}. Model parameters have diverged across ranks, "
                        "so the training result is not trustworthy."
                    )
                del param_t, all_params


            if is_main_process() and global_step % config.logging_steps == 0:
                stats = quadrant_stats(accum_results_all)
                turn_lens = [len(r["responses"]) for r in accum_results_all]
                mean_turns = sum(turn_lens) / max(len(turn_lens), 1)
                min_turns  = min(turn_lens) if turn_lens else 0
                mean_seq_length = (
                    sum(accum_seq_lengths) / len(accum_seq_lengths)
                    if accum_seq_lengths else 0.0
                )
                accum_elapsed = time.monotonic() - _accum_start_time
                tokens_per_second = accum_total_tokens / max(accum_elapsed, 1e-3)

                elapsed = time.monotonic() - _train_start_time
                eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0.0
                all_metrics = {
                    "progress/opt_step":  f"{global_step}/{total_steps}",
                    "progress/epoch":     f"{epoch + 1}/{config.num_train_epochs}",
                    "progress/elapsed":   fmt_duration(elapsed),
                    "progress/eta":       fmt_duration(eta),
                    "train/loss":              agg_metrics["loss"],
                    "train/kl":                agg_metrics.get("kl", 0.0),
                    "train/ref_kl":            agg_metrics.get("ref_kl", 0.0),
                    "train/clip_frac":         agg_metrics.get("clip_frac", 0.0),
                    "train/entropy":           agg_metrics.get("entropy", 0.0),
                    "train/grad_norm":         agg_metrics.get("grad_norm", 0.0),
                    "train/lr":                scheduler.get_last_lr()[0],
                    "train/tokens_per_second": tokens_per_second,
                    "reward/frac_correction":  stats["frac_correction"],
                    "reward/frac_robustness":  stats["frac_robustness"],
                    "reward/frac_sycophancy":  stats["frac_sycophancy"],
                    "reward/frac_boundary":    stats["frac_boundary"],
                    "reward/reward_mean":      stats["reward_mean"],
                    "reward/reward_std":       stats["reward_std"],
                    "reward/critic_mean":      stats["critic_mean"],
                    "reward/critic_std":       stats["critic_std"],
                    "reward/repetition_mean":  stats["repetition_mean"],
                    "reward/repetition_std":   stats["repetition_std"],
                    "reward/overlong_mean":    stats["overlong_mean"],
                    "reward/overlong_std":     stats["overlong_std"],
                    "reward/think_fmt_mean":   stats["think_format_mean"],
                    "reward/think_fmt_std":    stats["think_format_std"],
                    "reward/acc_delta":        stats["acc_delta"],
                    "rollout/n_samples":       accum_n_samples,
                    "rollout/n_dropped":       accum_n_dropped,
                    "rollout/mean_seq_length": mean_seq_length,
                    "rollout/mean_turns":      mean_turns,
                    "rollout/min_turns":       float(min_turns),
                }
                def _turn_acc_sort_key(key: str):



                    suffix = key.removeprefix("acc_turn_")
                    turn_str = suffix.split("_", 1)[0]
                    return int(turn_str)

                turn_acc_keys = sorted(
                    (k for k in stats.keys() if k.startswith("acc_turn_")),
                    key=_turn_acc_sort_key,
                )
                for key in turn_acc_keys:
                    all_metrics[f"reward/{key}"] = stats[key]
                log_metrics(all_metrics, global_step, config)
                turn_acc_log = " ".join(
                    f"{key.replace('acc_', '')}={stats[key]:.2f}" for key in turn_acc_keys
                )
                logger.info(
                    f"[Step {global_step:5d}] "
                    f"loss={agg_metrics['loss']:.4f}  "
                    f"grad={agg_metrics['grad_norm']:.3f}  "
                    f"r={stats['reward_mean']:.3f}±{stats['reward_std']:.3f}  "
                    f"critic={stats['critic_mean']:.3f}±{stats['critic_std']:.3f}  "
                    f"rep={stats['repetition_mean']:.3f}±{stats['repetition_std']:.3f}  "
                    f"fmt={stats['think_format_mean']:.2f}±{stats['think_format_std']:.2f}  "
                    f"syco={stats['frac_sycophancy']:.2f} rob={stats['frac_robustness']:.2f} "
                    f"corr={stats['frac_correction']:.2f}  "
                    f"acc_delta={stats['acc_delta']:.2f} "
                    f"{turn_acc_log}  "
                    f"drop={accum_n_dropped}/{accum_n_samples} "
                    f"seq={mean_seq_length:.0f} tok/s={tokens_per_second:.0f}"
                )


            if is_main_process() and config.save_steps > 0 and global_step % config.save_steps == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler,
                                epoch, global_step, config)


            accum_metrics_list.clear()
            accum_results_all.clear()
            accum_n_samples = 0
            accum_n_dropped = 0
            accum_seq_lengths.clear()
            accum_total_tokens = 0
            _accum_start_time = time.monotonic()





        if is_main_process():
            save_checkpoint(model, tokenizer, optimizer, scheduler,
                            epoch + 1, global_step, config)

    if is_main_process():
        if config.save_total_limit == 0:
            logger.info("[Train] Done. No checkpoint saved (save_total_limit=0).")
        else:
            logger.info("[Train] Done. Final checkpoint saved at epoch end.")

    if world_size > 1:
        dist.destroy_process_group()






if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
