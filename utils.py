"""
ReCrit utility helpers.

Includes logging, directory management, distributed initialization, model loading, optimizers, checkpoints, and metric logging.
"""

import os
import math
import json
import logging
import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_file_logging(log_path: str):
    """Write every logger output stream to a file while keeping terminal output enabled."""
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
    logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def make_run_dir(base_dir: str) -> str:
    """Create a versioned run directory under base_dir with the format vXXX_YYYYMMDD_HHMMSS."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir()
                if d.is_dir() and len(d.name) > 4 and d.name[0] == "v" and d.name[1:4].isdigit()]
    max_ver = max((int(d[1:4]) for d in existing), default=-1)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"v{max_ver + 1:03d}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize torch.distributed (torchrun sets RANK and related environment variables automatically)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)  # Ensure that .cuda() binds to the correct device in both single-GPU and multi-GPU settings

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        logger.info(f"[DDP] rank={rank}/{world_size}, local_rank={local_rank}")

    return rank, local_rank, world_size


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def broadcast_string(s: str) -> str:
    """Broadcast a string from rank 0 to all processes."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return s
    if is_main_process():
        data = s.encode("utf-8")
        size = torch.tensor([len(data)], dtype=torch.long).cuda()
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    else:
        size = torch.tensor([0], dtype=torch.long).cuda()
        tensor = torch.zeros(0, dtype=torch.uint8).cuda()
    dist.broadcast(size, src=0)
    if not is_main_process():
        tensor = torch.zeros(size.item(), dtype=torch.uint8).cuda()
    dist.broadcast(tensor, src=0)
    if not is_main_process():
        s = bytes(tensor.cpu().numpy().tobytes()).decode("utf-8")
    return s


# ---------------------------------------------------------------------------
# Model + vLLM initialization
# ---------------------------------------------------------------------------

def load_training_model(config, local_rank: int, world_size: int) -> torch.nn.Module:
    """Load the training model and wrap it with DDP when using multiple GPUs."""
    logger.info(f"[Model] Loading {config.model_path} ...")

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=dtype,
        device_map=None,          # Disable device_map and control placement manually
        trust_remote_code=True,
    )
    model = model.cuda(local_rank)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"[Model] {total_params:.2f}B parameters loaded on GPU {local_rank}")
    return model


def load_reference_model(config, local_rank: int) -> torch.nn.Module:
    """Load the frozen reference model used for the KL penalty.

    It uses the same checkpoint as the training model, but:
    - it is not wrapped with DDP (it does not participate in gradient synchronization)
    - gradient_checkpointing is disabled (it is unnecessary under no_grad)
    - every parameter is set to requires_grad=False
    - it is initially loaded onto the GPU and moved to the CPU during rollout to free GPU memory
    """
    logger.info(f"[RefModel] Loading frozen reference model from {config.model_path} ...")

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    ref_model = ref_model.cuda(local_rank)
    ref_model.requires_grad_(False)
    ref_model.eval()

    total_params = sum(p.numel() for p in ref_model.parameters()) / 1e9
    logger.info(f"[RefModel] {total_params:.2f}B frozen parameters loaded on GPU {local_rank}")
    return ref_model


def load_vllm_engine(config):
    """Initialize the vLLM LLM instance in sleep mode and wake it later when needed.

    In multi-GPU mode, every rank calls this function independently. Because run.sh restricts CUDA_VISIBLE_DEVICES to
    a single GPU, each rank's vLLM engine automatically binds to its corresponding physical GPU.
    """
    from vllm import LLM
    from rollout import vllm_sleep

    logger.info("[vLLM] Initializing engine (will sleep immediately after) ...")
    llm = LLM(
        model=config.model_path,
        tokenizer=config.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        enforce_eager=config.vllm_enforce_eager,
        max_model_len=config.vllm_max_model_len,
        enable_sleep_mode=True,
        trust_remote_code=True,
    )

    # Put the engine to sleep immediately after initialization so the training model can use the GPU
    vllm_sleep(llm, level=1)
    logger.info("[vLLM] Engine initialized and sleeping.")
    return llm


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(model, config, total_steps: int):
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight", "norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, config):
    """Save a checkpoint in the ms-swift-compatible format.

    Keep it consistent with the ms-swift SFT checkpoint format:
    - merge the LLM training weights with the base-model visual-encoder weights
    - save shards as safetensors (max 5 GB each)
    - copy the full multimodal config and processor settings
    """
    if not is_main_process():
        return
    if config.save_total_limit == 0:
        logger.info(f"[Checkpoint] save_total_limit=0, skipping checkpoint save.")
        return
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards

    # Delete old checkpoints first so space is released before writing the new checkpoint
    if config.save_total_limit > 0:
        ckpt_dirs = sorted(
            [d for d in Path(config.output_dir).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: d.stat().st_mtime,
        )
        # Keep save_total_limit - 1 checkpoints so there is room for the next one
        n_to_keep = config.save_total_limit - 1
        if len(ckpt_dirs) > n_to_keep:
            for old_dir in ckpt_dirs[:len(ckpt_dirs) - n_to_keep]:
                shutil.rmtree(str(old_dir))
                logger.info(f"[Checkpoint] Deleted old checkpoint: {old_dir}")

    out = Path(config.output_dir) / f"checkpoint-epoch{epoch}-step{step}"
    out.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP
    raw = model.module if hasattr(model, "module") else model

    # ── Build The Full state_dict (LLM + Visual Encoder) ──────────────────────
    # When AutoModelForCausalLM loads Qwen3_5ForConditionalGeneration,
    # state_dict() returns two copies of the tied weights (model.layers.* and
    # model.language_model.layers.*). These must be filtered so that only keys present in the base model are kept.
    base_path = Path(config.model_path)
    base_index_path = base_path / "model.safetensors.index.json"

    if base_index_path.exists():
        with open(base_index_path) as f:
            base_index = json.load(f)
        base_keys = set(base_index["weight_map"].keys())
    else:
        base_keys = None  # Skip filtering when no index file is present

    # Extract LLM weights and filter out tied-weight aliases
    full_sd = raw.state_dict()
    if base_keys is not None:
        state_dict = {k: v.cpu() for k, v in full_sd.items() if k in base_keys}
    else:
        state_dict = {k: v.cpu() for k, v in full_sd.items()}
    n_llm = len(state_dict)

    # Load missing weights such as the visual encoder from the base model
    if base_index_path.exists():
        visual_shards = set()
        for key, shard_file in base_index["weight_map"].items():
            if key not in state_dict:
                visual_shards.add(shard_file)
        for shard_file in visual_shards:
            shard_path = base_path / shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in state_dict:
                        state_dict[key] = f.get_tensor(key)
    else:
        base_single = base_path / "model.safetensors"
        if base_single.exists():
            with safe_open(str(base_single), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in state_dict:
                        state_dict[key] = f.get_tensor(key)

    n_visual = len(state_dict) - n_llm
    logger.info(f"[Checkpoint] state_dict: {n_llm} LLM + {n_visual} visual = {len(state_dict)} total")

    # ── Save Sharded safetensors ──────────────────────────────────────
    plan = split_torch_state_dict_into_shards(
        state_dict, max_shard_size="5GB",
    )
    for filename, tensors in plan.filename_to_tensors.items():
        shard = {k: state_dict[k] for k in tensors}
        save_file(shard, str(out / filename))

    # Write index.json
    if not plan.is_sharded:
        # Single-file checkpoint: no index is required
        pass
    else:
        index = {
            "metadata": plan.metadata,
            "weight_map": plan.tensor_to_filename,
        }
        with open(str(out / "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    # ── Copy Configuration Files (Using The Full Multimodal Base Configuration) ──────────────────────
    for cfg_file in [
        "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json",
    ]:
        src = base_path / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(out / cfg_file))

    tokenizer.save_pretrained(str(out))

    # Training-state metadata
    trainer_state = {"epoch": epoch, "opt_step": step}
    if config.save_optimizer:
        trainer_state["optimizer"] = optimizer.state_dict()
        trainer_state["scheduler"] = scheduler.state_dict()
        logger.info(f"[Checkpoint] Including optimizer/scheduler state.")
    torch.save(trainer_state, str(out / "trainer_state.pt"))
    logger.info(f"[Checkpoint] Saved to {out} ({len(plan.filename_to_tensors)} shards, {len(state_dict)} params)")


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------

_writer = None


def get_writer(log_dir: str):
    global _writer
    if _writer is None and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            _writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("TensorBoard not available, skipping.")
    return _writer


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a readable string such as '2h 05m' or '1d 3h 12m'."""
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d > 0:
        return f"{d}d {h}h {m:02d}m"
    return f"{h}h {m:02d}m"


def log_metrics(metrics: Dict, step: int, config):
    if not is_main_process():
        return
    w = get_writer(os.path.join(config.output_dir, "tb_logs"))
    if w:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                w.add_scalar(k, v, step)
    # Write to both the terminal and the log file
    parts = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )
    logger.info(f"[Metrics step={step}] {parts}")
    # Write to JSONL
    record = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    record.update(metrics)
    jsonl_path = os.path.join(config.output_dir, "metrics.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
