"""
ReCrit training hyperparameter configuration.
All command-line arguments are parsed by argparse and packed into ReCritConfig.
"""

import argparse
from dataclasses import dataclass


@dataclass
class ReCritConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_path: str = ""                    # HuggingFace model path or name

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_dataset: str = ""                 # Training JSONL path (must include question/answer fields)
    judge_mode: str = "both"               # Evaluation mode: "close", "open", or "both"
                                            #   close - treat all samples as close-form evaluation (short-answer mode)
                                            #   open  - treat all samples as open-form evaluation (long-form mode)
                                            #   both  - read judge_mode from each sample
                                            #           the dataset must provide the judge_mode field in this case
    add_format_prompt: bool = False         # Whether to append a format-requirements prompt to the end of the question
                                            #   off   - assume the data already contains the format constraint (or does not need one)
                                            #   on    - automatically append a prompt according to the effective judge_mode:
                                            #     close → "provide only the final answer …"
                                            #     open  → "provide the short final answer"

    # ── Judge Model ─────────────────────────────────────────────────────────
    judge_model: str = "gemini-3-flash-preview-nothinking"  # LLM used by structai Judge

    # ── Critic Prompt Mode ───────────────────────────────────────────────────
    critic_prompt_mode: str = "mixed"       # critic prompt mode:
                                            #   mixed      - keep random sampling from the current three-attitude template pool
                                            #   eval_fixed - use a fixed critic prompt aligned with benchmark evaluation
    critic_prompt_text: str = (
        "Can you verify your reasoning? I want to make sure nothing was overlooked."
    )                                       # Fixed critic prompt used when critic_prompt_mode=eval_fixed

    # ── Main Four-Quadrant Reward Weights ───────────────────────────────────────────────────────
    w_correction: float = 1.0               # Correction Reward: wrong -> correct
    w_robustness: float = 0.6               # Robustness Reward: correct -> correct
    w_sycophancy: float = 1.0               # Sycophancy Penalty: correct -> wrong (sycophancy)
    w_boundary: float = 0.1                 # Boundary Penalty: wrong -> wrong (capability boundary)

    # ── Auxiliary Reward Parameters ────────────────────────────────────────────────────────────
    repetition_n_grams: int = 8             # Window size for the n-gram repetition penalty; larger values are stricter
    soft_max_length: int = 4096             # Expected per-turn token limit; linear penalty starts beyond this point
    soft_cache_length: int = 1024           # Overlength buffer: after soft_max_length is exceeded,
                                            #   the penalty decays linearly to -1.0 within soft_cache_length
    repetition_weight: float = 0.2         # Scaling weight for repetition_penalty
    overlong_weight: float = 0.2           # Scaling weight for overlong_penalty
    think_fmt_weight: float = 0.2          # Scaling weight for think_format_reward
    turn_loss_weights: tuple[float, ...] = ()  # Optional per-turn loss weights.
                                               #   Leave empty to keep the default behavior: assistant tokens from all turns contribute equally to the RL loss
                                               #   If provided, the length must equal num_turns. For example:
                                               #     3 turns → 0.0,0.3,1.0
                                               #     4 turns → 0.0,0.1,0.4,1.0

    # ── Rollout Parameters ──────────────────────────────────────────────────────────
    num_generations: int = 8                # Number of sampled trajectories per prompt (GRPO group size G)
    num_turns: int = 2                      # Number of dialogue turns per rollout, including critic interaction
    completion_ratio: float = 0.75         # Early-stop threshold for dynamic max-turn rollout:
                                            #   stop once this fraction of samples reaches num_turns and 100% of samples have completed the minimum turn count required by early stopping
                                            #   for the current two-turn setting, the default minimum turn count is 1; setting this to 1.0 is equivalent to fixed max-turn rollout
    max_new_tokens: int = 4096              # Maximum number of generated tokens per vLLM turn (single-turn cap)
    temperature: float = 1.0               # Sampling temperature (1.0 = standard sampling; lower values are more deterministic)
    top_p: float = 1.0                     # Top-p sampling (1.0 = no truncation)
    system_prompt: str = "You are a helpful assistant."

    # ── vLLM Parameters ─────────────────────────────────────────────────────────────
    vllm_gpu_memory_utilization: float = 0.22   # Fraction of GPU memory pre-reserved by vLLM; smaller values leave more memory for training
    vllm_enforce_eager: bool = False             # Disable CUDA graph to save memory (debug only)
    vllm_max_model_len: int = 65536             # Maximum context window of the vLLM engine (prompt + all generated turns
                                                 #   the total-length upper bound). Must be >= num_turns * max_new_tokens
                                                 #   + prompt length. Setting it too small will truncate multi-turn dialogue

    # ── GRPO Algorithm Parameters ─────────────────────────────────────────────────────────
    epsilon: float = 0.2                    # PPO-clip coefficient that constrains the IS ratio to [1-epsilon, 1+epsilon],
                                            #   preventing overly large single-step policy updates. 0.2 is the standard value and usually does not need tuning
    kl_beta: float = 0.04                   # KL penalty coefficient. Adds beta * KL(pi || pi_ref) to the GRPO loss,
                                            #   anchoring the policy near the initial SFT model and preventing collapse from policy drift.
                                            #   Set to 0 to disable the KL penalty (no reference model is loaded)

    # ── Training Hyperparameters ────────────────────────────────────────────────────────────
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2    # Number of prompts processed per GPU per step (rollout is independent on each GPU)
    learning_rate: float = 2e-6
    warmup_ratio: float = 0.05              # Fraction of total steps used for linear learning-rate warmup
    max_grad_norm: float = 1.0              # Gradient clipping threshold
    weight_decay: float = 0.01
    max_seq_length: int = 8192             # Maximum training sequence length in tokens. The concatenated full sequence
                                            #   (prompt + all turn responses + bridge) exceeding this limit
                                            #   truncates the final turn first; if earlier context is already too long, the entire sample is dropped.
                                            #   Must be <= vllm_max_model_len
    gradient_accumulation_steps: int = 4   # Gradient accumulation steps. Perform one
                                            #   optimizer.step() every N rollout batches, increasing the effective batch size by a factor of N.
                                            #   Reference: ms-swift commonly uses gradient_accumulation_steps=16

    # ── Training Efficiency ──────────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True     # Gradient checkpointing (saves memory; recommended)
    bf16: bool = True                       # Train with bfloat16
    train_micro_batch_size: int = 2         # Number of samples per forward/backward micro-step (split large batches to avoid OOM)

    # ── Output and Logging ────────────────────────────────────────────────────────────
    output_dir: str = "output/recrit"
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    save_optimizer: bool = False           # Whether to save optimizer/scheduler state (~70GB+)

    # ── Debug ────────────────────────────────────────────────────────────────────
    debug: bool = False                    # Enable diagnostic logs for DDP data sharding, parameter synchronization, and related checks

    # ── Seed ──────────────────────────────────────────────────────────────
    seed: int = 42


def parse_args() -> ReCritConfig:
    parser = argparse.ArgumentParser(
        description="ReCrit: Multi-Turn Critic GRPO Training"
    )

    # Required arguments
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--output_dir", default="output/recrit")

    # Optional arguments (one-to-one with config fields)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--judge_mode", default="both",
                        choices=["close", "open", "both"],
                        help="Evaluation mode: close/open/both (for both, the dataset must include judge_mode)")
    parser.add_argument("--add_format_prompt", action="store_true",
                        help="Append a format-requirements prompt to the end of each question (different templates for close/open)")

    # Reward weights
    parser.add_argument("--judge_model", default="gemini-3-flash-preview-nothinking",
                        help="LLM model used by structai Judge")
    parser.add_argument("--critic_prompt_mode", default="mixed",
                        choices=["mixed", "eval_fixed"],
                        help="Critic prompt mode: mixed keeps the current attitude templates; "
                             "eval_fixed uses a single fixed critic prompt.")
    parser.add_argument("--critic_prompt_text",
                        default="Can you verify your reasoning? I want to make sure nothing was overlooked.",
                        help="Fixed critic prompt text used when --critic_prompt_mode eval_fixed.")
    parser.add_argument("--w_correction", type=float, default=1.0)
    parser.add_argument("--w_robustness", type=float, default=0.6)
    parser.add_argument("--w_sycophancy", type=float, default=1.0)
    parser.add_argument("--w_boundary",   type=float, default=0.1)

    # Auxiliary rewards
    parser.add_argument("--repetition_n_grams",     type=int,   default=8)
    parser.add_argument("--soft_max_length",        type=int,   default=4096)
    parser.add_argument("--soft_cache_length",      type=int,   default=1024)
    parser.add_argument("--repetition_weight",      type=float, default=0.2,
                        help="Scaling weight for repetition_penalty")
    parser.add_argument("--overlong_weight",        type=float, default=0.2,
                        help="Scaling weight for overlong_penalty")
    parser.add_argument("--think_fmt_weight",       type=float, default=0.2,
                        help="Scaling weight for think_format_reward")
    parser.add_argument("--turn_loss_weights", default="",
                        help="Optional comma-separated per-turn loss weights applied to assistant tokens only. "
                             "Leave empty to keep the default equal-weight behavior. "
                             "Example: 0.0,0.3,1.0")

    # Rollout
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_turns",       type=int, default=2)
    parser.add_argument("--completion_ratio", type=float, default=0.75,
                        help="Dynamic max-turn rollout early-stop threshold (1.0 means fixed max turns)")
    parser.add_argument("--max_new_tokens",  type=int, default=4096)
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--top_p",           type=float, default=1.0)

    # vLLM
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.22)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_max_model_len",  type=int, default=65536)

    # GRPO
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.04,
                        help="KL penalty coefficient beta to anchor the policy near the initial SFT model (0 = disabled)")

    # Training
    parser.add_argument("--num_train_epochs",           type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size", type=int,  default=2)
    parser.add_argument("--learning_rate",               type=float, default=2e-6)
    parser.add_argument("--warmup_ratio",                type=float, default=0.05)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--max_seq_length",              type=int,   default=8192)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4,
                        help="Gradient accumulation steps (effective batch size = per_device x num_gen x accum_steps x world_size)")

    # Efficiency
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_bf16",                   action="store_true")
    parser.add_argument("--train_micro_batch_size",    type=int, default=2)

    # Logging
    parser.add_argument("--logging_steps",    type=int, default=1)
    parser.add_argument("--save_steps",       type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_optimizer",   action="store_true",
                        help="Save optimizer/scheduler state (about 70GB+, for resume training)")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--debug",            action="store_true",
                        help="Emit diagnostic logs for DDP data sharding, parameter sync, and related checks")

    args = parser.parse_args()

    cfg = ReCritConfig(
        model_path=args.model_path,
        train_dataset=args.train_dataset,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        judge_mode=args.judge_mode,
        add_format_prompt=args.add_format_prompt,
        judge_model=args.judge_model,
        critic_prompt_mode=args.critic_prompt_mode,
        critic_prompt_text=args.critic_prompt_text,
        w_correction=args.w_correction,
        w_robustness=args.w_robustness,
        w_sycophancy=args.w_sycophancy,
        w_boundary=args.w_boundary,
        repetition_n_grams=args.repetition_n_grams,
        soft_max_length=args.soft_max_length,
        soft_cache_length=args.soft_cache_length,
        repetition_weight=args.repetition_weight,
        overlong_weight=args.overlong_weight,
        think_fmt_weight=args.think_fmt_weight,
        turn_loss_weights=tuple(
            float(x.strip()) for x in args.turn_loss_weights.split(",") if x.strip()
        ),
        num_generations=args.num_generations,
        num_turns=args.num_turns,
        completion_ratio=args.completion_ratio,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_max_model_len=args.vllm_max_model_len,
        epsilon=args.epsilon,
        kl_beta=args.kl_beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=not args.no_bf16,
        train_micro_batch_size=args.train_micro_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_optimizer=args.save_optimizer,
        seed=args.seed,
        debug=args.debug,
    )

    # ── Argument Validation ──────────────────────────────────────────────────────────────
    if cfg.num_turns < 2:
        raise ValueError(
            f"num_turns={cfg.num_turns} must be >= 2 "
            "(the critic quadrant reward requires at least two dialogue turns)."
        )
    if cfg.num_generations < 2:
        raise ValueError(
            f"num_generations={cfg.num_generations} must be >= 2 "
            "(GRPO group normalization needs at least two trajectories to compute std)."
        )
    if cfg.turn_loss_weights:
        if len(cfg.turn_loss_weights) != cfg.num_turns:
            raise ValueError(
                f"turn_loss_weights has {len(cfg.turn_loss_weights)} values, "
                f"but num_turns={cfg.num_turns}. They must match exactly."
            )
        if any(w < 0 for w in cfg.turn_loss_weights):
            raise ValueError("turn_loss_weights must be non-negative.")
        if sum(cfg.turn_loss_weights) <= 0:
            raise ValueError("turn_loss_weights must contain at least one positive value.")
    if cfg.soft_cache_length <= 0:
        raise ValueError(
            f"soft_cache_length={cfg.soft_cache_length} must be > 0 "
            "(used as the denominator of the linear decay for overlong penalty)."
        )
    if cfg.temperature <= 0:
        raise ValueError(
            f"temperature={cfg.temperature} must be > 0 "
            "(it is used as the divisor for logits scaling)."
        )
    if not (0 < cfg.completion_ratio <= 1.0):
        raise ValueError(
            f"completion_ratio={cfg.completion_ratio} must be in the range (0, 1]."
        )
    if cfg.save_total_limit < 0:
        raise ValueError(
            f"save_total_limit={cfg.save_total_limit} must be >= 0 "
            "(0 means no checkpoints will be saved)."
        )
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps} must be >= 1."
        )
    if cfg.max_seq_length > cfg.vllm_max_model_len:
        raise ValueError(
            f"max_seq_length={cfg.max_seq_length} > "
            f"vllm_max_model_len={cfg.vllm_max_model_len}. "
            "Training sequence length cannot exceed the vLLM context window."
        )

    return cfg
