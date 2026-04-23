"""
ReCrit training hyperparameter configuration.
All command-line arguments are parsed by argparse and packed into ReCritConfig.
"""

import argparse
from dataclasses import dataclass


@dataclass
class ReCritConfig:
    # -- Model -----------------------------------------------------------------
    model_path: str = ""

    # -- Dataset ---------------------------------------------------------------
    train_dataset: str = ""
    judge_mode: str = "both"
    #   close - treat every sample as close-form evaluation
    #   open  - treat every sample as open-form evaluation
    #   both  - read judge_mode from each sample; the dataset must provide it
    add_format_prompt: bool = False
    # Whether to append a mode-specific format prompt to the question.

    # -- Judge model -----------------------------------------------------------
    judge_model: str = "gemini-3-flash-preview-nothinking"

    # -- Critic prompt mode ----------------------------------------------------
    critic_prompt_mode: str = "mixed"
    #   mixed      - randomly sample from the three attitude templates
    #   eval_fixed - use a single fixed critic prompt aligned with evaluation
    critic_prompt_text: str = (
        "Can you verify your reasoning? I want to make sure nothing was overlooked."
    )

    # -- Main four-quadrant reward weights ------------------------------------
    w_correction: float = 1.0
    w_robustness: float = 0.6
    w_sycophancy: float = 1.0
    w_boundary: float = 0.1

    # -- Auxiliary reward parameters ------------------------------------------
    repetition_n_grams: int = 8
    soft_max_length: int = 4096
    soft_cache_length: int = 1024
    repetition_weight: float = 0.2
    overlong_weight: float = 0.2
    think_fmt_weight: float = 0.2
    turn_loss_weights: tuple[float, ...] = ()
    # If non-empty, this must match num_turns and provides per-turn RL loss weights.

    # -- Rollout ---------------------------------------------------------------
    num_generations: int = 8
    num_turns: int = 2
    completion_ratio: float = 0.75
    # Stop submitting new requests once this fraction of samples reaches the
    # maximum turn count and all samples have passed the early-stop floor.
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    system_prompt: str = "You are a helpful assistant."

    # -- vLLM ------------------------------------------------------------------
    vllm_gpu_memory_utilization: float = 0.22
    vllm_enforce_eager: bool = False
    vllm_max_model_len: int = 65536

    # -- GRPO algorithm --------------------------------------------------------
    epsilon: float = 0.2
    kl_beta: float = 0.04
    # Reverse-KL anchor to the initial SFT policy. Set 0 to disable it.

    # -- Training hyperparameters ---------------------------------------------
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-6
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    max_seq_length: int = 8192
    # If the full concatenated sequence exceeds this limit, the final turn is
    # truncated first; if earlier context is already too long, the sample is dropped.
    gradient_accumulation_steps: int = 4
    # Effective batch size grows by this factor without increasing peak memory.

    # -- Training efficiency ---------------------------------------------------
    gradient_checkpointing: bool = True
    bf16: bool = True
    train_micro_batch_size: int = 2

    # -- Output and logging ----------------------------------------------------
    output_dir: str = "output/recrit"
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    save_optimizer: bool = False

    # -- Debug -----------------------------------------------------------------
    debug: bool = False

    # -- Seed ------------------------------------------------------------------
    seed: int = 42


def parse_args() -> ReCritConfig:
    parser = argparse.ArgumentParser(
        description="ReCrit: Multi-Turn Critic GRPO Training"
    )

    # Required arguments
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--output_dir", default="output/recrit")

    # Optional arguments matching config fields
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--judge_mode",
        default="both",
        choices=["close", "open", "both"],
        help="Evaluation mode: close/open/both (for both, the dataset must include judge_mode)",
    )
    parser.add_argument(
        "--add_format_prompt",
        action="store_true",
        help="Append a format-requirements prompt to the end of each question (different templates for close/open)",
    )

    parser.add_argument(
        "--judge_model",
        default="gemini-3-flash-preview-nothinking",
        help="LLM model used by structai Judge",
    )
    parser.add_argument(
        "--critic_prompt_mode",
        default="mixed",
        choices=["mixed", "eval_fixed"],
        help="Critic prompt mode: mixed keeps the current attitude templates; "
        "eval_fixed uses a single fixed critic prompt.",
    )
    parser.add_argument(
        "--critic_prompt_text",
        default="Can you verify your reasoning? I want to make sure nothing was overlooked.",
        help="Fixed critic prompt text used when --critic_prompt_mode eval_fixed.",
    )
    parser.add_argument("--w_correction", type=float, default=1.0)
    parser.add_argument("--w_robustness", type=float, default=0.6)
    parser.add_argument("--w_sycophancy", type=float, default=1.0)
    parser.add_argument("--w_boundary", type=float, default=0.1)

    parser.add_argument("--repetition_n_grams", type=int, default=8)
    parser.add_argument("--soft_max_length", type=int, default=4096)
    parser.add_argument("--soft_cache_length", type=int, default=1024)
    parser.add_argument(
        "--repetition_weight",
        type=float,
        default=0.2,
        help="Scaling weight for repetition_penalty",
    )
    parser.add_argument(
        "--overlong_weight",
        type=float,
        default=0.2,
        help="Scaling weight for overlong_penalty",
    )
    parser.add_argument(
        "--think_fmt_weight",
        type=float,
        default=0.2,
        help="Scaling weight for think_format_reward",
    )
    parser.add_argument(
        "--turn_loss_weights",
        default="",
        help="Optional comma-separated per-turn loss weights applied to assistant tokens only. "
        "Leave empty to keep the default equal-weight behavior. "
        "Example: 0.0,0.3,1.0",
    )

    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument(
        "--completion_ratio",
        type=float,
        default=0.75,
        help="Dynamic max-turn rollout early-stop threshold (1.0 means fixed max turns)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.22)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_max_model_len", type=int, default=65536)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.04,
        help="KL penalty coefficient beta to anchor the policy near the initial SFT model (0 = disabled)",
    )

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size = per_device x num_gen x accum_steps x world_size)",
    )

    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--train_micro_batch_size", type=int, default=2)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        help="Save optimizer/scheduler state (about 70GB+, for resume training)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit diagnostic logs for DDP data sharding, parameter sync, and related checks",
    )

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
