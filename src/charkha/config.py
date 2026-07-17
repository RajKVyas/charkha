"""CHARKHA configuration."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
@dataclass
class CharkhaConfig:
    vocab_size: int = 50304  # Comma BPE (50,257) padded to /128; toy mode: 256 bytes
    tokenizer_name: str = None  # HF hub name or local tokenizer.json path that produced the
    # training shards (dataprep.py's index.json 'tokenizer' field,
    # propagated by train.py's ShardLoader) — None for legacy
    # checkpoints predating this field. serve.py loads THIS, not a
    # hardcoded tokenizer, so inference always matches training.
    d_model: int = 1280
    embed_factor: int = 0  # >0: ALBERT-style factorized TIED embedding — codes (V, f) +
    # up-projection (f, d_model) replace the dense (V, d_model)
    # table. At V=131072/d=1280/f=256 this reclaims ~134M params
    # (~0.5GB fp32 weights + 0.5GB grads + optimizer momentum) —
    # the difference between the full model fitting an 8GB card
    # or not. The LM head stays tied: logits = (h @ up) @ codesᵀ.
    # 0 = classic dense embedding (existing checkpoints).
    n_heads: int = 20  # head_dim 64
    n_kv_heads: int = 5  # GQA 4:1
    d_ff: int = 3456  # SwiGLU ~2.7x
    n_prelude: int = 4
    n_core: int = 8  # the recurrent block
    n_coda: int = 4
    window: int = 1024  # sliding-window size for SWA layers
    rope_base: float = 10000.0
    max_seq_len: int = 4096
    # recurrence (the effort dial)
    use_recurrence: bool = True
    mean_recurrence: int = 4  # train-time E[r]  (Poisson(mean-1)+1)
    max_recurrence_train: int = 8
    backprop_depth: int = 2  # truncated BPTT through last k loops (fixed mode)
    recurrent_state_noise: float = 0.02  # train-time noise scale for the initial recurrent state.
    # Eval/inference use deterministic zero state.
    per_seq_recurrence: bool = False  # LoopWM (arXiv:2606.18208): sample loop-count T per SEQUENCE
    # inside the micro-batch, not one T for the whole batch. Only
    # differs when batch>1 (at batch=1 the scalar path already
    # samples per-sequence). Reduces objective variance / loss
    # spikes; uses grad+detach masking so it needs batch>1 VRAM.
    use_loop_embed: bool = True  # sinusoidal loop-index signal each recurrent pass so the shared
    # core block can specialize behavior by depth (Huginn;
    # arXiv:2502.05171). iteration 1 != iteration 8. ~0 params.
    # adaptive halting (PonderNet/ACT - the *learned* per-token effort dial)
    use_halting: bool = True
    max_recurrence_infer: int = 16  # cap on adaptive loops at inference
    halt_threshold: float = 0.9  # inference: stop once cumulative halt prob >= this
    halt_granularity: str = "token"  # token|segment|sequence. Segment/sequence pool halt probs
    # before ACT mass accounting for stabler effort budgets.
    halt_segment_len: int = 16  # token span for halt_granularity='segment'.
    ponder_cost: float = 0.01  # weight on KL(halt || geometric prior)
    # equilibrium effort ('converge'): inference-only mode that runs the trained depth, then
    # freezes the loop-index signal (stationary core map) and Anderson-accelerates to a fixed
    # point — "think until settled" with a principled stop, instead of a fixed loop count.
    converge_tol: float = 5e-3  # stop when relative state change per iteration < tol
    converge_max_iter: int = 32  # hard cap on equilibrium iterations after the trained ramp
    converge_anderson_m: int = 3  # Anderson acceleration history window (1 = plain iteration)
    # training
    z_loss: float = 0.0  # PaLM z-loss weight (~1e-4): penalize (log Z)^2 so the logit
    # scale stays bounded — bf16 stability + calibration win.
    logit_softcap: float = 0.0  # Gemma-2 final-logit softcap (~30): logits = cap*tanh(l/cap).
    # Applied in BOTH training CE and every inference head read
    # (train/serve consistency). 0 = off.
    mtp_weight: float = 0.2
    conf_weight: float = 0.1  # confidence/calibration head ("knows its limits")
    conf_margin: float = 0.1  # RLCM (arXiv:2604.23333): target gap conf(correct-prefix) -
    # conf(incorrect-prefix) for the margin loss (used by
    # selfteach/RLVR via conf_margin_loss; 0 inference cost).
    use_nitp: bool = False  # Next Implicit Token Prediction (arXiv:2605.24956): predict
    nitp_weight: float = 0.1  # the next token's *shallow* rep from the deep hidden (cosine).
    # ~2% train-time FLOPs, 0 inference cost (head unused at infer).
    use_deep_supervision: bool = False  # A3: ANYTIME depth-recurrence: deep-supervise the earlier
    deepsup_weight: float = 0.1  # gradient-carrying core loops through coda+head (weighted
    # toward later loops). Makes every effort level a valid answer.
    use_process_head: bool = False  # train recurrent states to predict final token correctness.
    process_weight: float = 0.05  # weight for process-head BCE over stored recurrent states.
    cross_loop_consistency: bool = False  # A1: aux loss pulling consecutive core states toward
    cross_loop_weight: float = 0.02  # directional agreement (maximize per-token cosine
    # (s_n, s_{n-1})). Makes recurrence contract.
    effective_depth_scale: bool = True  # F1 fix: compute residual scaling for effective depth
    # (n_prelude + r * n_core + n_coda) instead of static depth.
    # Prevents entropy pump at high recurrence (CE r4 > CE r1).
    use_bipolar_gate: bool = False  # bipolar sign-gating: STE forces GDN k,v to ±1 in forward
    # pass (discrete facts), differentiable in backward. 0 params.
    use_mtp_routing: bool = False  # MTP-routed macro-states: feed predicted future rep
    # from mtp_proj into core loops for multi-token reasoning
    use_thermostat: bool = False  # epistemic thermostat: override greedy halting when
    # confidence variance is high or trajectory oscillates
    thermostat_conf_threshold: float = 0.02  # confidence variance ceiling for forced continue
    use_subconscious: bool = False  # EXPERIMENTAL (off by default): a small, SEPARATE-parameter,
    # low-dim recurrent scratchpad gated into the main residual
    # stream. Across the recurrent core's reasoning loops it keeps a
    # low-dim latent (a GRU cell, independent of the d_model trunk),
    # reads a per-token summary of the evolving core state, and
    # writes a GATED low-rank contribution back. Write proj is zero-
    # init + gate starts closed => exact no-op at init (must EARN
    # influence). NOT weight-slice-sharing. Applied BETWEEN core
    # loops, OUTSIDE the checkpointed core pass, so no checkpoint-
    # recompute hazard. Wired into the fixed/sampled and halting
    # cores (not the per-seq-depth path). Ablate vs the equal-FLOP
    # control (Subconscious.force_gate_zero) to separate the
    # recurrence's influence from the added params/FLOPs.
    subconscious_dim: int = 32  # width of the subconscious scratchpad latent
    use_latent_memory: bool = False  # EXPERIMENTAL (off by default): intrinsic latent memory spine.
    # Short-term memory is a causal EMA over hidden states within
    # the sequence; long-term memory is a trainable prototype
    # codebook that self-categorizes those states. Aux losses make
    # categories predict future latent representations, so raw
    # token memorization is not enough to satisfy the objective.
    latent_memory_slots: int = 64  # long-term prototype categories
    latent_memory_dim: int = 0  # 0 = auto d_model//4 (min 16)
    latent_memory_decay: float = 0.85  # short-term EMA decay inside a sequence
    latent_memory_temp: float = 0.20  # assignment softmax temperature
    latent_memory_commit_weight: float = 0.02
    latent_memory_pred_weight: float = 0.05
    latent_memory_balance_weight: float = 0.01

    use_granary: bool = False  # STAGED: product-key memory layer — knowledge params
    # outside the trunk (host-RAM-scale value table, top-k
    # sparse read/write). Exact no-op at init (output gate 0),
    # activates function-preservingly at a grow point.
    # Pointwise per token => streaming-exact.
    # Micro-G1 (granary_micro.py): at iso-FLOP, 0.99 recall vs
    granary_slots: int = 2**20  # value slots (rounded up to a square); 2^20 @ d2048 ≈ 4GB fp16
    granary_d_key: int = 32  # per-head key dim (two halves of 16)
    granary_heads: int = 4  # independent lookups per token
    granary_topk: int = 32  # slots read per head (sparse-backward width)
    granary_knn: int = 32  # candidates per sub-key half before re-score
    use_loop_adapters: bool = False  # EXPERIMENTAL (off by default): per-loop low-rank adapters.
    # Each recurrence iteration n applies its own zero-init
    # rank-`loop_adapter_rank` delta to the core state, letting
    # weight-tied iterations specialize like distinct layers at
    # ~0.05% parameter cost — the direct answer to the
    # "weight-tying may prevent slab specialization" failure mode
    # of virtual-depth alignment (paper §virtualdepth). Zero-init
    # up-projection => exact no-op at init; wired inside
    # _core_step so every path (fixed/halting/converge/per-seq/
    # streaming decode) is covered identically.
    loop_adapter_rank: int = 8  # low-rank width per iteration
    loop_adapter_max: int = 16  # adapters allocated; loops beyond reuse the last one
    use_osdn: bool = False  # OSDN (arXiv:2605.13473): per-dimension KEY preconditioning for
    # the delta rule — k̃ = d⊙k. +H·head_dim params, clamped 0.25-4.
    use_gdn2: bool = True  # GDN-2 (arXiv:2605.22791, NVIDIA NC): replace the scalar
    # erase/write gate with channel-wise b_t (erase, key axis) and
    # w_t (write, value axis). Strict generalization of GDN/KDA.
    # +~0.5% params in this repo. Current implementation is exact
    # PyTorch recurrence; use preflight/mem probes to prove speed.
    track_convergence: bool = False  # at inference, expose a per-token convergence signal
    convergence_mode: str = (
        "extrapolation"  # 'extrapolation' (1st-order) or 'acceleration' (2nd diff)
    )
    use_accel_exit: bool = False  # two-scale-latent (arXiv:2509.23314): at inference, stop the
    accel_exit_threshold: float = 0.01  # adaptive-halting loop early once the core trajectory's 2nd
    # difference (curvature) falls below threshold — the answer has
    # settled, so extra loops are wasted. Pure geometry, 0 params,
    # inference-only (training still unrolls fully for the ponder
    # loss). ORs with the existing cumulative-halt-prob stop.
    convergence_window: int = 3  # signal = linear-EXTRAPOLATION error of the core trajectory
    # (predict s_n from a fit over the last `window` states; signal
    # = ||s_n - s_pred||). Small => answer settled => high confidence
    # / early-exit; large => keep thinking / lower conf / abstain.
    # NOT raw cosine displacement (arXiv:2606.05346 shows that
    # predicts the OPPOSITE direction). 0 params, ~0 infer cost.
    sngp_enabled: bool = False  # SNGP (arXiv:2006.10108): a distance-aware epistemic head run in
    sngp_rff_dim: int = 256  # PARALLEL with conf_head. Random Fourier Features (fixed random
    sngp_scale: float = 1.0  # projection of the final hidden) -> learnable GP mean trained on
    sngp_ridge: float = 1.0  # the same "is-top1-correct" target, plus a running precision
    sngp_spectral_norm: bool = (
        False  # matrix accumulated over RFF features. At inference the predictive
    )
    # VARIANCE diag(Φ Σ Φᵀ) grows for inputs far from the training
    # manifold (epistemic uncertainty conf_head's point estimate
    # misses). +rff_dim·d_model params (random, frozen) + a learnable
    # (rff_dim,) mean + an (rff_dim,rff_dim) precision buffer. 0 cost
    # when off. sngp_spectral_norm wraps the coda's linear maps in
    # spectral_norm so the feature map is distance-preserving (the
    # paper's bi-Lipschitz condition) — medium risk (alters training),
    # so it is independently gated from the GP head.
    sngp_accumulate_train: bool = (
        False  # precision accumulation is an explicit calibration action, not
    )
    # a default pretraining-side O(D^2) tax every batch. The SNGP
    # mean head still trains when sngp_enabled=True.
    laplace_enabled: bool = (
        False  # Laplace-Redux (arXiv:2106.14806): POST-HOC last-layer Laplace on
    )
    # conf_head -> (mean, var). No training change; fit on a small
    # calib set after training (see LaplaceConf). Flag only records
    # intent; fitting/prediction live in the standalone LaplaceConf.
    dropout: float = 0.0
    grad_checkpoint: bool = False  # wrap each recurrent-core step in torch.utils.checkpoint
    # (trades recompute for activation VRAM; needed at 0.4B/8GB)
    ce_chunk: int = 2048  # rows per chunk for fused cross-entropy; caps peak head
    gdn_chunk: int = 32  # chunk size for both Gated-DeltaNet chunkwise-parallel scans
    # (GDN-1 _gdn_chunk_scan, GDN-2 _gdn2_chunk_scan). Turns the O(T)
    # per-timestep recurrence into O(T/chunk) sequential steps of
    # batched matmuls + one unit-triangular solve per chunk — the
    # speed AND memory fix on the 8GB path. Keep ≤64: the in-chunk
    # solve grows ~C² in work/VRAM and can get ill-conditioned for
    # large C under strong decay. Identical math to the per-step loop.
    rev_bptt: bool = False  # reversible-recurrence BPTT for GDN-2 (train-time): backward
    # reconstructs chunk states by closed-form inversion instead of
    # storing them (see _RevGDN2Scan). O(anchors) state memory,
    # checkpoint-equivalent recompute cost. Weights untouched —
    # safe to enable/disable across resumes of the same checkpoint.
    rev_anchor: int = 8  # store an exact anchor state every K chunks (bounds the float
    # error the division-by-decay inversion accumulates).
    ce_vchunk: int = 0  # >0: stream the fused-CE VOCAB axis in chunks of this size
    # (CCE-style, arXiv:2411.09009): per row-chunk, the (rows, V)
    # logit block is never materialized — only (rows, ce_vchunk)
    # exists transiently per checkpointed logsumexp piece. Kills
    # the ~0.5GB backward logit spike at V=131072. 0 = off
    # (existing whole-vocab row-chunk path). Math identical.
    reversible: bool = False  # two-stream reversible residual coupling (RevNet-style) on the
    # block stacks: block-boundary activations are reconstructed by
    # inversion in backward, not stored. ARCH-CHANGING at the
    # function level (streams evolve differently than single-stream)
    # even though the parameter set is identical — flipping this on
    # an existing checkpoint perturbs the computed function (measure
    # with grow_init.py --to-reversible before committing a run).
    # ── Task-based RL (latent-space reinforcement learning) ──
    use_task_rl: bool = False  # enable task-based RL with advantage-weighted regression
    awr_temperature: float = 0.5  # lower = more selective (only strong successes)
    value_weight: float = 1.0  # weight on the value-head critic loss (MSE)
    task_replay_size: int = 1000  # capacity of the task-trajectory replay buffer

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, d):
        """Build a config from a checkpoint's cfg dict, IGNORING unknown keys — checkpoints
        saved before a feature was removed (e.g. error_feedback) must keep loading."""
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @staticmethod
    def toy():
        return CharkhaConfig(
            vocab_size=256,
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            d_ff=352,
            n_prelude=2,
            n_core=2,
            n_coda=2,
            window=64,
            max_seq_len=256,
            mean_recurrence=2,
            max_recurrence_train=4,
            max_recurrence_infer=6,
        )

    @staticmethod
    def small():
        # ~0.3B variant: d_model 1024 (head_dim 64 -> 16 heads, GQA 4:1), d_ff 2.75x, same depth.
        # Trains with real headroom on 8GB (bigger batch + lighter checkpointing -> higher tok/s).
        # At this scale tokens-seen dominates params, so a faster-cooking 0.3B can beat a 0.42B
        # that crawls - see the throughput math in the M1 notes.
        return CharkhaConfig(d_model=1024, n_heads=16, n_kv_heads=4, d_ff=2816)

    @staticmethod
    def nano():
        # ~25-35M ladder seed: d_model 512 doubles EXACTLY to small() (1024) and again to the
        # 0.97B width (2048) — the integer-only HyperCloning path (fractional
        # grows are banned after the 2026-07-06 1280->2048 divergence). head_dim 64 -> 8 heads.
        return CharkhaConfig(d_model=512, n_heads=8, n_kv_heads=4, d_ff=1408)

    @staticmethod
    def mini():
        # ~40-70M fluency proxy (d_model 640 = main/2, heads 10, same depth/recurrence family).
        # Trains at several-thousand tok/s on the 8GB card; two uses:
        #   1. grow-init (scripts/grow_init.py): HyperCloning-style width expansion into the main
        #      model's init — the big run skips learning English mechanics (~2-4x token savings);
        #   2. frozen residual logit prior (train.py --logit-prior): the big model trains on
        #      logits_big + w*logits_mini with w annealed to 0, spending gradient only on what
        #      the fluency prior cannot already predict.
        return CharkhaConfig(d_model=640, n_heads=10, n_kv_heads=5, d_ff=1728)

    @staticmethod
    def medium():
        # ~0.55B variant: d_model 1536 (head_dim 64 -> 24 heads, GQA 4:1), d_ff 2.7x, same depth.
        # Sized to TRAIN on a 12-20GB card (3060 12GB / 7900XT 20GB) with grad-checkpoint + Muon
        # offload — NOT the 8GB training target. INFERENCE still fits 8GB comfortably (~1.1GB weights
        # in bf16; far less quantized), so you can train bigger on a bigger card and still serve on the
        # 8GB box. Verify the exact VRAM on your card with `train.py --medium --mem-probe`.
        return CharkhaConfig(d_model=1536, n_heads=24, n_kv_heads=6, d_ff=4096)


# --------------------------------------------------------------------------
# Common modules
# --------------------------------------------------------------------------
