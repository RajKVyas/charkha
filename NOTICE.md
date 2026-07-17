# Third-Party Notices

CHARKHA includes or depends on the following third-party software.

## Runtime Dependencies

| Package | License | Usage |
|---|---|---|
| PyTorch | BSD-3-Clause | Deep learning framework |
| flash-linear-attention (fla) | Apache 2.0 | Fused Triton kernels for Gated DeltaNet (GDN-1, GDN-2) |
| Triton | MIT | GPU kernel compiler used by fla |
| HuggingFace Transformers | Apache 2.0 | Tokenizer loading for evaluation |
| HuggingFace Datasets | Apache 2.0 | Streaming data loader |
| bitsandbytes | MIT | 8-bit optimizer states |
| tokenizers | Apache 2.0 | BPE tokenizer training and loading |
| safetensors | Apache 2.0 | Safe tensor serialization |
| numpy | BSD-3-Clause | Numerical computing |
| pyyaml | MIT | Configuration parsing |

## Research and Architecture Attribution

CHARKHA's architecture draws from published research. Key influences:

- **Gated DeltaNet (GDN-2)**: arXiv:2605.22791 — channel-wise erase+write gates
- **Gated DeltaNet (GDN-1)**: Yang et al., "Gated Delta Networks" — implemented via flash-linear-attention
- **Huginn**: arXiv:2502.05171 — depth-recurrent core with test-time effort dial
- **PonderNet**: arXiv:2107.05407 — adaptive computation with learned halting
- **OSDN**: arXiv:2605.13473 — key preconditioning for DeltaNet
- **Muon optimizer**: Keller Jordan et al., modded-nanogpt / Kimi K2
- **GQA**: arXiv:2305.13245 — grouped-query attention
- **RoPE**: arXiv:2104.09864 — rotary position embeddings
- **DeepSeek-V3**: Multi-token prediction auxiliary heads
- **Two-Scale Latent Acceleration**: arXiv:2509.23314 — early-exit convergence
- **Laplace-Redux**: arXiv:2106.14806 — post-hoc uncertainty
- **RLCM**: arXiv:2604.23333 — margin-based confidence
- **NITP**: arXiv:2605.24956 — next-implicit-token prediction
- **SNGP**: arXiv:2006.10108 — distance-aware epistemic uncertainty
- **Conformal Prediction**: Angelopoulos & Bates (arXiv:2107.07511) for uncertainty calibration

CHARKHA's custom Sutra-131k tokenizer uses the HuggingFace `tokenizers` library
(Rust BPE implementation, Apache 2.0).

The GDN Triton kernels are imported at runtime from the `flash-linear-attention`
package; they are not vendored in this repository.

## Training Data

Training data sources and their licenses are documented in `configs/sources.yaml`.
Data shards are not distributed with this repository.
