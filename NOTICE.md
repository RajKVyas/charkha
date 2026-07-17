# Third-Party Notices

CHARKHA's source code is MIT-licensed except where a file says otherwise. The
following notice covers code adapted into the repository. Packages installed
as dependencies remain under their own licenses and are not vendored here.

## Adapted Muon code

`src/charkha/_optim.py` contains an adapted Newton–Schulz orthogonalization
helper based on [Keller Jordan's Muon reference
implementation](https://github.com/KellerJordan/Muon).

MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Optional runtime kernels

CHARKHA can import Gated DeltaNet kernels from
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention),
which is MIT-licensed (copyright 2023–2026 Songlin Yang, Yu Zhang, and Zhiyuan
Li). Those kernels are installed separately and are not copied into this
repository.

The GatedDeltaNet-2 architecture is described by the
[paper](https://arxiv.org/abs/2605.22791) and the
[NVIDIA reference repository](https://github.com/NVlabs/GatedDeltaNet-2).
NVIDIA's repository uses the NVIDIA Source Code License-NC. CHARKHA does not
include or import that repository's source or kernels.

## Other dependencies

Direct dependencies and exact versions are declared in `pyproject.toml` and
`requirements.txt`. Their principal licenses include BSD-3-Clause (PyTorch,
NumPy, SciPy, pandas, scikit-learn), MIT (Triton, flash-linear-attention,
fla-core, PyYAML), and Apache-2.0 (Hugging Face Transformers, Datasets, Hub,
Tokenizers, Safetensors, Evaluate, and NLTK). Dependency distributions carry
their own complete notices.

## Research attribution

The implementation draws on published methods; citations identify ideas, not
copied source code.

- Gated DeltaNet: arXiv:2412.06464 and arXiv:2605.22791
- Huginn / depth recurrence: arXiv:2502.05171
- PonderNet: arXiv:2107.05407
- OSDN: arXiv:2605.13473
- Muon: Keller Jordan et al.; Kimi K2 technical report
- Grouped-query attention: arXiv:2305.13245
- RoPE: arXiv:2104.09864
- Multi-token prediction: DeepSeek-V3 technical report
- Two-scale latent acceleration: arXiv:2509.23314
- Laplace Redux: arXiv:2106.14806
- RLCM: arXiv:2604.23333
- NITP: arXiv:2605.24956
- SNGP: arXiv:2006.10108
- Conformal prediction: arXiv:2107.07511
- Model Soups: arXiv:2203.05482
- TIES-Merging: arXiv:2306.01708

## Data and generated artifacts

No training-data shards are distributed. `configs/sources.example.yaml` is an
illustrative user configuration, not a record of the corpus used for any
released artifact.

`charkha_tokenizer.json` is a generated BPE vocabulary and merge table created
with Hugging Face Tokenizers. It contains no raw training documents. Its exact
training-corpus manifest is not included in this release.
