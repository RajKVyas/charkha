# Contributing to CHARKHA

Thanks for your interest in contributing.

## Scope

CHARKHA is a research codebase for training and serving a sub-billion-parameter
language model on consumer hardware. Contributions that improve correctness,
performance, documentation, or test coverage are welcome.

## Before You Start

- Read the [README](README.md) to understand the architecture and goals.
- Run `python tests/run_tests.py` to verify the environment works.
- Open an issue to discuss significant changes before investing time.

## Development Flow

1. Fork the repository and create a feature branch.
2. Make focused, testable changes.
3. Run the fast test suite: `python tests/run_tests.py`
4. Run the full proof bundle: `python scripts/proof_bundle.py`
5. Ensure `make check` and `make proof` pass.
6. Submit a pull request with a clear description of the change and its motivation.

## Code Style

- Follow existing patterns in the codebase.
- Use Python 3.10+ features (match statements, union types).
- Keep the 8GB local train/serve path working.
- Add selftests for new features (`--selftest` flag pattern).
- The `--profile frontier` stack is the default; experimental features
  should be opt-in flags, not defaults.

## Testing

```bash
python tests/run_tests.py       # fast regression suite
python src/preflight.py         # full local gate
python scripts/proof_bundle.py  # CPU-only proof (pre-push hook)
```

## Report a Bug

Use GitHub Issues. Include:
- OS, Python version, PyTorch version, CUDA version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Any relevant logs or error messages

## Security

Report security vulnerabilities privately. Do not open a public issue.
Contact a maintainer directly.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see [LICENSE](LICENSE)).
