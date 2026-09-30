# Contributing

Issues and pull requests are welcome.

- This repository accompanies a paper under review; the default branch is a
  pinned release. Open an issue before larger changes.
- Never commit API keys, tokens, or cluster-internal endpoints — credentials
  belong in environment variables (see `docs/ENVIRONMENT.md`).
- Datasets, checkpoints, and generated videos stay outside the repository.
- Paper result tables live with the paper and the project page, not in this
  branch.

## Development setup

```bash
git clone https://github.com/open-gigaai/giga-world-0 third_party/giga-world-0
git clone https://github.com/open-gigaai/giga-models  third_party/giga-models
conda env create -f envs/gigaworld.yaml && conda activate gigaworld
pip install -e ".[train,eval]" && pip install -e third_party/giga-models
pip install pre-commit && pre-commit install
```

The hooks (isort / black / flake8 / mdformat / docformatter) are configured in
`.pre-commit-config.yaml`; run `pre-commit run --all-files` before submitting.

## Reporting issues

Include the command you ran, the environment (`pip list`, GPU/CUDA), and the
full error trace. For benchmark discrepancies, state the exact evaluation
protocol (detector settings, seeds, CFG weight) you used.
