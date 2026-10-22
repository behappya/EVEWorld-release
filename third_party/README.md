# Third-Party Backbones

EVEWorld is a supervision recipe, not a backbone. The GigaWorld-0 backbone and
training framework stay upstream, and this repository only holds the glue: annotation and
corruption builders, correspondence adapters, and the training/evaluation entry
points under `eveworld/` and `benchmarks/`.

No checkout is tracked here — the root `.gitignore` drops
`third_party/giga-world-0/` and `third_party/giga-models/` — so clone them after checking out this repository:

```bash
git clone https://github.com/open-gigaai/giga-world-0 third_party/giga-world-0
git clone https://github.com/open-gigaai/giga-models  third_party/giga-models
pip install -e third_party/giga-models
```

## GigaWorld-0

- Upstream: <https://github.com/open-gigaai/giga-world-0>
- Role: the main backbone. DreamGenBench, WorldArena, EWMBench (AgiBot) and
  PBench all fine-tune and evaluate GigaWorld-0 Video-Pretrain-2B.
- Training also imports two upstream framework packages, which are not on PyPI:

  ```bash
  pip install git+https://github.com/open-gigaai/giga-train.git
  pip install git+https://github.com/open-gigaai/giga-datasets.git
  ```

- The checkpoint downloads and the isolated training venv come from the
  upstream `scripts/` directory; `docs/ENVIRONMENT.md` and `docs/DATASETS.md`
  point at the individual entry points.
- Apache-2.0; see the `LICENSE` inside the checkout.

## FlowWAM

- Upstream: <https://github.com/YixiangChen515/FlowWAM> (Wan2.2-TI2V-5B based)
- Role in the paper: the cross-backbone experiment on held-out RoboTwin episodes.
- The FlowWAM integration scripts and environment are not included in this code release.

## Other external tools

| Tool | Used for |
|---|---|
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | target detection for IGR annotation and for MLR counting |
| [SAM2](https://github.com/facebookresearch/sam2) | instance tracking and the robot-occluder masks behind the MLR occlusion check |
| [WorldArena](https://github.com/WorldArena-Official/WorldArena) | the WorldArena metric harness |
| [DreamGen](https://github.com/NVIDIA/DreamGen) | the DreamGen GR1 split and its judging protocol |
| [EWMBench](https://github.com/AgibotTech/EWMBench) | the AgiBot metric suite |

Their weights and config paths are read from the environment; copy
[`.env.example`](../.env.example) to `.env` and fill in what you have.
