#!/usr/bin/env python3
"""Report which parts of the EVEWorld environment are usable.

The script probes four things and prints one table per group:

* the interpreter version and the core packages that ``pip install -e .``
  brings in,
* the optional training and evaluation stacks, which are only needed for the
  runs that import them,
* the environment variables listed in ``.env.example`` (values of API keys are
  never echoed),
* the third-party checkouts and the data roots under ``GAGI_ROOT``.

Only a missing core item makes the script exit non-zero; missing optional items
are reported and otherwise ignored unless ``--strict`` is given. ``--json``
prints the same information as a machine-readable summary.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

MINIMUM_PYTHON = (3, 10)

CORE_PACKAGES = (
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("omegaconf", "omegaconf"),
    ("einops", "einops"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
)

TRAIN_PACKAGES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("datasets", "datasets"),
    ("deepspeed", "deepspeed"),
    ("fairscale", "fairscale"),
)

EVALUATION_PACKAGES = (
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("skvideo", "scikit-video"),
    ("lpips", "lpips"),
    ("segment_anything", "segment-anything"),
    ("timm", "timm"),
    ("av", "av"),
    ("imageio", "imageio"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("openai", "openai"),
    ("google.genai", "google-genai"),
)

CHECKOUTS = (
    ("third_party/giga-world-0", "GigaWorld-0 backbone", "third_party/README.md"),
    ("third_party/giga-models", "GigaWorld-0 framework", "third_party/README.md"),
    ("third_party/FlowWAM", "FlowWAM backbone", "third_party/README.md"),
)

# (variable, label, default when the variable is unset)
DATA_ROOTS = (
    ("GAGI_ROOT", "data root", "$HOME/gagi"),
    ("GW0_MODEL_DIR", "GigaWorld-0 weights", "$GAGI_ROOT/giga_world_0_video_pretrain"),
    ("GR1_DATA_ROOT", "GR1 fine-tuning split", "$GAGI_ROOT/gr1_finetune_data"),
    ("EVE_VIDEO_ROOT", "GigaWorld-0 outputs", "$GAGI_ROOT/giga_world_0_outputs"),
    ("EVE_OUT", "EVEWorld outputs", "$GAGI_ROOT/eve_outputs"),
)

CHECKPOINT_FILES = (
    ("GDINO_PATH", "GroundingDINO weights"),
    ("SAM2_CHECKPOINT", "SAM2 occlusion checkpoint"),
)

# Used when .env.example is not present next to the checkout.
FALLBACK_ENV_KEYS = (
    "GAGI_ROOT",
    "GW0_MODEL_DIR",
    "GR1_DATA_ROOT",
    "EVE_VIDEO_ROOT",
    "EVE_OUT",
    "PYBIN",
    "CONDA_ENV",
    "MLR_CONDA_ENV",
    "TRAIN_VENV",
    "EVAL_CONDA_ENV",
    "GEN_CONDA_ENV",
    "GDINO_PATH",
    "SAM2_CHECKPOINT",
    "OPENAI_API_KEY",
    "DIFROST_API_TOKEN",
    "DIFROST_GENAI_BASE_URL",
    "DIFROST_HOST",
    "DIFROST_MODEL",
)

SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_KEY", "_HOST")

ENV_KEY_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the EVEWorld environment is ready to run.",
    )
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--strict",
        action="store_true",
        help="also exit 1 when an optional package, variable or checkout is missing",
    )
    runtime.add_argument(
        "--json",
        action="store_true",
        help="print a JSON summary instead of the aligned tables",
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the checks that would run, without inspecting the environment",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    """Repository root, taken from the tree that holds this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("cannot locate the repository root: no pyproject.toml above this file")


def env_var_keys(root: Path) -> list[str]:
    """Keys listed in ``.env.example``, falling back to the built-in list.

    Most lines in the template are commented out with their default, so the
    leading ``#`` is stripped before the key is read.
    """
    example = root / ".env.example"
    keys: list[str] = []
    if example.is_file():
        for line in example.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lstrip("#").strip()
            match = ENV_KEY_PATTERN.match(stripped)
            if match:
                keys.append(match.group(1))
    return keys or list(FALLBACK_ENV_KEYS)


def expand(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value)))


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def describe_module(module: str, distribution: str) -> tuple[str, str]:
    """Availability and version of one importable module."""
    if not module_available(module):
        return "missing", "-"
    version = installed_version(distribution)
    return "ok", version or "unknown"


def render_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    formatted = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)).rstrip()]
    formatted.append("  ".join("-" * width for width in widths).rstrip())
    for row in rows:
        formatted.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(formatted)


def package_record(row: tuple[str, ...]) -> dict[str, str]:
    keys = ("module", "status", "version", "distribution", "note")
    return dict(zip(keys, row))


def check_python() -> tuple[list[tuple[str, ...]], list[str]]:
    version = platform.python_version()
    ok = sys.version_info >= MINIMUM_PYTHON
    required = ".".join(str(part) for part in MINIMUM_PYTHON)
    status = "ok" if ok else "too old"
    rows = [("interpreter", status, version, f">= {required}", sys.executable)]
    failures = [] if ok else [f"python {version} is older than {required}"]
    return rows, failures


def check_packages(
    group: str,
    packages: tuple[tuple[str, str], ...],
    required: bool,
) -> tuple[list[tuple[str, ...]], list[str]]:
    rows: list[tuple[str, ...]] = []
    failures: list[str] = []
    for module, distribution in packages:
        status, version = describe_module(module, distribution)
        rows.append((f"{group}: {module}", status, version, distribution, "" if required else "optional"))
        if status == "missing" and required:
            failures.append(f"required package {distribution} ({module}) is not importable")
    return rows, failures


def check_environment_variables(root: Path) -> tuple[list[tuple[str, ...]], list[str]]:
    rows: list[tuple[str, ...]] = []
    missing: list[str] = []
    for key in env_var_keys(root):
        value = os.environ.get(key, "")
        if not value:
            rows.append((key, "unset", "", "export it in .env when the run needs it"))
            missing.append(key)
            continue
        masked = key.endswith(SECRET_SUFFIXES)
        shown = "<set>" if masked else value
        detail = ""
        if not masked:
            path = expand(value)
            if path.exists():
                detail = "path exists" if value.startswith("/") else "value set"
            else:
                detail = "path not found"
        rows.append((key, "set", shown, detail))
    return rows, missing


def check_paths(root: Path) -> tuple[list[tuple[str, ...]], list[str]]:
    rows: list[tuple[str, ...]] = []
    missing: list[str] = []
    for relative, label, hint in CHECKOUTS:
        if (root / relative).is_dir():
            rows.append((label, "ok", relative, ""))
        else:
            rows.append((label, "missing", relative, f"clone it, see {hint}"))
            missing.append(label)
    for key, label, default in DATA_ROOTS:
        raw = os.environ.get(key, "")
        path = expand(raw or default)
        if path.is_dir():
            rows.append((label, "ok", str(path), "default" if not raw else ""))
        else:
            rows.append((label, "missing", str(path), f"set {key} in .env when the run needs it"))
            missing.append(label)
    for key, label in CHECKPOINT_FILES:
        value = os.environ.get(key, "")
        if not value:
            rows.append((label, "unset", key, ""))
            continue
        exists = expand(value).exists()
        rows.append((label, "ok" if exists else "missing", key, "path exists" if exists else "path not found"))
    return rows, missing


def check_library() -> tuple[list[tuple[str, ...]], list[str]]:
    if module_available("eveworld"):
        version = installed_version("eveworld") or "not installed (on PYTHONPATH)"
        return [("eveworld", "ok", version, "package under test", "")], []
    return [("eveworld", "missing", "-", "run pip install -e .", "required")], ["the eveworld package is not importable"]


def print_plan(root: Path, args: argparse.Namespace) -> None:
    print(f"repository root: {root}")
    print(f"interpreter:     {sys.executable} (need >= {'.'.join(str(p) for p in MINIMUM_PYTHON)})")
    print(f"strict:          {args.strict}")
    print()
    print(f"core packages:   {', '.join(distribution for _, distribution in CORE_PACKAGES)}")
    print(f"train packages:  {', '.join(distribution for _, distribution in TRAIN_PACKAGES)}")
    print(f"eval packages:   {', '.join(distribution for _, distribution in EVALUATION_PACKAGES)}")
    print()
    print("environment variables parsed from .env.example:")
    keys = env_var_keys(root)
    print(f"  {len(keys)} keys: {', '.join(keys)}")
    print()
    print("checkouts and data roots:")
    for relative, label, _ in CHECKOUTS:
        print(f"  {label}: {root / relative}")
    for key, label, default in DATA_ROOTS:
        print(f"  {label}: {expand(os.environ.get(key, '') or default)}")
    print()
    print("no probing performed (--dry-run)")


def print_report(sections: dict[str, list[tuple[str, ...]]]) -> None:
    for title, rows in sections.items():
        print(f"== {title}")
        if title == "packages":
            print(render_table(("package", "status", "version", "distribution", "note"), rows))
        elif title == "environment":
            print(render_table(("variable", "status", "value", "note"), rows))
        elif title == "checkouts":
            print(render_table(("item", "status", "location", "note"), rows))
        else:
            print(render_table(("item", "status", "version", "requirement", "note"), rows))
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()

    if args.dry_run:
        print_plan(root, args)
        return 0

    runtime_rows, failures = check_python()
    core_rows, core_failures = check_packages("core", CORE_PACKAGES, required=True)
    train_rows, train_failures = check_packages("train", TRAIN_PACKAGES, required=False)
    eval_rows, eval_failures = check_packages("eval", EVALUATION_PACKAGES, required=False)
    failures.extend(core_failures)

    env_rows, env_missing = check_environment_variables(root)
    path_rows, path_missing = check_paths(root)
    library_rows, library_failures = check_library()
    failures.extend(library_failures)

    optional_missing = train_failures + eval_failures + [f"{key} is unset" for key in env_missing]
    optional_missing += [f"{item} not found" for item in path_missing]
    if args.strict:
        failures.extend(optional_missing)

    if args.json:
        summary = {
            "repository_root": str(root),
            "python": platform.python_version(),
            "required_failures": failures,
            "optional_missing": optional_missing,
            "packages": {
                "core": [package_record(row) for row in core_rows],
                "train": [package_record(row) for row in train_rows],
                "eval": [package_record(row) for row in eval_rows],
            },
            "environment": [{"variable": row[0], "status": row[1], "note": row[3]} for row in env_rows],
            "checkouts": [{"item": row[0], "status": row[1], "location": row[2], "note": row[3]} for row in path_rows],
            "library": [{"item": row[0], "status": row[1], "version": row[2], "note": row[4]} for row in library_rows],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("EVEWorld environment report")
        print()
        print_report(
            {
                "interpreter": runtime_rows,
                "packages": core_rows + train_rows + eval_rows,
                "environment": env_rows,
                "checkouts": path_rows,
                "library": library_rows,
            }
        )
        if failures:
            print("required items missing:")
            for failure in failures:
                print(f"  - {failure}")
        elif args.strict and optional_missing:
            print("optional items missing (--strict):")
            for item in optional_missing:
                print(f"  - {item}")
        else:
            print("everything required for the offline recipes is in place.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
