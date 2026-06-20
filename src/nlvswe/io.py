"""Artifact read/write helpers with manifest sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from nlvswe.config import config_hash, get_config, project_root
from nlvswe.logging import get_logger
from nlvswe.repro import git_commit, run_manifest

logger = get_logger(__name__)

_PROJECT_ROOT = project_root()
_DATA_ROOT = _PROJECT_ROOT / "data"
_MODELS_ROOT = _PROJECT_ROOT / "models"
_FIGURES_ROOT = _PROJECT_ROOT / "reports" / "figures"


def _artifact_dir(subdir: str) -> Path:
    path = _DATA_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _table_paths(name: str, subdir: str) -> tuple[Path, Path]:
    base = _artifact_dir(subdir)
    return base / f"{name}.parquet", base / f"{name}.manifest.json"


def write_table(
    df: pd.DataFrame,
    name: str,
    subdir: str,
    *,
    sort_by: list[str] | str | None = None,
    plan: str = "01",
    schema_name: str | None = None,
) -> Path:
    """Write parquet + manifest sidecar. Sorts deterministically before write."""
    parquet_path, manifest_path = _table_paths(name, subdir)
    out = df.copy()
    if sort_by is not None:
        keys = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        out = out.sort_values(keys, kind="mergesort").reset_index(drop=True)

    cfg = get_config()
    manifest = run_manifest(
        plan,
        cfg.scope,
        row_count=len(out),
        schema_name=schema_name or name,
        config_hash=config_hash(cfg),
    )

    out.to_parquet(parquet_path, index=False, compression="snappy")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote table %s (%d rows) -> %s", name, len(out), parquet_path)
    return parquet_path


def read_table(name: str, subdir: str) -> pd.DataFrame:
    """Read parquet; warn if manifest git_commit differs from current."""
    parquet_path, manifest_path = _table_paths(name, subdir)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Table not found: {parquet_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_commit = manifest.get("git_commit")
    current_commit = git_commit()
    if stored_commit and stored_commit != current_commit:
        logger.warning(
            "Manifest git_commit (%s) differs from current (%s); artifact may be stale.",
            stored_commit,
            current_commit,
        )

    return pd.read_parquet(parquet_path)


def write_json(obj: Any, name: str, subdir: str, *, plan: str = "01") -> Path:
    """Write JSON artifact with manifest sidecar under data/<subdir>."""
    base = _artifact_dir(subdir)
    json_path = base / f"{name}.json"
    manifest_path = base / f"{name}.manifest.json"

    cfg = get_config()
    manifest = run_manifest(plan, cfg.scope, config_hash=config_hash(cfg))

    json_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def read_json(name: str, subdir: str) -> Any:
    """Read JSON artifact from data/<subdir>."""
    json_path = _artifact_dir(subdir) / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_model(model: Any, name: str) -> Path:
    """Serialize model to models/<name>.joblib."""
    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    path = _MODELS_ROOT / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved model -> %s", path)
    return path


def load_model(name: str) -> Any:
    """Load model from models/<name>.joblib."""
    path = _MODELS_ROOT / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def figure_path(name: str) -> Path:
    """Return path under reports/figures/ (without extension)."""
    _FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    return _FIGURES_ROOT / name


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save figure as PNG. (SVG generation disabled for now.)"""
    from nlvswe.plotting.theme import THEME

    base = figure_path(name)
    png_path = base.with_suffix(".png")
    save_kw = {"bbox_inches": "tight", "facecolor": THEME.bg, "edgecolor": THEME.bg}
    fig.savefig(png_path, dpi=150, **save_kw)
    logger.info("Saved figure -> %s", png_path)
    return png_path


def write_raw_artifact(
    content: bytes,
    dest_path: Path | str,
    *,
    source: str,
    url: str,
    license_note: str,
    plan: str = "02",
    params: dict[str, Any] | None = None,
    row_count: int | None = None,
    captured_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write raw bytes and a manifest sidecar (immutable acquisition contract)."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)

    cfg = get_config()
    manifest: dict[str, Any] = {
        **run_manifest(plan, cfg.scope, row_count=row_count, config_hash=config_hash(cfg)),
        "source": source,
        "url": url,
        "params": params or {},
        "license": license_note,
    }
    if captured_at is not None:
        manifest["captured_at"] = captured_at
    if extra:
        manifest.update(extra)

    manifest_path = dest_path.with_suffix(dest_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote raw artifact %s (%d bytes) -> %s", source, len(content), dest_path)
    return dest_path


def read_raw_manifest(dest_path: Path | str) -> dict[str, Any]:
    """Read manifest for a raw artifact path."""
    dest_path = Path(dest_path)
    manifest_path = dest_path.with_suffix(dest_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
