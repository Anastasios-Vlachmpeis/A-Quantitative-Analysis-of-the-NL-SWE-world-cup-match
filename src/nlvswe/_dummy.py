"""Smoke test that config, parquet IO, and manifests actually work."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nlvswe.config import load_config
from nlvswe.io import read_table, write_table
from nlvswe.logging import get_logger
from nlvswe.repro import set_seeds

logger = get_logger("nlvswe._dummy")

PLAN = "01"
ARTIFACT_NAME = "dummy"
SUBDIR = "processed"


def build_dummy_frame(seed: int) -> pd.DataFrame:
    """Tiny deterministic DataFrame for round-trip testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "label": ["alpha", "beta", "gamma"],
            "value": [float(seed), float(seed + 1), float(seed + 2)],
        }
    )


def main() -> None:
    cfg = load_config()
    set_seeds(cfg.seed)
    logger.info("Dummy run (plan %s, scope=%s, seed=%d)", PLAN, cfg.scope, cfg.seed)

    df = build_dummy_frame(cfg.seed)
    write_table(df, ARTIFACT_NAME, SUBDIR, sort_by="id", plan=PLAN, schema_name="dummy")

    round_trip = read_table(ARTIFACT_NAME, SUBDIR)
    pd.testing.assert_frame_equal(round_trip, df.sort_values("id", kind="mergesort").reset_index(drop=True))

    manifest_path = Path("data") / SUBDIR / f"{ARTIFACT_NAME}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"created_at", "git_commit", "config_hash", "source_plan", "row_count", "schema_name", "scope"}
    missing = required - set(manifest.keys())
    if missing:
        raise RuntimeError(f"Manifest missing fields: {missing}")
    logger.info("Manifest: %s", manifest)
    logger.info("Dummy pipeline OK (%d rows)", len(round_trip))


if __name__ == "__main__":
    main()
