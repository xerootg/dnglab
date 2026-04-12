"""Shared fixtures for dnglab-py integration tests.

These tests require RAW sample files downloaded by
``scripts/fetch_raw_samples.py``.  Run that script first::

    python scripts/fetch_raw_samples.py

The manifest at ``test_samples/manifest.json`` drives parametrisation:
each entry describes one RAW file with its make, model, and relative path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


_MANIFEST: list[dict] = _load_manifest()


def _sample_ids() -> list[str]:
    """Human-readable test IDs like 'Canon/EOS_5D'."""
    ids = []
    for entry in _MANIFEST:
        label = f"{entry['make']}/{entry['model']}"
        if entry.get("mode"):
            label += f"({entry['mode']})"
        ids.append(label)
    return ids


def _resolve_path(entry: dict) -> Path:
    return SAMPLES_DIR / entry["path"]


@pytest.fixture(params=_MANIFEST, ids=_sample_ids())
def raw_sample(request) -> dict:
    """Yield a manifest entry with an absolute ``abs_path`` key added."""
    entry = dict(request.param)
    entry["abs_path"] = _resolve_path(entry)
    if not entry["abs_path"].exists():
        pytest.skip(f"Sample file missing: {entry['abs_path']}")
    return entry


@pytest.fixture
def all_samples() -> list[dict]:
    """Return all manifest entries with absolute paths resolved."""
    if not _MANIFEST:
        pytest.skip("No manifest.json — run scripts/fetch_raw_samples.py first")
    result = []
    for entry in _MANIFEST:
        e = dict(entry)
        e["abs_path"] = _resolve_path(entry)
        if e["abs_path"].exists():
            result.append(e)
    if not result:
        pytest.skip("No sample files found on disk")
    return result


def pytest_collection_modifyitems(config, items):
    """Skip sample-dependent tests when no samples are available.

    Tests that don't use the ``raw_sample`` or ``all_samples`` fixtures
    (e.g. ``test_support.py`` unit tests) are left untouched.
    """
    if _MANIFEST:
        return
    skip = pytest.mark.skip(reason="No test samples — run scripts/fetch_raw_samples.py first")
    sample_fixtures = {"raw_sample", "all_samples"}
    for item in items:
        if sample_fixtures & set(item.fixturenames):
            item.add_marker(skip)
