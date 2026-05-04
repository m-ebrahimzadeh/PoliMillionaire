"""Config invariants. Tiny, but cheap."""
import dataclasses
import pytest

from polimibot import PATHS, RUNTIME, CATEGORIES, Category


def test_paths_resolve_to_repo_root():
    # The repo root must contain pyproject.toml; if not, _resolve_project_root
    # silently fell back to cwd which would be a regression.
    assert (PATHS.project_root / "pyproject.toml").is_file()


def test_runtime_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RUNTIME.api_min_delay_seconds = 0.0  # type: ignore[misc]


def test_categories_cover_the_four_known_competitions():
    assert {c.category for c in CATEGORIES.values()} == set(Category)