from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


@dataclass(frozen=True)
class WellData:
    well_id: str
    split: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame
    horizontal_path: Path
    typewell_path: Path


def well_id_from_path(path: Path) -> str:
    name = path.name
    if "__" not in name:
        raise ValueError(f"Cannot parse well id from {path}")
    return name.split("__", 1)[0]


def split_dir(data_dir: str | Path, split: str) -> Path:
    path = Path(data_dir) / split
    if not path.exists():
        raise FileNotFoundError(f"Missing split directory: {path}")
    return path


def list_well_ids(data_dir: str | Path, split: str) -> list[str]:
    directory = split_dir(data_dir, split)
    ids = [well_id_from_path(path) for path in directory.glob(f"*{HORIZONTAL_SUFFIX}")]
    return sorted(ids)


def get_test_well_ids(data_dir: str | Path) -> list[str]:
    test_path = Path(data_dir) / "test"
    if not test_path.exists():
        return []
    return list_well_ids(data_dir, "test")


def read_horizontal(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["row_index"] = range(len(df))
    if "MD" in df.columns:
        df = df.sort_values("MD", kind="mergesort").reset_index(drop=True)
    return df


def read_typewell(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TVT" not in df.columns or "GR" not in df.columns:
        raise ValueError(f"Typewell file must contain TVT and GR columns: {path}")
    return df.sort_values("TVT", kind="mergesort").reset_index(drop=True)


def load_well(data_dir: str | Path, split: str, well_id: str) -> WellData:
    directory = split_dir(data_dir, split)
    horizontal_path = directory / f"{well_id}{HORIZONTAL_SUFFIX}"
    typewell_path = directory / f"{well_id}{TYPEWELL_SUFFIX}"
    if not horizontal_path.exists():
        raise FileNotFoundError(horizontal_path)
    if not typewell_path.exists():
        raise FileNotFoundError(typewell_path)
    return WellData(
        well_id=well_id,
        split=split,
        horizontal=read_horizontal(horizontal_path),
        typewell=read_typewell(typewell_path),
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
    )


def load_wells(
    data_dir: str | Path,
    split: str,
    well_ids: Iterable[str] | None = None,
) -> list[WellData]:
    ids = list(well_ids) if well_ids is not None else list_well_ids(data_dir, split)
    return [load_well(data_dir, split, well_id) for well_id in ids]


def read_submission_template(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "sample_submission.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    sub = pd.read_csv(path)
    required = {"id", "tvt"}
    missing = required.difference(sub.columns)
    if missing:
        raise ValueError(f"sample_submission.csv missing columns: {sorted(missing)}")
    return sub
