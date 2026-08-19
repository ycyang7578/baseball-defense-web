"""
RE24 run expectancy table and ΔRE values, computed empirically from statcast.

ΔRE uses "empirical runner advancement": for each hit, the actual base-out state after the hit is inferred
from the base state of the *next* plate appearance, giving ΔRE_emp = runs_scored + RE(observed new state) -
RE(old state). Situations with sample size n < min_n fall back to the deterministic theoretical advancement
value. This method matches empirical_delta_re from the thesis (Model_3).

Usage:
  from src.re24 import build_re24_table, build_delta_re, save_re24, load_re24

  re24  = build_re24_table([2021, 2022, 2023, 2024])
  delta = build_delta_re(re24, [2021, 2022, 2023, 2024])   # the empirical method needs the season years
  save_re24(re24, delta, Path("data/precomputed"))

  re24, delta = load_re24(Path("data/precomputed"))
"""
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psycopg2

from .config import DSN

# Base-out state key: (on_1b, on_2b, on_3b, outs_when_up), each base is 0/1, outs is 0-2
BaseOutState = tuple[int, int, int, int]
# ΔRE key: (hit_type, on_1b, on_2b, on_3b, outs_when_up)
HitDeltaKey = tuple[str, int, int, int, int]

# Only include plate appearances with a non-null events value (the last pitch of each PA), regular season only
_QUERY = """
    SELECT
        game_pk, inning, inning_topbot, at_bat_number, events,
        (on_1b IS NOT NULL)::int   AS on_1b,
        (on_2b IS NOT NULL)::int   AS on_2b,
        (on_3b IS NOT NULL)::int   AS on_3b,
        outs_when_up,
        bat_score, post_bat_score,
        post_bat_score - bat_score AS runs_in_pa
    FROM statcast
    WHERE game_year = ANY(%(years)s)
      AND game_type = 'R'
      AND events IS NOT NULL
      AND bat_score IS NOT NULL
      AND post_bat_score IS NOT NULL
"""


def _load_pa_data(years: list[int], dsn: str = DSN) -> pd.DataFrame:
    """Loads plate-appearance-level data (last pitch of each PA), including base state and score."""
    with psycopg2.connect(dsn) as conn:
        df = pd.read_sql(_QUERY, conn, params={"years": years})
    return df


def build_re24_table(years: list[int], dsn: str = DSN) -> dict[BaseOutState, float]:
    """
    Compute empirical RE24 table from statcast data.

    Returns:
        dict: (on_1b, on_2b, on_3b, outs_when_up) -> expected runs to end of half-inning
              Keys are tuples of four ints (each 0 or 1, last is 0/1/2).
    """
    df = _load_pa_data(years, dsn)

    # Within each half-inning, runs_rest = total runs scored from this PA to the end of the half-inning (inclusive)
    grp = ["game_pk", "inning", "inning_topbot"]
    df = df.sort_values(grp + ["at_bat_number"])
    df["runs_rest"] = (
        df.groupby(grp)["runs_in_pa"]
          .transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    )

    state_cols = ["on_1b", "on_2b", "on_3b", "outs_when_up"]
    table_raw = df.groupby(state_cols)["runs_rest"].mean().to_dict()

    return {
        (int(k[0]), int(k[1]), int(k[2]), int(k[3])): float(v)
        for k, v in table_raw.items()
    }


# Deterministic runner-advancement model (theoretical fallback when the empirical sample is insufficient)
# Each advance function returns (new_on_1b, new_on_2b, new_on_3b, runs_scored)
_ADVANCE: dict[str, Callable[[int, int, int], tuple[int, int, int, int]]] = {
    "1B": lambda b1, b2, b3: (1,  b1, b2,          b3         ),
    "2B": lambda b1, b2, b3: (0,  1,  b1,           b2 + b3   ),
    "3B": lambda b1, b2, b3: (0,  0,  1,             b1+b2+b3 ),
}


def build_delta_re_deterministic(re24: dict[BaseOutState, float]) -> dict[HitDeltaKey, float]:
    """
    Compute ΔRE(k, on_1b, on_2b, on_3b, outs) using deterministic runner advancement.
    Runners advance by a fixed rule (theoretical value), used as a fallback when the empirical sample is insufficient.
    """
    delta: dict[HitDeltaKey, float] = {}
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                for outs in (0, 1, 2):
                    old_re = re24.get((b1, b2, b3, outs), 0.0)
                    for hit_type, advance in _ADVANCE.items():
                        n1, n2, n3, runs = advance(b1, b2, b3)
                        new_re = re24.get((n1, n2, n3, outs), 0.0)
                        delta[(hit_type, b1, b2, b3, outs)] = runs + new_re - old_re
    return delta


_HIT_MAP: dict[str, str] = {"single": "1B", "double": "2B", "triple": "3B"}


def build_delta_re(
    re24: dict[BaseOutState, float],
    years: list[int],
    dsn: str = DSN,
    min_n: int = 30,
) -> dict[HitDeltaKey, float]:
    """
    Compute ΔRE(k, on_1b, on_2b, on_3b, outs) using *empirical* runner advancement.

    The actual base state after each hit is inferred from the base state of the "next plate appearance":
        ΔRE_emp = runs_scored + RE(observed new state) - RE(old state)
    When a group's sample size n >= min_n, use the empirical average; otherwise fall back to the deterministic
    theoretical value.

    Returns:
        dict: (hit_type, on_1b, on_2b, on_3b, outs) -> delta_run_expectancy
    """
    theory = build_delta_re_deterministic(re24)

    df = _load_pa_data(years, dsn)
    grp = ["game_pk", "inning", "inning_topbot"]
    df = df.sort_values(grp + ["at_bat_number"]).reset_index(drop=True)

    # Next-PA base state = base state after this PA's hit (only valid within the same half-inning)
    for c in ("on_1b", "on_2b", "on_3b"):
        df["next_" + c] = df[c].shift(-1)
    df["next_game"] = df["game_pk"].shift(-1)
    df["next_inn"]  = df["inning"].shift(-1)
    df["next_top"]  = df["inning_topbot"].shift(-1)
    same = (
        (df["next_game"] == df["game_pk"])
        & (df["next_inn"] == df["inning"])
        & (df["next_top"] == df["inning_topbot"])
    )
    for c in ("on_1b", "on_2b", "on_3b"):
        df["post_" + c] = np.where(same, df["next_" + c], np.nan)

    hits = df[df["events"].isin(_HIT_MAP) & df["post_on_1b"].notna()].copy()
    hits["hit_type"] = hits["events"].map(_HIT_MAP)
    hits["runs_scored"] = (hits["post_bat_score"] - hits["bat_score"]).clip(lower=0)

    def _compute_empirical_delta_re(row: pd.Series) -> float:
        old_state = (int(row.on_1b), int(row.on_2b), int(row.on_3b), int(row.outs_when_up))
        new_state = (int(row.post_on_1b), int(row.post_on_2b), int(row.post_on_3b), int(row.outs_when_up))
        if old_state not in re24 or new_state not in re24:
            return np.nan
        return row.runs_scored + re24[new_state] - re24[old_state]

    hits["demp"] = hits.apply(_compute_empirical_delta_re, axis=1)
    hits = hits[hits["demp"].notna()]

    # Start from the theoretical values, and overwrite cells with sufficient sample size with empirical values
    delta: dict[HitDeltaKey, float] = dict(theory)
    for (b1, b2, b3, outs), g in hits.groupby(["on_1b", "on_2b", "on_3b", "outs_when_up"]):
        for ht in ("1B", "2B", "3B"):
            sub = g[g["hit_type"] == ht]
            if len(sub) >= min_n:
                delta[(ht, int(b1), int(b2), int(b3), int(outs))] = float(sub["demp"].mean())
    return delta


def save_re24(re24: dict[BaseOutState, float], delta: dict[HitDeltaKey, float], out_dir: Path) -> None:
    """Save both tables to JSON in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # key: "0-0-0-2" (four ints joined by dash)
    re24_str = {"-".join(str(x) for x in k): v for k, v in re24.items()}
    # key: "1B-0-0-0-2" (hit_type + four ints)
    delta_str = {"-".join(str(x) for x in k): v for k, v in delta.items()}

    with open(out_dir / "re24_table.json", "w", encoding="utf-8") as f:
        json.dump(re24_str, f, indent=2)
    with open(out_dir / "delta_re.json", "w", encoding="utf-8") as f:
        json.dump(delta_str, f, indent=2)


def load_re24(data_dir: Path) -> tuple[dict[BaseOutState, float], dict[HitDeltaKey, float]]:
    """Load RE24 table and ΔRE dict from out_dir."""
    data_dir = Path(data_dir)

    with open(data_dir / "re24_table.json", encoding="utf-8") as f:
        re24_str: dict[str, float] = json.load(f)
    with open(data_dir / "delta_re.json", encoding="utf-8") as f:
        delta_str: dict[str, float] = json.load(f)

    re24: dict[BaseOutState, float] = {
        tuple(int(x) for x in k.split("-")): v for k, v in re24_str.items()
    }

    delta: dict[HitDeltaKey, float] = {}
    for k_str, v in delta_str.items():
        parts = k_str.split("-")
        # parts[0] = hit_type (e.g. "1B"), parts[1:5] = ints
        delta[(parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))] = v

    return re24, delta
