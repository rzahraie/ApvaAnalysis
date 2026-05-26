#!/usr/bin/env python3
"""Build the canonical APVA row-level signed forward-return dataset.

This extracts the transformation used in cells 35 and 40 of
``apva_phase01_exploration.ipynb``:

1. Raw xApvaV01 state logs are resolved into rolling pressure features from
   MacroState quadruplets and the canonical archetype lookup.
2. OHLC and ActiveDirection are used to build direction-signed forward
   outcomes normalized by the 14-bar ATR calculated on the pressure timeline.

The implementation reconstructs a dataset pipeline; it does not search for
candidate rules or tune validation thresholds.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_dataset_readiness_check import build_candidate_entries

RAW_REQUIRED_COLUMNS = [
    "Instrument",
    "BarIndex",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "MacroState",
    "ActiveDirection",
]
CANONICAL_REQUIRED_COLUMNS = [
    "Instrument",
    "File",
    "BarIndex",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "HorizonBars",
    "DominantPressure",
    "RollingDirectionalPresence",
    "RollingEntropy",
    "SignedNormalizedReturn",
    "DirectionalNormalizedMAE",
]
DESIRED_COLUMNS = [
    "DominantPressureValue",
    "SignedReturn",
    "RawReturn",
    "NormalizedReturn",
    "DirectionalMFE",
    "DirectionalMAE",
    "DirectionalNormalizedMFE",
    "FutureClose",
]
PRESSURE_COLUMNS = [
    "Instrument",
    "File",
    "BarIndex",
    "Time",
    "MacroState",
    "Quadruplet",
    "ResolvedArchetype",
    "ResolutionMethod",
    "Similarity",
    "RollingEntropy",
    "RollingVelocity",
    "RollingDominantFraction",
    "RollingDirectionalPresence",
    "CurrentDirectionalEmergence",
    "DirectionalReleasePressure",
    "ExhaustionPressure",
    "CompressionPressure",
    "RotationalPressure",
    "DominantPressure",
    "DominantPressureValue",
]
OUTPUT_COLUMNS = PRESSURE_COLUMNS + [
    "Open",
    "High",
    "Low",
    "Close",
    "ActiveDirection",
    "DirectionSign",
    "PrevClose",
    "TR",
    "ATR14",
    "HorizonBars",
    "FutureClose",
    "RawReturn",
    "NormalizedReturn",
    "SignedReturn",
    "SignedNormalizedReturn",
    "DirectionalMFE",
    "DirectionalMAE",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
    "DirectionalHit",
    "HasDirection",
]
def display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def expand_inputs(patterns: list[str], workspace: Path) -> list[Path]:
    paths: set[Path] = set()
    for raw in patterns:
        pattern = str(Path(raw) if Path(raw).is_absolute() else workspace / raw)
        paths.update(Path(match).resolve() for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() == ".csv")


def parse_horizons(values: list[str]) -> list[int]:
    horizons = []
    for value in values:
        horizons.extend(int(token.strip()) for token in str(value).split(",") if token.strip())
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("At least one positive --horizons value is required.")
    return sorted(set(horizons))


def parse_instrument_map(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--instrument-map entries must use PATH_OR_FILENAME=INSTRUMENT.")
        key, instrument = value.split("=", maxsplit=1)
        mapping[key.strip()] = instrument.strip().upper()
    return mapping


def state_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    probs = pd.Series(values).value_counts(normalize=True).values
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def transition_velocity(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    return float(sum(values[i] != values[i - 1] for i in range(1, len(values))) / (len(values) - 1))


def dominant_fraction(values: list[str]) -> float:
    return float(pd.Series(values).value_counts(normalize=True).iloc[0]) if values else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_inverse(value: float, low: float, high: float) -> float:
    return 0.0 if high <= low else clamp01((high - value) / (high - low))


def normalize_forward(value: float, low: float, high: float) -> float:
    return 0.0 if high <= low else clamp01((value - low) / (high - low))


def has_directional_emergence(quadruplet: str) -> bool:
    q = str(quadruplet)
    return (
        "Unresolved->Directional" in q
        or "Balance->Directional" in q
        or "Degrading->Directional" in q
        or q.endswith("Directional")
    )


def vectorize_quadruplet(states: list[str], index: int) -> Optional[str]:
    return None if index < 3 else "->".join(states[index - 3:index + 1])


def feature_vector(quadruplet: str) -> Optional[np.ndarray]:
    states = str(quadruplet).split("->")
    if len(states) != 4:
        return None

    def count_state(name: str) -> int:
        return sum(state == name for state in states)

    def has_pattern(pattern: str) -> int:
        parts = pattern.split("->")
        return int(any(states[i:i + len(parts)] == parts for i in range(len(states) - len(parts) + 1)))

    return np.array([
        count_state("Unresolved"),
        count_state("Directional"),
        count_state("Degrading"),
        count_state("Balance"),
        count_state("Unknown"),
        sum(states[i] != states[i - 1] for i in range(1, len(states))),
        int(states[-1] == "Directional"),
        int(states[-1] == "Degrading"),
        int(states[-1] == "Balance"),
        int(states[-1] == "Unresolved"),
        int(states == ["Unresolved"] * 4),
        int(states == ["Directional"] * 4),
        int(states == ["Degrading"] * 4),
        int(states == ["Balance"] * 4),
        has_pattern("Unresolved->Directional"),
        has_pattern("Directional->Degrading"),
        int(bool(has_pattern("Balance->Unresolved") or has_pattern("Unresolved->Balance"))),
        int(bool(has_pattern("Degrading->Unresolved") or has_pattern("Unresolved->Degrading"))),
    ], dtype=float)


class ArchetypeResolver:
    """Resolve raw MacroState quadruplets as defined by notebook cell 35."""

    CANONICAL_NAMES = {
        "Persistent Non-Directional Compression": "Stable Compression Basin",
        "Fragile Rotational Churn": "Unstable Rotational Balance",
        "Degradation / Exhaustion Collapse": "Sponsor Exhaustion Collapse",
        "Stable Compression Basin": "Stable Compression Basin",
        "Unstable Rotational Balance": "Unstable Rotational Balance",
        "Sponsor Exhaustion Collapse": "Sponsor Exhaustion Collapse",
        "Directional Continuation / Sponsor Release": "Directional Continuation / Sponsor Release",
        "Pure Unresolved Compression Equilibrium": "Pure Unresolved Compression Equilibrium",
    }

    def __init__(self, lookup_path: Path) -> None:
        lookup = pd.read_csv(lookup_path)
        required = {"Quadruplet", "Archetype"}
        missing = required - set(lookup.columns)
        if missing:
            raise ValueError(f"Archetype lookup is missing required columns: {sorted(missing)}")
        self.exact = {
            str(row.Quadruplet): self.canonicalize(str(row.Archetype))
            for row in lookup.itertuples(index=False)
        }
        vectors = []
        arches = []
        for quadruplet, archetype in self.exact.items():
            vector = feature_vector(quadruplet)
            if vector is not None:
                vectors.append(vector)
                arches.append(archetype)
        if not vectors:
            raise ValueError("Archetype lookup contains no usable quadruplet vectors.")
        self.known_vectors = np.vstack(vectors)
        self.known_archetypes = arches

    def canonicalize(self, name: str) -> str:
        return self.CANONICAL_NAMES.get(name, name)

    def resolve(self, quadruplet: str) -> tuple[str, float, str]:
        exact = self.exact.get(quadruplet)
        if exact is not None:
            return exact, 1.0, "Exact"
        vector = feature_vector(quadruplet)
        if vector is None:
            return "Unknown", 0.0, "Invalid"
        distances = np.linalg.norm(self.known_vectors - vector, axis=1)
        index = int(np.argmin(distances))
        return self.known_archetypes[index], 1.0 / (1.0 + float(distances[index])), "Soft"


def input_instrument(path: Path, frame: pd.DataFrame, mapping: dict[str, str]) -> str:
    candidates = [str(path), str(path.resolve()), path.name]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    instruments = frame["Instrument"].dropna().astype(str).str.upper().unique().tolist()
    if len(instruments) != 1:
        raise ValueError(
            f"Input {path} must contain one Instrument value or be specified in --instrument-map."
        )
    return instruments[0]


def load_raw_input(
    path: Path,
    session_filter: str,
    mapping: dict[str, str],
    workspace: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(path)
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw input {path.name} is missing columns: {missing}")
    embedded_headers = raw.index[raw["Instrument"].astype(str).eq("Instrument")].tolist()
    if embedded_headers:
        csv_rows = ",".join(str(index + 2) for index in embedded_headers)
        raise ValueError(
            f"Input {path.name} contains embedded repeated header rows at CSV row(s) {csv_rows}; "
            "export or clean the raw state log before canonical dataset construction."
        )
    instrument = input_instrument(path, raw, mapping)
    source_rows = len(raw)
    if session_filter != "all":
        if "SessionContext" not in raw.columns:
            raise ValueError(f"Input {path.name} does not include SessionContext for --session-filter.")
        raw = raw.loc[
            raw["SessionContext"].fillna("").astype(str).str.upper().eq(session_filter.upper())
        ].copy()
    raw["Instrument"] = instrument
    raw["File"] = path.name
    raw["BarIndex"] = pd.to_numeric(raw["BarIndex"], errors="raise")
    for column in ["Open", "High", "Low", "Close"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw = raw.sort_values("BarIndex").reset_index(drop=True)
    summary = {
        "InputPath": display_path(path, workspace),
        "Instrument": instrument,
        "File": path.name,
        "SourceRows": source_rows,
        "RowsAfterSessionFilter": len(raw),
        "SessionFilter": session_filter,
        "TimeMin": str(raw["Time"].min()) if not raw.empty else "",
        "TimeMax": str(raw["Time"].max()) if not raw.empty else "",
    }
    return raw, summary


def compute_pressure(raw: pd.DataFrame, resolver: ArchetypeResolver) -> pd.DataFrame:
    states = raw["MacroState"].fillna("Unknown").astype(str).tolist()
    rows = []
    window = 10
    for index in range(len(raw)):
        quadruplet = vectorize_quadruplet(states, index)
        if quadruplet is None:
            continue
        archetype, similarity, resolution_method = resolver.resolve(quadruplet)
        start = max(0, len(rows) - window + 1)
        arch_values = [row["ResolvedArchetype"] for row in rows[start:]] + [archetype]
        recent_quads = [row["Quadruplet"] for row in rows[start:]] + [quadruplet]
        entropy = state_entropy(arch_values)
        velocity = transition_velocity(arch_values)
        dom_frac = dominant_fraction(arch_values)
        low_entropy = normalize_inverse(entropy, 0.0, 1.6)
        low_velocity = normalize_inverse(velocity, 0.0, 0.8)
        high_dominance = normalize_forward(dom_frac, 0.4, 1.0)
        high_entropy = normalize_forward(entropy, 0.0, 1.6)
        high_velocity = normalize_forward(velocity, 0.0, 0.8)
        directional_presence = sum("Directional" in q for q in recent_quads) / max(1, len(recent_quads))
        current_emergence = float(has_directional_emergence(quadruplet))
        is_compression = archetype in {
            "Pure Unresolved Compression Equilibrium", "Stable Compression Basin"
        }
        is_rotational = archetype == "Unstable Rotational Balance"
        is_directional = archetype == "Directional Continuation / Sponsor Release"
        is_exhaustion = archetype == "Sponsor Exhaustion Collapse"

        # Notebook cell 35 explicitly uses zeroes here for raw validation logs.
        high_failure = normalize_forward(0.0, 0.0, 70.0)
        high_continuation = normalize_forward(0.0, 0.0, 75.0)
        release_structure = clamp01(
            0.35 * low_entropy + 0.25 * low_velocity + 0.20 * high_dominance + 0.20 * directional_presence
        )
        pressures = {
            "DirectionalReleasePressure": clamp01(
                release_structure * (0.35 + 0.45 * current_emergence + 0.20 * float(is_directional))
            ),
            "ExhaustionPressure": clamp01(
                0.35 * high_continuation + 0.25 * high_failure + 0.20 * float(is_directional)
                + 0.10 * high_velocity + 0.10 * float(is_exhaustion)
            ),
            "CompressionPressure": clamp01(
                0.45 * float(is_compression) + 0.25 * high_dominance
                + 0.20 * low_velocity + 0.10 * low_entropy
            ),
            "RotationalPressure": clamp01(
                0.35 * high_entropy + 0.30 * high_velocity + 0.25 * float(is_rotational)
                + 0.10 * high_failure
            ),
        }
        dominant_pressure = max(pressures, key=pressures.get)
        rows.append({
            "Instrument": raw.loc[index, "Instrument"],
            "File": raw.loc[index, "File"],
            "BarIndex": raw.loc[index, "BarIndex"],
            "Time": raw.loc[index, "Time"],
            "MacroState": states[index],
            "Quadruplet": quadruplet,
            "ResolvedArchetype": archetype,
            "ResolutionMethod": resolution_method,
            "Similarity": similarity,
            "RollingEntropy": entropy,
            "RollingVelocity": velocity,
            "RollingDominantFraction": dom_frac,
            "RollingDirectionalPresence": directional_presence,
            "CurrentDirectionalEmergence": current_emergence,
            **pressures,
            "DominantPressure": dominant_pressure,
            "DominantPressureValue": pressures[dominant_pressure],
        })
    return pd.DataFrame(rows, columns=PRESSURE_COLUMNS)


def direction_sign(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"up", "long", "bull", "bullish", "black"}:
        return 1
    if normalized in {"down", "short", "bear", "bearish", "red"}:
        return -1
    return 0


def compute_forward_returns(
    pressures: list[pd.DataFrame],
    raw_frames: list[pd.DataFrame],
    horizons: list[int],
) -> pd.DataFrame:
    pressure = pd.concat(pressures, ignore_index=True)
    price = pd.concat(raw_frames, ignore_index=True)
    merged = pressure.merge(
        price[["Instrument", "File", "BarIndex", "Open", "High", "Low", "Close", "ActiveDirection"]],
        on=["Instrument", "File", "BarIndex"],
        how="inner",
    ).sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)
    merged["DirectionSign"] = merged["ActiveDirection"].apply(direction_sign)
    merged["PrevClose"] = merged.groupby(["Instrument", "File"])["Close"].shift(1)
    merged["TR"] = np.maximum.reduce([
        merged["High"] - merged["Low"],
        (merged["High"] - merged["PrevClose"]).abs(),
        (merged["Low"] - merged["PrevClose"]).abs(),
    ])
    merged["ATR14"] = merged.groupby(["Instrument", "File"])["TR"].transform(lambda values: values.rolling(14).mean())
    rows = []
    for _, group in merged.groupby(["Instrument", "File"], sort=True):
        group = group.sort_values("BarIndex").reset_index(drop=True)
        closes = group["Close"].to_numpy()
        highs = group["High"].to_numpy()
        lows = group["Low"].to_numpy()
        signs = group["DirectionSign"].to_numpy()
        for index in range(len(group)):
            for horizon in horizons:
                if index + horizon >= len(group):
                    continue
                close = closes[index]
                future_close = closes[index + horizon]
                raw_return = future_close - close
                signed_return = raw_return * signs[index]
                atr = group.loc[index, "ATR14"]
                valid_atr = pd.notna(atr) and atr > 0
                normalized_return = raw_return / atr if valid_atr else np.nan
                signed_normalized_return = signed_return / atr if valid_atr else np.nan
                future_high = float(np.max(highs[index + 1:index + horizon + 1]))
                future_low = float(np.min(lows[index + 1:index + horizon + 1]))
                if signs[index] > 0:
                    mfe, mae = future_high - close, future_low - close
                elif signs[index] < 0:
                    mfe, mae = close - future_low, close - future_high
                else:
                    mfe, mae = np.nan, np.nan
                row = group.iloc[index].to_dict()
                row.update({
                    "HorizonBars": horizon,
                    "FutureClose": future_close,
                    "RawReturn": raw_return,
                    "NormalizedReturn": normalized_return,
                    "SignedReturn": signed_return,
                    "SignedNormalizedReturn": signed_normalized_return,
                    "DirectionalMFE": mfe,
                    "DirectionalMAE": mae,
                    "DirectionalNormalizedMFE": mfe / atr if valid_atr and pd.notna(mfe) else np.nan,
                    "DirectionalNormalizedMAE": mae / atr if valid_atr and pd.notna(mae) else np.nan,
                    "DirectionalHit": int(signed_return > 0 and signs[index] != 0),
                    "HasDirection": int(signs[index] != 0),
                })
                rows.append(row)
    forward = pd.DataFrame(rows)
    if forward.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return forward.loc[forward["HasDirection"].eq(1), OUTPUT_COLUMNS].reset_index(drop=True)


def base_and_candidate_counts(frame: pd.DataFrame, source: str) -> list[dict[str, object]]:
    validation_args = argparse.Namespace(
        horizon=5,
        pressure="CompressionPressure",
        entropy_min=0.88,
        entropy_max=1.30,
        directional_presence=0.0,
        lookback=5,
    )
    selected = build_candidate_entries(frame, validation_args)
    rows = []
    for candidate, subset in selected.items():
        instruments = subset["Instrument"].astype(str).str.upper() if not subset.empty else pd.Series(dtype=str)
        rows.append({
            "Source": source,
            "Candidate": candidate,
            "Count": len(subset),
            "ES_Count": int(instruments.eq("ES").sum()),
            "NQ_Count": int(instruments.eq("NQ").sum()),
        })
    return rows


def column_report(frame: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "DominantPressure": "Notebook cell 35 pressure maximum",
        "RollingDirectionalPresence": "Notebook cell 35 recent quadruplet directional fraction",
        "RollingEntropy": "Notebook cell 35 entropy of rolling resolved archetypes",
        "SignedNormalizedReturn": "Notebook cell 40 SignedReturn / ATR14",
        "DirectionalNormalizedMAE": "Notebook cell 40 DirectionalMAE / ATR14",
        "DominantPressureValue": "Notebook cell 35 selected pressure value",
        "NormalizedReturn": "Notebook cell 40 RawReturn / ATR14",
    }
    rows = []
    for column in CANONICAL_REQUIRED_COLUMNS:
        rows.append({
            "Column": column, "Requirement": "Required", "Present": column in frame.columns,
            "SourceOrFormula": formulas.get(column, "Raw state-log field or pipeline key"),
        })
    for column in DESIRED_COLUMNS:
        rows.append({
            "Column": column, "Requirement": "OptionalDesired", "Present": column in frame.columns,
            "SourceOrFormula": formulas.get(column, "Notebook cell 40 directional forward outcome"),
        })
    return pd.DataFrame(rows)


def compare_report(generated: pd.DataFrame, reference: pd.DataFrame, candidate_counts: pd.DataFrame) -> pd.DataFrame:
    keys = ["Instrument", "File", "BarIndex", "HorizonBars"]
    generated_keys = set(map(tuple, generated[keys].to_numpy().tolist()))
    reference_keys = set(map(tuple, reference[keys].to_numpy().tolist()))
    rows: list[dict[str, object]] = []

    def add(metric: str, generated_value: object, reference_value: object, detail: object = "") -> None:
        if isinstance(generated_value, (float, np.floating)) and isinstance(reference_value, (float, np.floating)):
            matches = bool(np.isclose(generated_value, reference_value, equal_nan=True))
        else:
            matches = generated_value == reference_value
        rows.append({
            "Metric": metric, "Generated": generated_value, "CompareTo": reference_value,
            "Matches": matches, "Detail": detail,
        })

    add("RowCount", len(generated), len(reference))
    add("ColumnOrder", ",".join(generated.columns), ",".join(reference.columns))
    add("KeySet", len(generated_keys), len(reference_keys), f"overlap={len(generated_keys & reference_keys)}")
    add("MeanSignedNormalizedReturn", float(generated["SignedNormalizedReturn"].mean()), float(reference["SignedNormalizedReturn"].mean()))
    add("MedianSignedNormalizedReturn", float(generated["SignedNormalizedReturn"].median()), float(reference["SignedNormalizedReturn"].median()))
    for pressure in sorted(set(generated["DominantPressure"].dropna()) | set(reference["DominantPressure"].dropna())):
        add(
            f"DominantPressureCount:{pressure}",
            int(generated["DominantPressure"].eq(pressure).sum()),
            int(reference["DominantPressure"].eq(pressure).sum()),
        )
    for candidate in ["Base_ES_NQ", "RRCCC", "CCRRR", "PriorSlope_DominantPressureValue_Q3"]:
        generated_count = int(candidate_counts.loc[
            candidate_counts["Source"].eq("Generated") & candidate_counts["Candidate"].eq(candidate), "Count"
        ].iloc[0])
        reference_count = int(candidate_counts.loc[
            candidate_counts["Source"].eq("CompareTo") & candidate_counts["Candidate"].eq(candidate), "Count"
        ].iloc[0])
        add(f"CandidateCount:{candidate}", generated_count, reference_count)
    common_keys = list(generated_keys & reference_keys)
    if common_keys:
        generated_common = generated.set_index(keys).loc[common_keys].sort_index()
        reference_common = reference.set_index(keys).loc[common_keys].sort_index()
        common_columns = [column for column in generated.columns if column in reference.columns and column not in keys]
        mismatch_count = 0
        for column in common_columns:
            left, right = generated_common[column], reference_common[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                mismatch_count += int((~np.isclose(left.to_numpy(), right.to_numpy(), equal_nan=True)).sum())
            else:
                mismatch_count += int((left.fillna("").astype(str) != right.fillna("").astype(str)).sum())
        add("SharedKeyCellDifferences", mismatch_count, 0, f"shared_keys={len(common_keys)}")
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="tables/apva_forward_signed_return_dataset_generated.csv")
    parser.add_argument("--outdir", default="outputs/build_forward_return_dataset")
    parser.add_argument("--lookup", default="tables/apva_online_archetype_lookup.csv")
    parser.add_argument("--horizons", nargs="+", default=["5"])
    parser.add_argument("--instrument-map", nargs="*", default=[])
    parser.add_argument("--session-filter", choices=["all", "RTH", "ETH"], default="all")
    parser.add_argument("--normalization-mode", choices=["atr14"], default="atr14")
    parser.add_argument("--compare-to")
    args = parser.parse_args(argv)

    workspace = Path.cwd()
    paths = expand_inputs(args.inputs, workspace)
    if not paths:
        raise FileNotFoundError("No CSV inputs matched --inputs.")
    horizons = parse_horizons(args.horizons)
    mapping = parse_instrument_map(args.instrument_map)
    resolver = ArchetypeResolver((workspace / args.lookup).resolve())
    raw_frames = []
    pressure_frames = []
    source_rows = []
    keys_seen = set()
    for path in paths:
        raw, summary = load_raw_input(path, args.session_filter, mapping, workspace)
        identity = (summary["Instrument"], summary["File"])
        if identity in keys_seen:
            raise ValueError(f"Duplicate Instrument/File input identity is ambiguous: {identity}")
        keys_seen.add(identity)
        pressure = compute_pressure(raw, resolver)
        summary["PressureRows"] = len(pressure)
        source_rows.append(summary)
        raw_frames.append(raw)
        pressure_frames.append(pressure)
    generated = compute_forward_returns(pressure_frames, raw_frames, horizons)
    out_path = (workspace / args.out).resolve()
    outdir = (workspace / args.outdir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    generated.to_csv(out_path, index=False)
    source_summary = pd.DataFrame(source_rows)
    generated_candidate_rows = base_and_candidate_counts(generated, "Generated")
    candidate_rows = list(generated_candidate_rows)
    compare = pd.DataFrame()
    if args.compare_to:
        reference = pd.read_csv((workspace / args.compare_to).resolve())
        candidate_rows.extend(base_and_candidate_counts(reference, "CompareTo"))
        compare = compare_report(generated, reference, pd.DataFrame(candidate_rows))
        compare.to_csv(outdir / "build_dataset_compare_report.csv", index=False)
    candidate_counts = pd.DataFrame(candidate_rows)
    build_summary = pd.DataFrame([
        {"Metric": "pipeline_source", "Value": "apva_phase01_exploration.ipynb cells 35 and 40"},
        {"Metric": "transformation_status", "Value": "recovered notebook formula extraction"},
        {"Metric": "input_files", "Value": ",".join(display_path(path, workspace) for path in paths)},
        {"Metric": "input_file_count", "Value": len(paths)},
        {"Metric": "output_path", "Value": display_path(out_path, workspace)},
        {"Metric": "output_rows", "Value": len(generated)},
        {"Metric": "instruments", "Value": ",".join(sorted(generated["Instrument"].unique()))},
        {"Metric": "horizons", "Value": ",".join(str(horizon) for horizon in horizons)},
        {"Metric": "session_filter", "Value": args.session_filter},
        {"Metric": "normalization_mode", "Value": "ATR14 from pressure-timeline rows, matching notebook cell 40"},
        {"Metric": "dominant_pressure_mapping", "Value": "MacroState quadruplet archetype resolution plus notebook v2 pressure formulas"},
        {"Metric": "raw_score_fields_used_for_pressure", "Value": "none"},
        {"Metric": "compare_to", "Value": args.compare_to or "not supplied"},
        {"Metric": "compare_exact_equivalence", "Value": bool(not compare.empty and compare["Matches"].all()) if args.compare_to else "not evaluated"},
    ])
    source_summary.to_csv(outdir / "build_dataset_source_inventory.csv", index=False)
    build_summary.to_csv(outdir / "build_dataset_summary.csv", index=False)
    column_report(generated).to_csv(outdir / "build_dataset_column_report.csv", index=False)
    candidate_counts.to_csv(outdir / "build_dataset_candidate_counts.csv", index=False)
    print("APVA canonical forward-return dataset build complete")
    print(build_summary.to_string(index=False))
    if not compare.empty:
        print(compare[["Metric", "Matches", "Detail"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
