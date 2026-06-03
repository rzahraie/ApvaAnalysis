#!/usr/bin/env python3
"""
APVA Structural Life Cycle Study v0.1

Purpose:
    Classify each structural state-age bucket into a mechanical life-cycle phase:
    Birth, Growth, Maturity, Decay, Terminal.

    The phase assignment is based on state-age hazard/persistence. Outcome is
    reported separately and is not used to assign phases.

Design:
    - Standard library only.
    - No trades.
    - No entries/exits.
    - No fitting.
    - No optimization.
    - Research only.

Inputs:
    One or more APVA evidence CSV files or directories.

Outputs:
    Per instrument:
        Evidence/Output/<Instrument>/StructuralLifeCycle_<Instrument>.txt

    Aggregate:
        Evidence/Output/StructuralLifeCycle/StructuralLifeCycle_All.txt

Usage:
    python APVA_StructuralLifeCycle_44.py Evidence/6E Evidence/NQ Evidence/CL
    python APVA_StructuralLifeCycle_44.py file1.csv file2.csv file3.csv --out-root Evidence/Output
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

FAMILIES = ("A", "B", "C", "D", "N")

STRUCTURAL_STATES = (
    "RecoveryResolution",
    "ExhaustionPersistence",
    "CompressionProcessing",
    "NeutralProcessing",
    "ReassertionProcessing",
    "DecayToNeutral",
    "DestructiveRotation",
    "ConstructiveEmergence",
    "MixedStructure",
)

ARCHETYPE_ORDER = (
    "Recovery",
    "Exhaustion",
    "Reassertion",
    "Decay",
    "Compression Resolution",
    "Destructive Persistence",
    "Constructive Emergence",
    "Neutral Drift",
    "Unclassified",
)

AGE_BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11-20", "21+")
AGE_BUCKET_ORDER = {x: i for i, x in enumerate(AGE_BUCKETS)}
LAGS = (1, 3, 5)
MIN_COUNT = 20


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def find_col(headers: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {norm_key(h): h for h in headers}
    for c in candidates:
        k = norm_key(c)
        if k in normalized:
            return normalized[k]
    return None


def val(row: Dict[str, str], col: Optional[str], default: str = "") -> str:
    if not col:
        return default
    return row.get(col, default)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() in {"n/a", "na", "none", "null"}:
            return None
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return stats.median(xs) if xs else 0.0


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:.{nd}f}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_flags(s: str) -> Tuple[str, ...]:
    if not s:
        return tuple()
    return tuple(x.strip() for x in str(s).split(";") if x.strip())


@dataclass
class Bar:
    instrument: str
    source: str
    rownum: int
    time: str
    close: Optional[float]
    volume_polarity: str
    participation_state: str
    expansion_state: str
    compression_state: str
    dissipation_state: str
    acceptance_state: str
    flags: Tuple[str, ...]
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    session_key: str = ""
    has_ohlc: bool = False
    has_volume: bool = False
    has_volume_polarity: bool = False
    bar_range: Optional[float] = None
    body: Optional[float] = None
    true_range: Optional[float] = None
    close_location: Optional[float] = None
    volume_relative_to_previous: Optional[float] = None
    volume_relative_to_session_mean: Optional[float] = None
    volume_relative_to_rolling_mean: Optional[float] = None
    range_relative_to_previous: Optional[float] = None
    body_relative_to_previous: Optional[float] = None
    true_range_relative_to_previous: Optional[float] = None
    volume_expansion_flag: Optional[bool] = None
    volume_contraction_flag: Optional[bool] = None
    range_expansion_flag: Optional[bool] = None
    range_contraction_flag: Optional[bool] = None
    family: str = "N"
    path3: Tuple[str, ...] = field(default_factory=tuple)
    path4: Tuple[str, ...] = field(default_factory=tuple)
    path5: Tuple[str, ...] = field(default_factory=tuple)
    archetype: str = "Unclassified"
    state: str = "MixedStructure"
    age: int = 0
    age_bucket: str = ""
    position: str = ""


@dataclass
class Outcome:
    count: int = 0
    valid: int = 0
    cont: int = 0
    fail: int = 0
    flat: int = 0
    values: List[float] = field(default_factory=list)

    def add(self, dr: Optional[float]) -> None:
        self.count += 1
        if dr is None:
            return
        self.valid += 1
        self.values.append(dr)
        if dr > 1e-12:
            self.cont += 1
        elif dr < -1e-12:
            self.fail += 1
        else:
            self.flat += 1

    @property
    def cont_rate(self) -> float:
        return self.cont / self.valid if self.valid else 0.0

    @property
    def fail_rate(self) -> float:
        return self.fail / self.valid if self.valid else 0.0

    @property
    def flat_rate(self) -> float:
        return self.flat / self.valid if self.valid else 0.0

    @property
    def skew(self) -> float:
        return self.cont_rate - self.fail_rate

    @property
    def mean_dr(self) -> float:
        return mean(self.values)

    @property
    def median_dr(self) -> float:
        return median(self.values)


@dataclass
class LifeCycleRow:
    state: str
    age: str
    count: int
    persistence1: float
    hazard1: float
    persistence3: float
    hazard3: float
    persistence5: float
    hazard5: float
    continuation: float
    failure: float
    skew: float
    mean_dr: float
    dominant_death_target: str
    dominant_death_prob: float
    phase: str


@dataclass
class InstrumentResult:
    instrument: str
    bars: List[Bar]
    source_paths: List[str]
    rows: List[LifeCycleRow]


def has_flag(b: Bar, needle: str) -> bool:
    return needle in b.flags


def is_accepted(b: Bar) -> bool:
    return b.acceptance_state.lower() == "accepted" or has_flag(b, "AcceptanceAccepted")


def is_compressed(b: Bar) -> bool:
    return b.compression_state.lower() in {"local", "clustered", "contained"} or any(
        f.startswith("Compression") and f != "CompressionAbsent" for f in b.flags
    )


def is_dissipating(b: Bar) -> bool:
    return b.dissipation_state.lower() in {"local", "contained", "accepted"} or any(
        f.startswith("Dissipation") and f != "DissipationAbsent" for f in b.flags
    )


def is_expanding(b: Bar) -> bool:
    return b.expansion_state.lower() in {"local", "strong", "climactic", "expanding"} or any(
        f.startswith("Expansion") and f != "ExpansionAbsent" for f in b.flags
    )


def is_peak(b: Bar) -> bool:
    return (
        b.participation_state.lower() == "peak"
        or has_flag(b, "ParticipationPeak")
        or has_flag(b, "PeakVolume")
    )


def is_climactic(b: Bar) -> bool:
    ps = b.participation_state.lower()
    es = b.expansion_state.lower()
    return (
        ps == "climactic"
        or es == "climactic"
        or has_flag(b, "ParticipationClimactic")
        or has_flag(b, "ExpansionClimactic")
    )


def assign_family(b: Bar) -> str:
    accepted = is_accepted(b)
    compressed = is_compressed(b)
    dissipating = is_dissipating(b)
    expanding = is_expanding(b)
    peak = is_peak(b)
    climactic = is_climactic(b)

    if accepted and compressed:
        return "A"
    if compressed and peak:
        return "B"
    if (accepted and peak) or (accepted and climactic) or (accepted and dissipating and peak):
        return "C"
    if accepted and expanding:
        return "D"
    return "N"


def path_contains(path: Tuple[str, ...], sub: Tuple[str, ...]) -> bool:
    if len(sub) > len(path):
        return False
    return any(path[i:i + len(sub)] == sub for i in range(len(path) - len(sub) + 1))


def primary_archetype(path: Tuple[str, ...]) -> str:
    matches = set()
    if (len(path) >= 2 and path[-2:] == ("C", "B")) or path_contains(path, ("C", "C", "B")):
        matches.add("Recovery")
    if path_contains(path, ("C", "C", "C")) or (len(path) >= 2 and path[-2:] == ("C", "C")):
        matches.add("Exhaustion")
    if all(x in ("D", "N") for x in path) and len(set(path)) > 1 and (
        path[-1] == "D" or (len(path) >= 2 and path[-2:] == ("N", "D"))
    ):
        matches.add("Reassertion")
    if path and path[-1] == "N" and any(x in ("B", "C") for x in path[:-1]):
        matches.add("Decay")
    if any(path[i] == "B" and path[i + 1] in ("N", "C") for i in range(len(path) - 1)):
        matches.add("Compression Resolution")
    if path_contains(path, ("C", "C")) or path_contains(path, ("D", "D")):
        matches.add("Destructive Persistence")
    if path and path[-1] in ("A", "B") and any(x in ("N", "D") for x in path[:-1]):
        matches.add("Constructive Emergence")
    if path.count("N") >= len(path) - 1:
        matches.add("Neutral Drift")
    if not matches:
        matches.add("Unclassified")
    for a in ARCHETYPE_ORDER:
        if a in matches:
            return a
    return "Unclassified"


def assign_structural_state(path: Tuple[str, ...], archetype: str) -> str:
    if not path:
        return "MixedStructure"
    n = path.count("N")
    if n >= len(path) - 1 or (path_contains(path, ("N", "N")) and path[-1] == "N"):
        return "NeutralProcessing"
    if archetype == "Recovery" or (len(path) >= 2 and path[-2:] == ("C", "B")) or path_contains(path, ("C", "C", "B")):
        return "RecoveryResolution"
    if archetype == "Exhaustion" or path_contains(path, ("C", "C", "C")) or (len(path) >= 2 and path[-2:] == ("C", "C")):
        return "ExhaustionPersistence"
    if archetype == "Compression Resolution" or "B" in path or any(
        path[i] == "B" and path[i + 1] in ("N", "C") for i in range(len(path) - 1)
    ):
        return "CompressionProcessing"
    if archetype == "Decay" or (path[-1] == "N" and any(x in ("B", "C") for x in path[:-1])):
        return "DecayToNeutral"
    if archetype == "Reassertion" or (all(x in ("D", "N") for x in path) and path[-1] == "D"):
        return "ReassertionProcessing"
    if archetype == "Destructive Persistence" or path_contains(path, ("C", "C")) or path_contains(path, ("D", "D")):
        return "DestructiveRotation"
    if archetype == "Constructive Emergence" or (
        path[-1] in ("A", "B") and any(x in ("N", "D", "C") for x in path[:-1])
    ):
        return "ConstructiveEmergence"
    return "MixedStructure"


def age_bucket(age: int) -> str:
    if age <= 5:
        return str(age)
    if age <= 10:
        return "6-10"
    if age <= 20:
        return "11-20"
    return "21+"


def age_sort_key(age: str) -> int:
    return AGE_BUCKET_ORDER.get(age, 999)


def assign_run_age(bars: List[Bar]) -> None:
    i = 0
    while i < len(bars):
        j = i
        st = bars[i].state
        while j + 1 < len(bars) and bars[j + 1].state == st:
            j += 1
        run_len = j - i + 1
        for k in range(i, j + 1):
            a = k - i + 1
            bars[k].age = a
            bars[k].age_bucket = age_bucket(a)
            if run_len == 1:
                bars[k].position = "Only"
            elif a == 1:
                bars[k].position = "First"
            elif a == run_len:
                bars[k].position = "Last"
            elif a <= math.ceil(run_len * 0.33):
                bars[k].position = "Early"
            elif a >= math.ceil(run_len * 0.67):
                bars[k].position = "Late"
            else:
                bars[k].position = "Middle"
        i = j + 1


def infer_instrument(path: str) -> str:
    p = path.replace("\\", "/").upper()
    for sym in ("6E", "NQ", "CL", "ES"):
        if f"/{sym}/" in p or os.path.basename(p).upper().startswith(sym):
            return sym
    return os.path.basename(os.path.dirname(path)) or "Unknown"


def session_from_time(time_value: str, fallback: str) -> str:
    s = str(time_value or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if match:
        return match.group(0)
    match = re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s)
    if match:
        return match.group(0)
    return fallback


def relative_value(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def compute_market_fields(bars: List[Bar]) -> None:
    prior_by_session: Dict[str, Bar] = {}
    volumes_by_session: Dict[str, List[float]] = defaultdict(list)
    rolling_window = 20
    for b in bars:
        b.has_ohlc = b.open is not None and b.high is not None and b.low is not None and b.close is not None
        b.has_volume = b.volume is not None
        b.has_volume_polarity = bool(str(b.volume_polarity or "").strip())
        if b.has_ohlc:
            b.bar_range = b.high - b.low
            b.body = abs(b.close - b.open)
            b.close_location = (b.close - b.low) / max(b.high - b.low, 1e-12)
        key = b.session_key or session_from_time(b.time, b.source)
        b.session_key = key
        prior = prior_by_session.get(key)
        if b.has_ohlc:
            if prior and prior.close is not None:
                b.true_range = max(
                    b.high - b.low,
                    abs(b.high - prior.close),
                    abs(b.low - prior.close),
                )
            else:
                b.true_range = b.bar_range
        if prior:
            b.range_relative_to_previous = relative_value(b.bar_range, prior.bar_range)
            b.body_relative_to_previous = relative_value(b.body, prior.body)
            b.true_range_relative_to_previous = relative_value(b.true_range, prior.true_range)
            b.volume_relative_to_previous = relative_value(b.volume, prior.volume)
        seen_volumes = volumes_by_session[key]
        if b.volume is not None:
            session_mean_so_far = mean(seen_volumes + [b.volume])
            b.volume_relative_to_session_mean = relative_value(b.volume, session_mean_so_far)
            if len(seen_volumes) >= 5:
                rolling_values = seen_volumes[-rolling_window:]
                b.volume_relative_to_rolling_mean = relative_value(b.volume, mean(rolling_values))
            seen_volumes.append(b.volume)
        b.volume_expansion_flag = b.volume_relative_to_previous > 1 if b.volume_relative_to_previous is not None else None
        b.volume_contraction_flag = b.volume_relative_to_previous < 1 if b.volume_relative_to_previous is not None else None
        b.range_expansion_flag = b.range_relative_to_previous > 1 if b.range_relative_to_previous is not None else None
        b.range_contraction_flag = b.range_relative_to_previous < 1 if b.range_relative_to_previous is not None else None
        prior_by_session[key] = b


def read_csv(path: str) -> List[Bar]:
    instrument = infer_instrument(path)
    bars: List[Bar] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        c_time = find_col(headers, ["Time", "DateTime", "Timestamp", "BarTime"])
        c_inst = find_col(headers, ["Instrument", "Symbol"])
        c_session = find_col(headers, ["SessionDate"])
        c_open = find_col(headers, ["Open", "O"])
        c_high = find_col(headers, ["High", "H"])
        c_low = find_col(headers, ["Low", "L"])
        c_close = find_col(headers, ["Close", "C"])
        c_volume = find_col(headers, ["Volume", "Vol", "V"])
        c_pol = find_col(headers, ["VolumePolarity", "Polarity", "BarPolarity"])
        c_part = find_col(headers, ["ParticipationState"])
        c_exp = find_col(headers, ["ExpansionState"])
        c_comp = find_col(headers, ["CompressionState"])
        c_diss = find_col(headers, ["DissipationState"])
        c_acc = find_col(headers, ["AcceptanceState"])
        c_flags = find_col(headers, ["EvidenceFlags", "Flags"])
        for idx, row in enumerate(r):
            b = Bar(
                instrument=val(row, c_inst, instrument) or instrument,
                source=path,
                rownum=idx,
                time=val(row, c_time),
                close=safe_float(val(row, c_close)),
                volume_polarity=val(row, c_pol),
                participation_state=val(row, c_part),
                expansion_state=val(row, c_exp),
                compression_state=val(row, c_comp),
                dissipation_state=val(row, c_diss),
                acceptance_state=val(row, c_acc),
                flags=split_flags(val(row, c_flags)),
                open=safe_float(val(row, c_open)),
                high=safe_float(val(row, c_high)),
                low=safe_float(val(row, c_low)),
                volume=safe_float(val(row, c_volume)),
                session_key=val(row, c_session) or session_from_time(val(row, c_time), path),
            )
            b.family = assign_family(b)
            bars.append(b)
    return bars


def finalize_bars(bars: List[Bar]) -> None:
    compute_market_fields(bars)
    for i, b in enumerate(bars):
        b.path3 = tuple(x.family for x in bars[max(0, i - 2): i + 1])
        b.path4 = tuple(x.family for x in bars[max(0, i - 3): i + 1])
        b.path5 = tuple(x.family for x in bars[max(0, i - 4): i + 1])
        if len(b.path5) == 5:
            b.archetype = primary_archetype(b.path5)
            b.state = assign_structural_state(b.path5, b.archetype)
        else:
            b.archetype = "Unclassified"
            b.state = "MixedStructure"
    assign_run_age(bars)


def collect_inputs(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in inputs:
        if os.path.isdir(x):
            for root, _, files in os.walk(x):
                for fn in files:
                    if fn.lower().endswith(".csv"):
                        out.append(os.path.join(root, fn))
        elif os.path.isfile(x):
            out.append(x)
        else:
            raise FileNotFoundError(x)
    return sorted(out)


def directional_return(bars: List[Bar], idx: int, horizon: int = 5) -> Optional[float]:
    if idx + horizon >= len(bars):
        return None
    c0 = bars[idx].close
    c1 = bars[idx + horizon].close
    if c0 is None or c1 is None:
        return None
    pol = bars[idx].volume_polarity.strip().lower()
    if pol == "black":
        return c1 - c0
    if pol == "red":
        return c0 - c1
    return None


def build_lifecycle_rows(bars: List[Bar]) -> List[LifeCycleRow]:
    counts: Counter[Tuple[str, str]] = Counter()
    survive: Dict[int, Counter[Tuple[str, str]]] = {lag: Counter() for lag in LAGS}
    death_targets: Counter[Tuple[str, str, str]] = Counter()
    death_totals: Counter[Tuple[str, str]] = Counter()
    outcomes: Dict[Tuple[str, str], Outcome] = defaultdict(Outcome)

    for i, b in enumerate(bars):
        key = (b.state, b.age_bucket)
        counts[key] += 1
        outcomes[key].add(directional_return(bars, i, 5))
        for lag in LAGS:
            if i + lag < len(bars) and bars[i + lag].state == b.state:
                survive[lag][key] += 1
        if i + 1 < len(bars) and bars[i + 1].state != b.state:
            death_totals[key] += 1
            death_targets[(b.state, b.age_bucket, bars[i + 1].state)] += 1

    temp: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, n in counts.items():
        st, age = key
        p1 = survive[1][key] / n if n else 0.0
        p3 = survive[3][key] / n if n else 0.0
        p5 = survive[5][key] / n if n else 0.0
        dom_target = "N/A"
        dom_prob = 0.0
        if death_totals[key] > 0:
            candidates = [
                (target, death_targets[(st, age, target)])
                for target in STRUCTURAL_STATES
                if death_targets[(st, age, target)] > 0
            ]
            if candidates:
                dom_target, dom_n = max(candidates, key=lambda x: x[1])
                dom_prob = dom_n / death_totals[key]
        o = outcomes[key]
        temp[key] = {
            "state": st, "age": age, "count": n,
            "p1": p1, "h1": 1.0 - p1,
            "p3": p3, "h3": 1.0 - p3,
            "p5": p5, "h5": 1.0 - p5,
            "cont": o.cont_rate, "fail": o.fail_rate, "skew": o.skew,
            "mean_dr": o.mean_dr,
            "dom_target": dom_target, "dom_prob": dom_prob,
        }

    phases: Dict[Tuple[str, str], str] = {}
    for st in STRUCTURAL_STATES:
        keys = sorted([k for k in temp if k[0] == st and temp[k]["count"] > 0], key=lambda k: age_sort_key(k[1]))
        if not keys:
            continue
        valid = [k for k in keys if temp[k]["count"] >= MIN_COUNT]
        if not valid:
            for k in keys:
                phases[k] = "Insufficient"
            continue
        h_values = [temp[k]["h1"] for k in valid]
        min_h = min(h_values)
        max_h = max(h_values)
        min_key = min(valid, key=lambda k: (temp[k]["h1"], age_sort_key(k[1])))
        max_key = max(valid, key=lambda k: (temp[k]["h1"], -age_sort_key(k[1])))
        min_order = age_sort_key(min_key[1])

        if len(valid) == 1:
            for k in keys:
                phases[k] = "Birth" if k in valid else "Insufficient"
            continue

        for k in keys:
            if temp[k]["count"] < MIN_COUNT:
                phases[k] = "Insufficient"
                continue
            age = k[1]
            h = temp[k]["h1"]
            order = age_sort_key(age)
            if age == "1" and h >= 0.80:
                phases[k] = "Terminal"
            elif order == 0:
                phases[k] = "Birth"
            elif k == min_key or h <= min_h + 0.05:
                phases[k] = "Maturity"
            elif order < min_order:
                phases[k] = "Growth"
            elif order > min_order:
                if h >= max_h - 0.05 or age in ("11-20", "21+"):
                    phases[k] = "Terminal"
                else:
                    phases[k] = "Decay"
            else:
                phases[k] = "Growth"
        for k in keys:
            phases.setdefault(k, "Insufficient")

    rows: List[LifeCycleRow] = []
    for key, x in temp.items():
        rows.append(LifeCycleRow(
            state=x["state"], age=x["age"], count=x["count"],
            persistence1=x["p1"], hazard1=x["h1"],
            persistence3=x["p3"], hazard3=x["h3"],
            persistence5=x["p5"], hazard5=x["h5"],
            continuation=x["cont"], failure=x["fail"], skew=x["skew"], mean_dr=x["mean_dr"],
            dominant_death_target=x["dom_target"], dominant_death_prob=x["dom_prob"],
            phase=phases.get(key, "Insufficient"),
        ))
    rows.sort(key=lambda r: (STRUCTURAL_STATES.index(r.state), age_sort_key(r.age)))
    return rows


def load_results(inputs: Sequence[str]) -> List[InstrumentResult]:
    by_inst: Dict[str, List[Bar]] = defaultdict(list)
    sources: Dict[str, List[str]] = defaultdict(list)
    for p in collect_inputs(inputs):
        bars = read_csv(p)
        if not bars:
            continue
        inst = bars[0].instrument
        by_inst[inst].extend(bars)
        sources[inst].append(p)
    results: List[InstrumentResult] = []
    for inst, bars in by_inst.items():
        bars.sort(key=lambda b: (b.time, b.rownum))
        finalize_bars(bars)
        rows = build_lifecycle_rows(bars)
        results.append(InstrumentResult(inst, bars, sources[inst], rows))
    results.sort(key=lambda r: r.instrument)
    return results


def write_instrument_report(r: InstrumentResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, r.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"StructuralLifeCycle_{r.instrument}.txt")
    state_counts = Counter(b.state for b in r.bars)
    phase_counts = Counter(row.phase for row in r.rows if row.count >= MIN_COUNT)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"APVA Structural Life Cycle Study v0.1 - {r.instrument}\n")
        f.write("=" * 64 + "\n\n")
        f.write("Diagnostics\n===========\n")
        f.write(f"Instrument: {r.instrument}\nRows: {len(r.bars)}\nSources:\n")
        for s in r.source_paths:
            f.write(f"  {s}\n")
        f.write("\nStructural State Counts\n=======================\n")
        for st in STRUCTURAL_STATES:
            f.write(f"{st:<28} {state_counts[st]:8d} {pct(state_counts[st] / len(r.bars) if r.bars else 0):>10}\n")
        f.write("\nLife Cycle Phase Counts\n=======================\n")
        for ph, n in phase_counts.most_common():
            f.write(f"{ph:<16} {n:8d}\n")
        f.write("\nLife Cycle Table\n================\n")
        f.write(f"{'State':<28} {'Age':<6} {'N':>8} {'P1':>9} {'H1':>9} {'P3':>9} {'H3':>9} {'P5':>9} {'H5':>9} {'Cont':>9} {'Fail':>9} {'Skew':>9} {'MeanDR':>12} {'DeathTarget':<28} {'DeathProb':>10} {'Phase':<14}\n")
        for row in r.rows:
            f.write(f"{row.state:<28} {row.age:<6} {row.count:8d} {pct(row.persistence1):>9} {pct(row.hazard1):>9} {pct(row.persistence3):>9} {pct(row.hazard3):>9} {pct(row.persistence5):>9} {pct(row.hazard5):>9} {pct(row.continuation):>9} {pct(row.failure):>9} {pct(row.skew):>9} {fmt(row.mean_dr, 5):>12} {row.dominant_death_target:<28} {pct(row.dominant_death_prob):>10} {row.phase:<14}\n")
        f.write("\nState Life Cycle Summary\n========================\n")
        for st in STRUCTURAL_STATES:
            state_rows = [x for x in r.rows if x.state == st and x.count >= MIN_COUNT]
            if not state_rows:
                continue
            min_row = min(state_rows, key=lambda x: (x.hazard1, age_sort_key(x.age)))
            max_row = max(state_rows, key=lambda x: (x.hazard1, -age_sort_key(x.age)))
            best_skew = max(state_rows, key=lambda x: x.skew)
            worst_skew = min(state_rows, key=lambda x: x.skew)
            f.write(f"\n{st}\n" + "-" * len(st) + "\n")
            f.write(f"Most stable age: {min_row.age} | H1={pct(min_row.hazard1)} | P1={pct(min_row.persistence1)} | Phase={min_row.phase}\n")
            f.write(f"Most fragile age: {max_row.age} | H1={pct(max_row.hazard1)} | P1={pct(max_row.persistence1)} | Phase={max_row.phase}\n")
            f.write(f"Best skew age: {best_skew.age} | Skew={pct(best_skew.skew)} | MeanDR={fmt(best_skew.mean_dr,5)}\n")
            f.write(f"Worst skew age: {worst_skew.age} | Skew={pct(worst_skew.skew)} | MeanDR={fmt(worst_skew.mean_dr,5)}\n")
            phases = defaultdict(list)
            for x in state_rows:
                phases[x.phase].append(x.age)
            for ph in ("Birth", "Growth", "Maturity", "Decay", "Terminal", "Insufficient"):
                if phases.get(ph):
                    f.write(f"{ph:<12}: {', '.join(phases[ph])}\n")
        f.write("\nMechanical Research Notes\n=========================\n")
        f.write("- Birth/Growth/Maturity/Decay/Terminal are mechanical labels inferred from t+1 hazard by state-age bucket.\n")
        f.write("- Maturity generally corresponds to the lowest hazard age bucket for that structural state.\n")
        f.write("- Terminal generally corresponds to high hazard or late/rising-hazard age buckets.\n")
        f.write("- DeathTarget is the dominant next-bar state when the current state exits.\n")
        f.write("- No price outcome is used to assign phase. Outcome is reported separately.\n")
    return out_path


def aggregate_rows(results: List[InstrumentResult]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_inst: Dict[str, Dict[Tuple[str, str], LifeCycleRow]] = {r.instrument: {(x.state, x.age): x for x in r.rows} for r in results}
    instruments = [r.instrument for r in results]
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for st in STRUCTURAL_STATES:
        for age in AGE_BUCKETS:
            key = (st, age)
            rows = [by_inst[inst].get(key) for inst in instruments if by_inst[inst].get(key)]
            rows_valid = [x for x in rows if x and x.count >= MIN_COUNT]
            if not rows:
                continue
            phases = Counter(x.phase for x in rows_valid)
            phase = phases.most_common(1)[0][0] if phases else "Insufficient"
            out[key] = {
                "rows": {inst: by_inst[inst].get(key) for inst in instruments},
                "validn": len(rows_valid),
                "mean_h1": mean([x.hazard1 for x in rows_valid]),
                "mean_p1": mean([x.persistence1 for x in rows_valid]),
                "mean_h3": mean([x.hazard3 for x in rows_valid]),
                "mean_h5": mean([x.hazard5 for x in rows_valid]),
                "mean_cont": mean([x.continuation for x in rows_valid]),
                "mean_fail": mean([x.failure for x in rows_valid]),
                "mean_skew": mean([x.skew for x in rows_valid]),
                "mean_dr": mean([x.mean_dr for x in rows_valid]),
                "phase": phase,
            }
    return out


def write_aggregate_report(results: List[InstrumentResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "StructuralLifeCycle")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "StructuralLifeCycle_All.txt")
    instruments = [r.instrument for r in results]
    agg = aggregate_rows(results)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("APVA Structural Life Cycle Study v0.1 - Aggregate\n")
        f.write("=" * 62 + "\n\n")
        f.write("Instruments: " + ", ".join(instruments) + "\n")
        f.write(f"Minimum per-instrument count for phase inference: {MIN_COUNT}\n\n")
        f.write("Aggregate Life Cycle Table\n==========================\n")
        f.write(f"{'State':<28} {'Age':<6}")
        for inst in instruments:
            f.write(f" {inst+'_N':>8} {inst+'_H1':>9} {inst+'_Skew':>9} {inst+'_Phase':>12}")
        f.write(f" {'ValidN':>7} {'MeanP1':>9} {'MeanH1':>9} {'MeanH3':>9} {'MeanH5':>9} {'MeanSkew':>10} {'MeanDR':>12} {'Phase':<14}\n")
        for st in STRUCTURAL_STATES:
            for age in AGE_BUCKETS:
                key = (st, age)
                if key not in agg:
                    continue
                a = agg[key]
                line = f"{st:<28} {age:<6}"
                for inst in instruments:
                    row = a["rows"].get(inst)
                    if row:
                        line += f" {row.count:8d} {pct(row.hazard1):>9} {pct(row.skew):>9} {row.phase:>12}"
                    else:
                        line += f" {0:8d} {'N/A':>9} {'N/A':>9} {'N/A':>12}"
                line += f" {a['validn']:7d} {pct(a['mean_p1']):>9} {pct(a['mean_h1']):>9} {pct(a['mean_h3']):>9} {pct(a['mean_h5']):>9} {pct(a['mean_skew']):>10} {fmt(a['mean_dr'],5):>12} {a['phase']:<14}\n"
                f.write(line)
        f.write("\nState Life Cycle Summary\n========================\n")
        for st in STRUCTURAL_STATES:
            rows = [(age, agg[(st, age)]) for age in AGE_BUCKETS if (st, age) in agg and agg[(st, age)]["validn"] >= 2]
            if not rows:
                continue
            min_age, min_a = min(rows, key=lambda x: (x[1]["mean_h1"], age_sort_key(x[0])))
            max_age, max_a = max(rows, key=lambda x: (x[1]["mean_h1"], -age_sort_key(x[0])))
            best_age, best_a = max(rows, key=lambda x: x[1]["mean_skew"])
            worst_age, worst_a = min(rows, key=lambda x: x[1]["mean_skew"])
            f.write(f"\n{st}\n" + "-" * len(st) + "\n")
            f.write(f"Most stable age: {min_age} | MeanH1={pct(min_a['mean_h1'])} | Phase={min_a['phase']}\n")
            f.write(f"Most fragile age: {max_age} | MeanH1={pct(max_a['mean_h1'])} | Phase={max_a['phase']}\n")
            f.write(f"Best skew age: {best_age} | MeanSkew={pct(best_a['mean_skew'])} | MeanDR={fmt(best_a['mean_dr'],5)}\n")
            f.write(f"Worst skew age: {worst_age} | MeanSkew={pct(worst_a['mean_skew'])} | MeanDR={fmt(worst_a['mean_dr'],5)}\n")
            phases = defaultdict(list)
            for age, a in rows:
                phases[a["phase"]].append(age)
            for ph in ("Birth", "Growth", "Maturity", "Decay", "Terminal", "Insufficient"):
                if phases.get(ph):
                    f.write(f"{ph:<12}: {', '.join(phases[ph])}\n")
        f.write("\nAggregate Rankings\n==================\n")
        valid_items = [(st, age, a) for (st, age), a in agg.items() if a["validn"] >= 2]
        f.write("\n1. Most stable replicated state-ages\n....................................\n")
        for i, (st, age, a) in enumerate(sorted(valid_items, key=lambda x: x[2]["mean_h1"])[:25], 1):
            f.write(f"{i:3d}. {st} Age={age} | MeanH1={pct(a['mean_h1'])} | Phase={a['phase']}\n")
        f.write("\n2. Most fragile replicated state-ages\n.....................................\n")
        for i, (st, age, a) in enumerate(sorted(valid_items, key=lambda x: x[2]["mean_h1"], reverse=True)[:25], 1):
            f.write(f"{i:3d}. {st} Age={age} | MeanH1={pct(a['mean_h1'])} | Phase={a['phase']}\n")
        f.write("\n3. Best replicated state-age outcome skew\n.........................................\n")
        for i, (st, age, a) in enumerate(sorted(valid_items, key=lambda x: x[2]["mean_skew"], reverse=True)[:25], 1):
            f.write(f"{i:3d}. {st} Age={age} | MeanSkew={pct(a['mean_skew'])} | MeanDR={fmt(a['mean_dr'],5)} | Phase={a['phase']}\n")
        f.write("\n4. Worst replicated state-age outcome skew\n..........................................\n")
        for i, (st, age, a) in enumerate(sorted(valid_items, key=lambda x: x[2]["mean_skew"])[:25], 1):
            f.write(f"{i:3d}. {st} Age={age} | MeanSkew={pct(a['mean_skew'])} | MeanDR={fmt(a['mean_dr'],5)} | Phase={a['phase']}\n")
        f.write("\nMechanical Research Notes\n=========================\n")
        f.write("- Life-cycle phase inference is derived from t+1 hazard within each structural state.\n")
        f.write("- The phase labels are descriptive and should not be treated as trade instructions.\n")
        f.write("- A useful next step is to validate whether phase improves transition and outcome separation beyond state+age alone.\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="APVA Structural Life Cycle Study v0.1")
    ap.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    ap.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = ap.parse_args(argv)
    results = load_results(args.inputs)
    if not results:
        raise SystemExit("No input rows loaded.")
    for r in results:
        write_instrument_report(r, args.out_root)
    out = write_aggregate_report(results, args.out_root)
    print(f"Wrote StructuralLifeCycle reports under {args.out_root}")
    print(f"Aggregate: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
