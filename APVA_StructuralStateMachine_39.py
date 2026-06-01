#!/usr/bin/env python3
"""
APVA Structural State Machine Study v0.1

Purpose:
    Collapse APVA family paths into finite structural states and study the
    resulting state-machine behavior.

Design goals:
    - Standard library only.
    - No trading rules, no optimization, no fitting.
    - Works with one or more CSV evidence files.
    - Reuses Study 30 family definitions mechanically.
    - Builds structural states from path signatures rather than exact paths.

Outputs:
    Per instrument:
        Evidence/Output/<instrument>/StructuralStateMachine_<instrument>.txt
    Aggregate:
        Evidence/Output/StructuralStateMachine/StructuralStateMachine_All.txt

Usage examples:
    python APVA_StructuralStateMachine_39.py Evidence/6E.csv Evidence/NQ.csv Evidence/CL.csv
    python APVA_StructuralStateMachine_39.py --out-root Evidence/Output *.csv
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
LAGS = (1, 2, 3, 5)
PATH_LENGTHS = (3, 4, 5)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def truthy(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "x", "on"}


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
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


def instrument_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"\b(6E|NQ|CL|ES|YM|RTY|GC|SI|BTC|ETH)\b", base, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    parts = re.split(r"[_\- .]+", base)
    return parts[0].upper() if parts else "UNKNOWN"


# ---------------------------------------------------------------------------
# Column mapping and row model
# ---------------------------------------------------------------------------

@dataclass
class ColumnMap:
    instrument: Optional[str] = None
    time: Optional[str] = None
    close: Optional[str] = None
    volume_polarity: Optional[str] = None
    accepted: Optional[str] = None
    compressed: Optional[str] = None
    dissipating: Optional[str] = None
    expanding: Optional[str] = None
    peak: Optional[str] = None
    climactic: Optional[str] = None
    participation_state: Optional[str] = None
    acceptance_state: Optional[str] = None
    compression_state: Optional[str] = None
    dissipation_state: Optional[str] = None
    expansion_state: Optional[str] = None
    population_b_fields: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]] = (None, None, None, None)


def build_colmap(headers: Sequence[str]) -> ColumnMap:
    return ColumnMap(
        instrument=find_col(headers, ["Instrument", "Symbol", "Market"]),
        time=find_col(headers, ["Time", "DateTime", "Timestamp", "BarTime"]),
        close=find_col(headers, ["Close", "ClosePrice", "Last"]),
        volume_polarity=find_col(headers, ["VolumePolarity", "Polarity", "Volume Color", "VolumeColor"]),
        accepted=find_col(headers, ["Accepted", "Flag_Accepted", "CurrentFlag_Accepted"]),
        compressed=find_col(headers, ["Compressed", "Flag_Compressed", "CurrentFlag_Compressed"]),
        dissipating=find_col(headers, ["Dissipating", "Flag_Dissipating", "CurrentFlag_Dissipating"]),
        expanding=find_col(headers, ["Expanding", "Flag_Expanding", "CurrentFlag_Expanding"]),
        peak=find_col(headers, ["Peak", "Flag_Peak", "CurrentFlag_Peak"]),
        climactic=find_col(headers, ["Climactic", "Flag_Climactic", "CurrentFlag_Climactic"]),
        participation_state=find_col(headers, ["ParticipationState", "Participation State"]),
        acceptance_state=find_col(headers, ["AcceptanceState", "Acceptance State"]),
        compression_state=find_col(headers, ["CompressionState", "Compression State"]),
        dissipation_state=find_col(headers, ["DissipationState", "Dissipation State"]),
        expansion_state=find_col(headers, ["ExpansionState", "Expansion State"]),
        population_b_fields=(
            find_col(headers, ["DissipationContained"]),
            find_col(headers, ["Lateral"]),
            find_col(headers, ["Mature"]),
            find_col(headers, ["Aligned"]),
        ),
    )


@dataclass
class Bar:
    source_path: str
    instrument: str
    index: int
    time: str
    close: Optional[float]
    volume_polarity: str
    flags: Dict[str, bool]
    states: Dict[str, str]
    family: str = "N"
    paths: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
    structural_state: str = "MixedStructure"
    structural_reason: str = ""
    population_b: bool = False


def infer_flags(row: Dict[str, str], cm: ColumnMap) -> Dict[str, bool]:
    acc_state = val(row, cm.acceptance_state).strip()
    comp_state = val(row, cm.compression_state).strip()
    diss_state = val(row, cm.dissipation_state).strip()
    exp_state = val(row, cm.expansion_state).strip()
    part_state = val(row, cm.participation_state).strip()

    accepted = truthy(val(row, cm.accepted)) or acc_state.lower() == "accepted"
    compressed = truthy(val(row, cm.compressed)) or comp_state.lower() not in {"", "absent", "none", "false", "0"}
    dissipating = truthy(val(row, cm.dissipating)) or diss_state.lower() not in {"", "absent", "none", "false", "0"}
    expanding = truthy(val(row, cm.expanding)) or exp_state.lower() in {"local", "strong", "climactic", "expanding"}
    peak = truthy(val(row, cm.peak)) or part_state.lower() == "peak"
    climactic = truthy(val(row, cm.climactic)) or part_state.lower() == "climactic" or exp_state.lower() == "climactic"

    return {
        "Accepted": accepted,
        "Compressed": compressed,
        "Dissipating": dissipating,
        "Expanding": expanding,
        "Peak": peak,
        "Climactic": climactic,
    }


def assign_family(flags: Dict[str, bool]) -> str:
    accepted = flags["Accepted"]
    compressed = flags["Compressed"]
    dissipating = flags["Dissipating"]
    expanding = flags["Expanding"]
    peak = flags["Peak"]
    climactic = flags["Climactic"]

    # Study 30 precedence: A, then B, then C, then D, then N.
    if accepted and compressed:
        return "A"
    if compressed and peak:
        return "B"
    if (accepted and peak) or (accepted and climactic) or (accepted and dissipating and peak):
        return "C"
    if accepted and expanding:
        return "D"
    return "N"


def infer_population_b(row: Dict[str, str], cm: ColumnMap) -> bool:
    cols = cm.population_b_fields
    if not all(cols):
        return False
    return all(truthy(val(row, c)) for c in cols)


def read_bars(path: str) -> List[Bar]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        cm = build_colmap(headers)
        fallback_instr = instrument_from_path(path)
        bars: List[Bar] = []
        for i, row in enumerate(rdr):
            instr = val(row, cm.instrument, fallback_instr).strip() or fallback_instr
            instr = instr.split()[0].upper() if instr else fallback_instr
            flags = infer_flags(row, cm)
            states = {
                "ParticipationState": val(row, cm.participation_state).strip(),
                "AcceptanceState": val(row, cm.acceptance_state).strip(),
                "CompressionState": val(row, cm.compression_state).strip(),
                "DissipationState": val(row, cm.dissipation_state).strip(),
                "ExpansionState": val(row, cm.expansion_state).strip(),
            }
            b = Bar(
                source_path=path,
                instrument=instr,
                index=i,
                time=val(row, cm.time, str(i)).strip(),
                close=safe_float(val(row, cm.close)),
                volume_polarity=val(row, cm.volume_polarity).strip(),
                flags=flags,
                states=states,
                population_b=infer_population_b(row, cm),
            )
            b.family = assign_family(flags)
            bars.append(b)
    assign_paths_and_states(bars)
    return bars


# ---------------------------------------------------------------------------
# Structural signatures and state assignment
# ---------------------------------------------------------------------------

def class_of_family(f: str) -> str:
    if f in {"A", "B"}:
        return "Constructive"
    if f in {"C", "D"}:
        return "Destructive"
    return "Neutral"


def contains_subseq(path: Sequence[str], subseq: Sequence[str]) -> bool:
    n = len(subseq)
    if n == 0 or len(path) < n:
        return False
    return any(tuple(path[i:i+n]) == tuple(subseq) for i in range(len(path)-n+1))


def alternating_dn(path: Sequence[str]) -> bool:
    if len(path) < 3:
        return False
    if not all(f in {"D", "N"} for f in path):
        return False
    return all(path[i] != path[i-1] for i in range(1, len(path)))


def path_signature(path: Sequence[str]) -> Dict[str, Any]:
    c = Counter(path)
    L = len(path)
    start, end = path[0], path[-1]
    start_cls, end_cls = class_of_family(start), class_of_family(end)
    repeated = {f: any(path[i] == f and path[i-1] == f for i in range(1, L)) for f in FAMILIES}
    max_count = max(c.values()) if c else 0
    doms = [f for f, n in c.items() if n == max_count]
    dominant = doms[0] if len(doms) == 1 else "Mixed"
    transition_count = sum(1 for i in range(1, L) if path[i] != path[i-1])
    repeat_count = (L - 1) - transition_count
    unique_count = len(c)

    if unique_count == 1:
        persistence = "AllSame"
    elif alternating_dn(path):
        persistence = "Alternating"
    elif max_count >= 2:
        persistence = "RepeatedDominantFamily"
    else:
        persistence = "Mixed"

    sig = {
        "StartFamily": start,
        "EndFamily": end,
        "StartClass": start_cls,
        "EndClass": end_cls,
        "DirectionClass": f"{start_cls}To{end_cls}",
        "DominantFamily": dominant,
        "PersistenceClass": persistence,
        "ConstructiveCount": c["A"] + c["B"],
        "DestructiveCount": c["C"] + c["D"],
        "NeutralCount": c["N"],
        "TransitionCount": transition_count,
        "RepeatCount": repeat_count,
        "UniqueFamilyCount": unique_count,
        "ContainsA": c["A"] > 0,
        "ContainsB": c["B"] > 0,
        "ContainsC": c["C"] > 0,
        "ContainsD": c["D"] > 0,
        "ContainsN": c["N"] > 0,
        "ContainsRepeatedA": repeated["A"],
        "ContainsRepeatedB": repeated["B"],
        "ContainsRepeatedC": repeated["C"],
        "ContainsRepeatedD": repeated["D"],
        "ContainsRepeatedN": repeated["N"],
        "ContainsCB": contains_subseq(path, ("C", "B")),
        "ContainsCCB": contains_subseq(path, ("C", "C", "B")),
        "ContainsCC": contains_subseq(path, ("C", "C")),
        "ContainsCCC": contains_subseq(path, ("C", "C", "C")),
        "ContainsBN": contains_subseq(path, ("B", "N")),
        "ContainsCN": contains_subseq(path, ("C", "N")),
        "ContainsDN": contains_subseq(path, ("D", "N")),
        "ContainsNDN": contains_subseq(path, ("N", "D", "N")),
        "ContainsDND": contains_subseq(path, ("D", "N", "D")),
        "ContainsNNN": contains_subseq(path, ("N", "N", "N")),
    }
    sig["CompressionLike"] = bool(sig["ContainsB"] or sig["ContainsBN"] or sig["ContainsRepeatedN"] or (end == "N" and (sig["ContainsB"] or sig["ContainsC"])))
    sig["ExhaustionLike"] = bool(sig["ContainsC"] and (sig["ContainsRepeatedC"] or end == "C"))
    sig["RecoveryLike"] = bool(sig["ContainsCB"] or sig["ContainsCCB"] or sig["DirectionClass"] == "DestructiveToConstructive")
    sig["DecayLike"] = bool(end == "N" and (sig["ContainsB"] or sig["ContainsC"]))
    sig["ReassertionLike"] = bool(alternating_dn(path) or contains_subseq(path, ("N", "D")) or sig["ContainsDND"] or sig["ContainsNDN"])
    sig["NeutralDriftLike"] = bool(c["N"] >= L - 1 or sig["ContainsNNN"])
    return sig


def assign_structural_state(path: Sequence[str]) -> Tuple[str, str]:
    sig = path_signature(path)
    # Primary precedence is deliberately mechanical. It should be audited with data.
    if sig["RecoveryLike"]:
        return "RecoveryResolution", "RecoveryLike"
    if sig["ExhaustionLike"] and (sig["ContainsCCC"] or sig["ContainsRepeatedC"]):
        return "ExhaustionPersistence", "ExhaustionLike+RepeatedC"
    if sig["CompressionLike"] and (sig["ContainsB"] or sig["ContainsBN"]):
        return "CompressionProcessing", "CompressionLike+B/BN"
    if sig["NeutralDriftLike"]:
        return "NeutralProcessing", "NeutralDriftLike"
    if sig["ReassertionLike"]:
        return "ReassertionProcessing", "ReassertionLike"
    if sig["DecayLike"]:
        return "DecayToNeutral", "DecayLike"
    if sig["DestructiveCount"] >= 2 and sig["ConstructiveCount"] == 0:
        return "DestructiveRotation", "DestructiveCount>=2"
    if sig["DirectionClass"] in {"NeutralToConstructive", "DestructiveToConstructive"}:
        return "ConstructiveEmergence", sig["DirectionClass"]
    return "MixedStructure", "No primary signature"


def assign_paths_and_states(bars: List[Bar]) -> None:
    fams = [b.family for b in bars]
    for i, b in enumerate(bars):
        for L in PATH_LENGTHS:
            if i >= L - 1:
                b.paths[L] = tuple(fams[i-L+1:i+1])
        # Prefer longest available path for structural-state assignment.
        for L in reversed(PATH_LENGTHS):
            if L in b.paths:
                b.structural_state, b.structural_reason = assign_structural_state(b.paths[L])
                break


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

@dataclass
class OutcomeStats:
    count: int = 0
    mean_dr: float = 0.0
    median_dr: float = 0.0
    continuation_rate: float = 0.0
    failure_rate: float = 0.0
    flat_rate: float = 0.0


def directional_return(bars: Sequence[Bar], i: int, horizon: int = 5) -> Optional[float]:
    if i + horizon >= len(bars):
        return None
    c0 = bars[i].close
    c1 = bars[i + horizon].close
    if c0 is None or c1 is None:
        return None
    pol = (bars[i].volume_polarity or "").strip().lower()
    if pol == "black":
        return c1 - c0
    if pol == "red":
        return c0 - c1
    return None


def outcome_stats(bars: Sequence[Bar], indices: Iterable[int], horizon: int = 5) -> OutcomeStats:
    vals: List[float] = []
    for i in indices:
        r = directional_return(bars, i, horizon)
        if r is not None:
            vals.append(r)
    if not vals:
        return OutcomeStats()
    cont = sum(1 for x in vals if x > 0)
    fail = sum(1 for x in vals if x < 0)
    flat = len(vals) - cont - fail
    return OutcomeStats(
        count=len(vals),
        mean_dr=mean(vals),
        median_dr=median(vals),
        continuation_rate=cont / len(vals),
        failure_rate=fail / len(vals),
        flat_rate=flat / len(vals),
    )


def transition_table(labels: Sequence[str], lag: int) -> Dict[Tuple[str, str], int]:
    c: Dict[Tuple[str, str], int] = defaultdict(int)
    for i in range(lag, len(labels)):
        src = labels[i - lag]
        dst = labels[i]
        if src and dst:
            c[(src, dst)] += 1
    return c


def transition_stats(labels: Sequence[str], lag: int) -> List[Dict[str, Any]]:
    trans = transition_table(labels, lag)
    total = sum(trans.values())
    node_counts = Counter(labels)
    n = sum(node_counts.values())
    rows = []
    for (src, dst), cnt in sorted(trans.items()):
        prob = cnt / sum(v for (s, _), v in trans.items() if s == src) if total else 0.0
        expected = total * (node_counts[src] / n) * (node_counts[dst] / n) if n else 0.0
        lift = cnt / expected if expected > 0 else None
        rows.append({"Source": src, "Target": dst, "Count": cnt, "Probability": prob, "Lift": lift})
    return rows


def run_lengths(labels: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    if not labels:
        return out
    cur = labels[0]
    length = 1
    for x in labels[1:]:
        if x == cur:
            length += 1
        else:
            out[cur].append(length)
            cur = x
            length = 1
    out[cur].append(length)
    return out


def state_sequence(bars: Sequence[Bar], population_b_only: bool = False) -> List[str]:
    return [b.structural_state for b in bars if (not population_b_only or b.population_b)]


def valid_indices_for_state(bars: Sequence[Bar], state: str) -> List[int]:
    return [i for i, b in enumerate(bars) if b.structural_state == state]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class InstrumentResult:
    instrument: str
    path: str
    bars: List[Bar]
    state_counts: Counter
    family_counts: Counter
    transition_rows: Dict[int, List[Dict[str, Any]]]
    outcome_by_state: Dict[str, OutcomeStats]
    run_length_by_state: Dict[str, List[int]]


def analyze_instrument(path: str) -> InstrumentResult:
    bars = read_bars(path)
    instr = bars[0].instrument if bars else instrument_from_path(path)
    labels = [b.structural_state for b in bars]
    return InstrumentResult(
        instrument=instr,
        path=path,
        bars=bars,
        state_counts=Counter(labels),
        family_counts=Counter(b.family for b in bars),
        transition_rows={lag: transition_stats(labels, lag) for lag in LAGS},
        outcome_by_state={s: outcome_stats(bars, valid_indices_for_state(bars, s)) for s in STRUCTURAL_STATES},
        run_length_by_state=run_lengths(labels),
    )


def top_rows(rows: List[Dict[str, Any]], key: str, n: int = 25, reverse: bool = True) -> List[Dict[str, Any]]:
    def kval(r: Dict[str, Any]) -> float:
        x = r.get(key)
        if x is None:
            return float("-inf") if reverse else float("inf")
        return float(x)
    return sorted(rows, key=kval, reverse=reverse)[:n]


def write_instrument_report(res: InstrumentResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, res.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"StructuralStateMachine_{res.instrument}.txt")
    total = len(res.bars)
    valid_pol = sum(1 for b in res.bars if (b.volume_polarity or "").lower() in {"black", "red"})

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("APVA Structural State Machine Study v0.1\n")
        f.write("============================================\n\n")
        f.write(f"Instrument: {res.instrument}\n")
        f.write(f"Input: {res.path}\n")
        f.write(f"Total rows: {total}\n")
        f.write(f"Valid polarity rows: {valid_pol}\n\n")

        f.write("Family Counts\n")
        f.write("=============\n")
        for fam in FAMILIES:
            n = res.family_counts.get(fam, 0)
            f.write(f"{fam:>2} {n:8d} {pct(n/total if total else 0):>8}\n")
        f.write("\n")

        f.write("Structural State Counts\n")
        f.write("=======================\n")
        for s in STRUCTURAL_STATES:
            n = res.state_counts.get(s, 0)
            f.write(f"{s:28s} {n:8d} {pct(n/total if total else 0):>8}\n")
        f.write("\n")

        f.write("Outcome By Structural State\n")
        f.write("===========================\n")
        f.write(f"{'State':28s} {'N':>8s} {'MeanDR':>12s} {'MedDR':>12s} {'Cont':>8s} {'Fail':>8s} {'Flat':>8s}\n")
        for s in STRUCTURAL_STATES:
            o = res.outcome_by_state[s]
            f.write(f"{s:28s} {o.count:8d} {o.mean_dr:12.5f} {o.median_dr:12.5f} {pct(o.continuation_rate):>8s} {pct(o.failure_rate):>8s} {pct(o.flat_rate):>8s}\n")
        f.write("\n")

        f.write("Structural State Persistence\n")
        f.write("============================\n")
        f.write(f"{'State':28s} {'Runs':>8s} {'MeanDur':>10s} {'MedDur':>10s} {'MaxDur':>8s}\n")
        for s in STRUCTURAL_STATES:
            runs = res.run_length_by_state.get(s, [])
            f.write(f"{s:28s} {len(runs):8d} {mean(runs):10.2f} {median(runs):10.2f} {(max(runs) if runs else 0):8d}\n")
        f.write("\n")

        for lag in LAGS:
            f.write(f"Top Structural State Transitions Lag {lag}\n")
            f.write("=" * (35 + len(str(lag))) + "\n")
            f.write(f"{'Source':28s} {'Target':28s} {'Count':>8s} {'Prob':>8s} {'Lift':>8s}\n")
            rows = top_rows(res.transition_rows[lag], "Lift", 30)
            for r in rows:
                f.write(f"{r['Source']:28s} {r['Target']:28s} {r['Count']:8d} {pct(r['Probability']):>8s} {fmt(r['Lift'], 3):>8s}\n")
            f.write("\n")

        f.write("Population B Diagnostic\n")
        f.write("=======================\n")
        pb_count = sum(1 for b in res.bars if b.population_b)
        f.write(f"Population B rows: {pb_count}\n")
        if pb_count:
            pb_counts = Counter(b.structural_state for b in res.bars if b.population_b)
            for s in STRUCTURAL_STATES:
                n = pb_counts.get(s, 0)
                if n:
                    f.write(f"{s:28s} {n:8d} {pct(n/pb_count):>8}\n")
        else:
            f.write("Population B columns were missing or no rows qualified.\n")
        f.write("\n")

        f.write("Mechanical Research Notes\n")
        f.write("=========================\n")
        f.write("- Structural states are generated mechanically from family path signatures.\n")
        f.write("- No trading rule, fitting, optimization, or entry/exit policy is used.\n")
        f.write("- Exact path strings are not used as final states; they are compressed into structural labels.\n")
        f.write("- Outcome statistics are descriptive only and anchored on the current bar.\n")
    return out_path


def aggregate_results(results: Sequence[InstrumentResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "StructuralStateMachine")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "StructuralStateMachine_All.txt")

    # Aggregate transition rows by instrument.
    trans_by_instr: Dict[str, Dict[int, Dict[Tuple[str, str], Dict[str, Any]]]] = {}
    for r in results:
        trans_by_instr[r.instrument] = {}
        for lag, rows in r.transition_rows.items():
            trans_by_instr[r.instrument][lag] = {(x["Source"], x["Target"]): x for x in rows}

    instruments = [r.instrument for r in results]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("APVA Structural State Machine Study v0.1 - Aggregate\n")
        f.write("========================================================\n\n")
        f.write("Instruments: " + ", ".join(instruments) + "\n")
        f.write("Structural states are compressed from family-path signatures.\n\n")

        f.write("Aggregate Structural State Table\n")
        f.write("================================\n")
        header = ["State"]
        for inst in instruments:
            header += [f"{inst}_N", f"{inst}_Freq", f"{inst}_Cont", f"{inst}_Fail"]
        header += ["ValidN", "MeanFreq", "MeanCont", "MeanFail"]
        f.write(" ".join(f"{h:>18s}" for h in header) + "\n")
        for s in STRUCTURAL_STATES:
            vals_freq: List[float] = []
            vals_cont: List[float] = []
            vals_fail: List[float] = []
            line = [f"{s:>18s}"]
            for r in results:
                total = len(r.bars)
                n = r.state_counts.get(s, 0)
                freq = n / total if total else 0.0
                o = r.outcome_by_state[s]
                line += [f"{n:18d}", f"{pct(freq):>18s}", f"{pct(o.continuation_rate):>18s}", f"{pct(o.failure_rate):>18s}"]
                if n > 0:
                    vals_freq.append(freq)
                if o.count > 0:
                    vals_cont.append(o.continuation_rate)
                    vals_fail.append(o.failure_rate)
            line += [f"{len(vals_freq):18d}", f"{pct(mean(vals_freq)):>18s}", f"{pct(mean(vals_cont)):>18s}", f"{pct(mean(vals_fail)):>18s}"]
            f.write(" ".join(line) + "\n")
        f.write("\n")

        f.write("Aggregate Transition Table\n")
        f.write("==========================\n")
        for lag in LAGS:
            keys = sorted(set(k for inst in instruments for k in trans_by_instr[inst][lag].keys()))
            f.write(f"\nLag {lag}\n")
            f.write(f"{'Source':28s} {'Target':28s}")
            for inst in instruments:
                f.write(f" {inst+'_N':>8s} {inst+'_Prob':>10s} {inst+'_Lift':>10s}")
            f.write(f" {'ValidN':>7s} {'MeanProb':>10s} {'MeanLift':>10s}\n")
            agg_rows = []
            for src, dst in keys:
                probs = []
                lifts = []
                parts = [f"{src:28s} {dst:28s}"]
                valid = 0
                for inst in instruments:
                    row = trans_by_instr[inst][lag].get((src, dst))
                    if row:
                        valid += 1
                        probs.append(row["Probability"])
                        if row["Lift"] is not None:
                            lifts.append(row["Lift"])
                        parts.append(f" {row['Count']:8d} {pct(row['Probability']):>10s} {fmt(row['Lift'],3):>10s}")
                    else:
                        parts.append(f" {0:8d} {'0.00%':>10s} {'N/A':>10s}")
                mean_prob = mean(probs)
                mean_lift = mean(lifts)
                agg_rows.append((mean_lift, valid, "".join(parts) + f" {valid:7d} {pct(mean_prob):>10s} {fmt(mean_lift,3):>10s}\n"))
            for _, _, line in sorted(agg_rows, key=lambda x: (x[0], x[1]), reverse=True)[:80]:
                f.write(line)
        f.write("\n")

        f.write("Aggregate Rankings\n")
        f.write("==================\n\n")
        # Most common states
        f.write("1. Most common structural states\n")
        f.write("................................\n")
        state_mean_freq = []
        for s in STRUCTURAL_STATES:
            freqs = []
            for r in results:
                total = len(r.bars)
                if total:
                    freqs.append(r.state_counts.get(s, 0) / total)
            state_mean_freq.append((mean(freqs), s))
        for rank, (mf, s) in enumerate(sorted(state_mean_freq, reverse=True), 1):
            f.write(f"{rank:3d}. {s:28s} MeanFreq={pct(mf)}\n")
        f.write("\n")

        f.write("2. Highest continuation structural states\n")
        f.write(".........................................\n")
        cont_rank = []
        for s in STRUCTURAL_STATES:
            vals = [r.outcome_by_state[s].continuation_rate for r in results if r.outcome_by_state[s].count > 0]
            fails = [r.outcome_by_state[s].failure_rate for r in results if r.outcome_by_state[s].count > 0]
            if len(vals) >= 2:
                cont_rank.append((mean(vals), mean(fails), s, len(vals)))
        for rank, (mc, mf, s, v) in enumerate(sorted(cont_rank, reverse=True), 1):
            f.write(f"{rank:3d}. {s:28s} MeanCont={pct(mc)} MeanFail={pct(mf)} ValidN={v}\n")
        f.write("\n")

        f.write("3. Strongest replicated lag-1 transitions\n")
        f.write(".........................................\n")
        lag = 1
        keys = sorted(set(k for inst in instruments for k in trans_by_instr[inst][lag].keys()))
        trans_rank = []
        for key in keys:
            lifts = []
            probs = []
            counts = []
            for inst in instruments:
                row = trans_by_instr[inst][lag].get(key)
                if row and row["Lift"] is not None and row["Count"] >= 20:
                    lifts.append(row["Lift"])
                    probs.append(row["Probability"])
                    counts.append(row["Count"])
            if len(lifts) >= 2:
                trans_rank.append((mean(lifts), mean(probs), sum(counts), key, len(lifts)))
        for rank, (ml, mp, cnt, (src, dst), v) in enumerate(sorted(trans_rank, reverse=True)[:30], 1):
            f.write(f"{rank:3d}. {src}->{dst:28s} MeanLift={ml:.3f} MeanProb={pct(mp)} TotalN={cnt} ValidN={v}\n")
        f.write("\n")

        f.write("Mechanical Research Notes\n")
        f.write("=========================\n")
        f.write("- This study compresses exact family paths into finite structural states.\n")
        f.write("- A strong result would show persistent, replicated structural-state transitions across instruments.\n")
        f.write("- This is still descriptive research, not an entry model or trading policy.\n")
        f.write("- If useful, the next validation step should freeze these state rules and walk them forward.\n")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="APVA Structural State Machine Study v0.1")
    p.add_argument("csv_files", nargs="+", help="Input evidence CSV files")
    p.add_argument("--out-root", default=os.path.join("Evidence", "Output"), help="Output root directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results: List[InstrumentResult] = []
    for path in args.csv_files:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        res = analyze_instrument(path)
        results.append(res)
        out = write_instrument_report(res, args.out_root)
        print(f"Wrote {out}")
    agg = aggregate_results(results, args.out_root)
    print(f"Wrote {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
