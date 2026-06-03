#!/usr/bin/env python3
"""
APVA Age Transition Matrix Study v0.1

Standard-library APVA research script.
No trades, no fitting, no optimization.

Inputs:
    One or more APVA evidence CSV files or directories.

Outputs:
    Per instrument:
        Evidence/Output/<Instrument>/APVA_AgeTransitionMatrix_41_<Instrument>.txt
    Aggregate:
        Evidence/Output/AgeTransitionMatrix/APVA_AgeTransitionMatrix_41_All.txt

This script reuses the same mechanical family/archetype/structural-state logic
as the Study 39/40 branch of the APVA research pipeline.
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
LAGS = (1, 2, 3, 5)


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
        if not s:
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
class InstrumentResult:
    instrument: str
    bars: List[Bar]
    source_paths: List[str]


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
    return b.participation_state.lower() == "peak" or has_flag(b, "ParticipationPeak") or has_flag(b, "PeakVolume")


def is_climactic(b: Bar) -> bool:
    ps = b.participation_state.lower()
    es = b.expansion_state.lower()
    return ps == "climactic" or es == "climactic" or has_flag(b, "ParticipationClimactic") or has_flag(b, "ExpansionClimactic")


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
    if all(x in ("D", "N") for x in path) and len(set(path)) > 1 and (path[-1] == "D" or (len(path) >= 2 and path[-2:] == ("N", "D"))):
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
    if archetype == "Compression Resolution" or "B" in path or any(path[i] == "B" and path[i + 1] in ("N", "C") for i in range(len(path) - 1)):
        return "CompressionProcessing"
    if archetype == "Decay" or (path[-1] == "N" and any(x in ("B", "C") for x in path[:-1])):
        return "DecayToNeutral"
    if archetype == "Reassertion" or (all(x in ("D", "N") for x in path) and path[-1] == "D"):
        return "ReassertionProcessing"
    if archetype == "Destructive Persistence" or path_contains(path, ("C", "C")) or path_contains(path, ("D", "D")):
        return "DestructiveRotation"
    if archetype == "Constructive Emergence" or (path[-1] in ("A", "B") and any(x in ("N", "D", "C") for x in path[:-1])):
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


def assign_run_age(bars: List[Bar]) -> None:
    i = 0
    while i < len(bars):
        j = i
        st = bars[i].state
        while j + 1 < len(bars) and bars[j + 1].state == st:
            j += 1
        run_len = j - i + 1
        for k in range(i, j + 1):
            age = k - i + 1
            bars[k].age = age
            bars[k].age_bucket = age_bucket(age)
            if run_len == 1:
                bars[k].position = "Only"
            elif age == 1:
                bars[k].position = "First"
            elif age == run_len:
                bars[k].position = "Last"
            elif age <= math.ceil(run_len * 0.33):
                bars[k].position = "Early"
            elif age >= math.ceil(run_len * 0.67):
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


def read_csv(path: str) -> List[Bar]:
    instrument = infer_instrument(path)
    bars: List[Bar] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        c_time = find_col(headers, ["Time", "DateTime", "Timestamp"])
        c_close = find_col(headers, ["Close"])
        c_pol = find_col(headers, ["VolumePolarity"])
        c_part = find_col(headers, ["ParticipationState"])
        c_exp = find_col(headers, ["ExpansionState"])
        c_comp = find_col(headers, ["CompressionState"])
        c_diss = find_col(headers, ["DissipationState"])
        c_acc = find_col(headers, ["AcceptanceState"])
        c_flags = find_col(headers, ["EvidenceFlags", "Flags"])
        for idx, row in enumerate(r):
            b = Bar(
                instrument=instrument,
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
            )
            b.family = assign_family(b)
            bars.append(b)
    return bars


def finalize_bars(bars: List[Bar]) -> None:
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
    results = []
    for inst, bars in by_inst.items():
        bars.sort(key=lambda b: (b.time, b.rownum))
        finalize_bars(bars)
        results.append(InstrumentResult(inst, bars, sources[inst]))
    results.sort(key=lambda r: r.instrument)
    return results


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

def outcome_by_state_age(bars: List[Bar]) -> Dict[Tuple[str, str], Outcome]:
    out: Dict[Tuple[str, str], Outcome] = defaultdict(Outcome)
    for i,b in enumerate(bars):
        out[(b.state,b.age_bucket)].add(directional_return(bars,i,5))
    return out


def write_instrument(r: InstrumentResult, out_root: str) -> None:
    out_dir=os.path.join(out_root,r.instrument); ensure_dir(out_dir)
    path=os.path.join(out_dir,f"OutcomeVsAge_{r.instrument}.txt")
    rows=outcome_by_state_age(r.bars)
    with open(path,'w',encoding='utf-8') as f:
        f.write(f"APVA Outcome vs State Age Study v0.1 - {r.instrument}\n")
        f.write('='*60+'\n\n')
        f.write(f"Rows: {len(r.bars)}\n\n")
        f.write(f"{'State':<28} {'Age':<6} {'N':>8} {'Valid':>8} {'MeanDR':>12} {'MedianDR':>12} {'Cont':>9} {'Fail':>9} {'Flat':>9} {'Skew':>9}\n")
        for st in STRUCTURAL_STATES:
            for age in AGE_BUCKETS:
                o=rows.get((st,age))
                if o:
                    f.write(f"{st:<28} {age:<6} {o.count:8d} {o.valid:8d} {fmt(o.mean_dr,5):>12} {fmt(o.median_dr,5):>12} {pct(o.cont_rate):>9} {pct(o.fail_rate):>9} {pct(o.flat_rate):>9} {pct(o.skew):>9}\n")


def write_aggregate(results: List[InstrumentResult], out_root: str) -> None:
    out_dir=os.path.join(out_root,'OutcomeVsAge'); ensure_dir(out_dir)
    path=os.path.join(out_dir,'OutcomeVsAge_All.txt')
    instruments=[r.instrument for r in results]
    maps={r.instrument: outcome_by_state_age(r.bars) for r in results}
    with open(path,'w',encoding='utf-8') as f:
        f.write('APVA Outcome vs State Age Study v0.1 - Aggregate\n')
        f.write('='*56+'\n\n')
        f.write('Instruments: '+', '.join(instruments)+'\n\n')
        f.write(f"{'State':<28} {'Age':<6}")
        for inst in instruments:
            f.write(f" {inst+'_N':>8} {inst+'_Cont':>10} {inst+'_Fail':>10} {inst+'_Skew':>10} {inst+'_MeanDR':>12}")
        f.write(f" {'ValidN':>7} {'MeanCont':>10} {'MeanFail':>10} {'MeanSkew':>10} {'MeanDR':>12}\n")
        ranked=[]
        for st in STRUCTURAL_STATES:
            for age in AGE_BUCKETS:
                cont=[]; fail=[]; skew=[]; dr=[]; valid=0; line=f"{st:<28} {age:<6}"
                for inst in instruments:
                    o=maps[inst].get((st,age))
                    if o and o.valid >= 20:
                        valid+=1; cont.append(o.cont_rate); fail.append(o.fail_rate); skew.append(o.skew); dr.append(o.mean_dr)
                        line += f" {o.count:8d} {pct(o.cont_rate):>10} {pct(o.fail_rate):>10} {pct(o.skew):>10} {fmt(o.mean_dr,5):>12}"
                    else:
                        n=o.count if o else 0
                        line += f" {n:8d} {pct(0):>10} {pct(0):>10} {pct(0):>10} {fmt(0,5):>12}"
                if valid>=2:
                    ms=mean(skew); md=mean(dr)
                    line += f" {valid:7d} {pct(mean(cont)):>10} {pct(mean(fail)):>10} {pct(ms):>10} {fmt(md,5):>12}\n"; f.write(line)
                    ranked.append((ms,md,st,age))
        f.write('\nRankings\n========\n\nBest replicated state-age outcome skew\n......................................\n')
        for i,(sk,md,st,age) in enumerate(sorted(ranked, reverse=True)[:50],1):
            f.write(f"{i:3d}. {st} Age={age} | Skew={pct(sk)} MeanDR={fmt(md,5)}\n")
        f.write('\nWorst replicated state-age outcome skew\n.......................................\n')
        for i,(sk,md,st,age) in enumerate(sorted(ranked)[:50],1):
            f.write(f"{i:3d}. {st} Age={age} | Skew={pct(sk)} MeanDR={fmt(md,5)}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--out-root',default='Evidence/Output')
    args=ap.parse_args(argv); results=load_results(args.inputs)
    for r in results: write_instrument(r,args.out_root)
    write_aggregate(results,args.out_root)
    print(f"Wrote OutcomeVsAge reports under {args.out_root}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
