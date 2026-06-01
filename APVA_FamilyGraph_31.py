"""APVA Family Graph Study v0.1.

Research-only graph analysis over the Study 30 family projection.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import direction_relative_return, load_rows
from APVA_FamilyEvolution_30 import (
    CLASS_TRANSITIONS,
    FAMILIES,
    FAMILY_CLASS,
    PER_INSTRUMENT_MIN_COUNT,
    PopulationStudy,
    build_population,
    format_oer,
    instrument_columns,
)
from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/FamilyGraph/FamilyGraph_All.txt")
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25
CYCLE_PATHS = tuple(f"{outer}->{middle}->{outer}" for outer in FAMILIES for middle in FAMILIES if middle != outer)


@dataclass(frozen=True)
class Connectivity:
    family: str
    total: int
    mean_lift: float
    weighted_lift: float
    max_lift: float
    strongest: str


@dataclass(frozen=True)
class NodeStats:
    family: str
    count: int
    frequency: float
    run_count: int
    mean_duration: float
    median_duration: float
    percentile90: float
    max_duration: int
    persistence1: float
    persistence5: float


@dataclass(frozen=True)
class GraphScore:
    family: str
    incoming: float
    persistence1: float
    outgoing: float
    attractor: float
    source: float
    hub: float


@dataclass(frozen=True)
class ReturnStats:
    family: str
    count: int
    return5: float
    return10: float
    return20: float
    median_bars: float


@dataclass(frozen=True)
class ClassEdge:
    transition: str
    count: int
    probability: float
    expected_count: float
    lift: float | None
    outcome: OutcomeStats


@dataclass(frozen=True)
class GraphPopulation:
    base: PopulationStudy
    nodes: dict[str, NodeStats]
    inbound: dict[str, Connectivity]
    outbound: dict[str, Connectivity]
    scores: dict[str, GraphScore]
    returns: dict[str, ReturnStats]
    class_edges: dict[str, ClassEdge]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list
    populations: dict[str, GraphPopulation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study APVA Study 30 families as graph nodes and edges.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def lift_value(value) -> float:
    return value.lift if value.lift is not None else 0.0


def node_stats(population: PopulationStudy) -> dict[str, NodeStats]:
    output = {}
    for family in FAMILIES:
        persistence = population.persistence[family]
        output[family] = NodeStats(
            family,
            population.counts[family],
            population.frequencies[family],
            persistence.run_count,
            persistence.mean_duration,
            persistence.median_duration,
            persistence.percentile90,
            persistence.max_duration,
            population.transitions[(1, f"{family}->{family}")].probability,
            population.transitions[(5, f"{family}->{family}")].probability,
        )
    return output


def connectivity(population: PopulationStudy, inbound: bool) -> dict[str, Connectivity]:
    output = {}
    for family in FAMILIES:
        edges = []
        for other in FAMILIES:
            transition = f"{other}->{family}" if inbound else f"{family}->{other}"
            edges.append((other, population.transitions[(1, transition)]))
        total = sum(value.count for _, value in edges)
        lifts = [lift_value(value) for _, value in edges]
        strongest_name, strongest_edge = max(edges, key=lambda pair: (lift_value(pair[1]), pair[1].count, pair[0]))
        output[family] = Connectivity(
            family,
            total,
            mean(lifts),
            sum(value.count * lift_value(value) for _, value in edges) / total if total else 0.0,
            lift_value(strongest_edge),
            strongest_name,
        )
    return output


def graph_scores(inbound: dict[str, Connectivity], outbound: dict[str, Connectivity], nodes: dict[str, NodeStats]) -> dict[str, GraphScore]:
    output = {}
    for family in FAMILIES:
        incoming = inbound[family].weighted_lift
        outgoing = outbound[family].weighted_lift
        persistence = nodes[family].persistence1
        output[family] = GraphScore(
            family,
            incoming,
            persistence,
            outgoing,
            incoming + persistence - outgoing,
            outgoing - incoming,
            incoming + outgoing - persistence,
        )
    return output


def return_stats(population: PopulationStudy) -> dict[str, ReturnStats]:
    output = {}
    mapping = population.family_by_index
    for family in FAMILIES:
        occurrences = [index for index, current in mapping.items() if current == family]
        first_returns = []
        for index in occurrences:
            found = next((offset for offset in range(1, 21) if mapping.get(index + offset) == family), None)
            if found is not None:
                first_returns.append(found)
        denominator = len(occurrences) if occurrences else 1
        output[family] = ReturnStats(
            family,
            len(occurrences),
            sum(value <= 5 for value in first_returns) / denominator,
            sum(value <= 10 for value in first_returns) / denominator,
            sum(value <= 20 for value in first_returns) / denominator,
            statistics.median(first_returns) if first_returns else 0.0,
        )
    return output


def class_edges(rows: list, population: PopulationStudy) -> dict[str, ClassEdge]:
    classes = ("Constructive", "Destructive", "Neutral")
    class_counts = {name: sum(FAMILY_CLASS[family] == name for family in population.family_by_index.values()) for name in classes}
    total_family = len(population.family_by_index) if population.family_by_index else 1
    class_frequencies = {name: class_counts[name] / total_family for name in classes}
    raw = {transition: 0 for transition in CLASS_TRANSITIONS}
    values = {transition: [] for transition in CLASS_TRANSITIONS}
    for index, target in population.family_by_index.items():
        source = population.family_by_index.get(index - 1)
        if source is None:
            continue
        transition = f"{FAMILY_CLASS[source]}->{FAMILY_CLASS[target]}"
        raw[transition] += 1
        outcome = direction_relative_return(rows, index, 5)
        if outcome is not None:
            values[transition].append(outcome)
    total = sum(raw.values())
    output = {}
    for source in classes:
        source_total = sum(raw[f"{source}->{target}"] for target in classes)
        for target in classes:
            transition = f"{source}->{target}"
            expected = total * class_frequencies[source] * class_frequencies[target]
            output[transition] = ClassEdge(
                transition,
                raw[transition],
                raw[transition] / source_total if source_total else 0.0,
                expected,
                raw[transition] / expected if expected else None,
                summarize(values[transition]),
            )
    return output


def build_graph_population(rows: list, base: PopulationStudy) -> GraphPopulation:
    nodes = node_stats(base)
    inbound = connectivity(base, True)
    outbound = connectivity(base, False)
    return GraphPopulation(
        base,
        nodes,
        inbound,
        outbound,
        graph_scores(inbound, outbound, nodes),
        return_stats(base),
        class_edges(rows, base),
    )


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    full = build_population(rows, "Full Population", list(range(len(rows))))
    population_b = build_population(rows, "Population B", mature_aligned_lateral_indexes(rows))
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        {"Full Population": build_graph_population(rows, full), "Population B": build_graph_population(rows, population_b)},
    )


def append_node_table(lines: list[str], graph: GraphPopulation) -> None:
    lines.append(
        f"{'Family':<8} {'Count':>8} {'Freq':>9} {'Runs':>7} {'MeanDur':>9} {'Median':>9} "
        f"{'P90':>8} {'Max':>6} {'Persist1':>9} {'Persist5':>9}"
    )
    for family in FAMILIES:
        item = graph.nodes[family]
        lines.append(
            f"{family:<8} {item.count:>8} {item.frequency:>8.2%} {item.run_count:>7} "
            f"{item.mean_duration:>9.4f} {item.median_duration:>9.4f} {item.percentile90:>8.4f} "
            f"{item.max_duration:>6} {item.persistence1:>8.2%} {item.persistence5:>8.2%}"
        )


def append_connectivity(lines: list[str], graph: GraphPopulation, inbound: bool) -> None:
    heading = "Inbound" if inbound else "Outbound"
    values = graph.inbound if inbound else graph.outbound
    lines.append(f"{'Family':<8} {'Total':>8} {'MeanLift':>10} {'WeightedLift':>13} {'MaxLift':>10} {('StrongestSource' if inbound else 'StrongestTarget'):<16}")
    for family in FAMILIES:
        item = values[family]
        lines.append(f"{family:<8} {item.total:>8} {item.mean_lift:>10.4f} {item.weighted_lift:>13.4f} {item.max_lift:>10.4f} {item.strongest:<16}")
        for other in FAMILIES:
            transition = f"{other}->{family}" if inbound else f"{family}->{other}"
            edge = graph.base.transitions[(1, transition)]
            lines.append(f"  {transition:<10} Count={edge.count:<8} Probability={edge.probability:>8.2%} Lift={format_oer(edge.lift)}")
    lines.append(f"{heading} edge probability is conditional on source family.")


def append_score_table(lines: list[str], graph: GraphPopulation, metric: str) -> None:
    lines.append(f"{'Family':<8} {'Incoming':>10} {'Persist1':>10} {'Outgoing':>10} {metric:>12}")
    items = sorted(graph.scores.values(), key=lambda item: -getattr(item, metric.lower()))
    for item in items:
        lines.append(
            f"{item.family:<8} {item.incoming:>10.4f} {item.persistence1:>9.2%} "
            f"{item.outgoing:>10.4f} {getattr(item, metric.lower()):>12.4f}"
        )


def append_cycles(lines: list[str], graph: GraphPopulation, include_low: bool = False) -> None:
    lines.append(f"{'Cycle':<12} {'Count':>8} {'OutcomeN':>9} {'OER':>9} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for cycle in CYCLE_PATHS:
        item = graph.base.pathways.get((3, cycle))
        if item is None:
            continue
        if not include_low and item.count < PER_INSTRUMENT_MIN_COUNT:
            continue
        lines.append(
            f"{cycle:<12} {item.count:>8} {item.outcome.count:>9} {format_oer(item.oer):>9} "
            f"{item.outcome.mean:>12.6f} {item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%}"
        )


def append_returns(lines: list[str], graph: GraphPopulation) -> None:
    lines.append(f"{'Family':<8} {'Count':>8} {'Return5':>10} {'Return10':>10} {'Return20':>10} {'MedianBars':>12}")
    for family in FAMILIES:
        item = graph.returns[family]
        lines.append(f"{family:<8} {item.count:>8} {item.return5:>9.2%} {item.return10:>9.2%} {item.return20:>9.2%} {item.median_bars:>12.4f}")


def append_class_edges(lines: list[str], graph: GraphPopulation) -> None:
    lines.append(f"{'Transition':<32} {'Count':>8} {'Probability':>12} {'Lift':>9} {'OutcomeN':>9} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for transition in CLASS_TRANSITIONS:
        item = graph.class_edges[transition]
        lines.append(
            f"{transition:<32} {item.count:>8} {item.probability:>11.2%} {format_oer(item.lift):>9} "
            f"{item.outcome.count:>9} {item.outcome.mean:>12.6f} {item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%}"
        )


def append_population_graph(lines: list[str], graph: GraphPopulation) -> None:
    lines.extend(["\nNode Statistics", "---------------"])
    append_node_table(lines, graph)
    lines.extend(["\nInbound Connectivity", "--------------------"])
    append_connectivity(lines, graph, True)
    lines.extend(["\nOutbound Connectivity", "---------------------"])
    append_connectivity(lines, graph, False)
    for title, metric in (("Attractor Ranking", "Attractor"), ("Source Ranking", "Source"), ("Hub Ranking", "Hub")):
        lines.extend([f"\n{title}", "-" * len(title)])
        append_score_table(lines, graph, metric)
    lines.extend(["\nReturn Probabilities", "--------------------"])
    append_returns(lines, graph)
    lines.extend(["\nConstructive / Destructive / Neutral Graph", "------------------------------------------"])
    append_class_edges(lines, graph)


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA Family Graph Study v0.1 - {study.instrument}",
        "=" * (30 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        "Study 30 family precedence reused exactly: A, then B, then C, then D, then N.",
        "Edge probabilities are conditional on source family; weighted lifts are edge-count weighted.",
        "Return means first reappearance of the same family within the next original bars.",
        "",
        "Section 1 - Node Statistics",
        "===========================",
    ]
    append_node_table(lines, full)
    lines.extend(["\nSection 2 - Inbound Connectivity", "================================"])
    append_connectivity(lines, full, True)
    lines.extend(["\nSection 3 - Outbound Connectivity", "================================="])
    append_connectivity(lines, full, False)
    lines.extend(["\nSection 4 - Attractor Analysis", "=============================="])
    append_score_table(lines, full, "Attractor")
    lines.extend(["\nSection 5 - Source Analysis", "==========================="])
    append_score_table(lines, full, "Source")
    lines.extend(["\nSection 6 - Hub Analysis", "========================"])
    append_score_table(lines, full, "Hub")
    lines.extend(["\nSection 7 - Cycle Analysis", "=========================="])
    append_cycles(lines, full)
    lines.extend(["\nSection 8 - Return Probability", "=============================="])
    append_returns(lines, full)
    lines.extend(["\nSection 9 - Constructive / Destructive / Neutral Graph", "======================================================="])
    append_class_edges(lines, full)
    lines.extend(["\nSection 10 - Population B", "========================="])
    lines.append(f"Population B bars: {len(population_b.base.indexes)}")
    append_population_graph(lines, population_b)
    lines.extend(["\nSection 11 - Mechanical Research Notes", "======================================"])
    attractor = max(full.scores.values(), key=lambda item: item.attractor)
    source = max(full.scores.values(), key=lambda item: item.source)
    hub = max(full.scores.values(), key=lambda item: item.hub)
    lines.append(f"- Strongest attractor: Family {attractor.family}, AttractorScore={attractor.attractor:.4f}.")
    lines.append(f"- Strongest source: Family {source.family}, SourceScore={source.source:.4f}.")
    lines.append(f"- Strongest hub: Family {hub.family}, HubScore={hub.hub:.4f}.")
    for family in ("A", "B", "C", "N"):
        item = full.returns[family]
        lines.append(f"- Family {family} returns: Return5={item.return5:.2%}, Return10={item.return10:.2%}, Return20={item.return20:.2%}.")
    lines.append("- Graph scores are descriptive combinations of lift and persistence only.")
    return "\n".join(lines) + "\n"


def aggregate_nodes(studies: list[InstrumentStudy], population: str = "Full Population") -> dict[str, dict[str, NodeStats | ReturnStats]]:
    return {
        family: {
            study.instrument: (
                study.populations[population].nodes[family],
                study.populations[population].returns[family],
            )
            for study in studies
        }
        for family in FAMILIES
    }


def append_aggregate_nodes(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> None:
    values = aggregate_nodes(studies)
    lines.extend(["\nAggregate Node Table", "===================="])
    header = f"{'Family':<8}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Freq_' + instrument):>9} {('Persist_' + instrument):>10} {('Ret5_' + instrument):>9} {('Ret10_' + instrument):>9} {('Ret20_' + instrument):>9}"
    header += f" {'Valid':>5} {'MeanPersist':>11} {'MeanRet5':>9} {'MeanRet10':>9} {'MeanRet20':>9}"
    lines.append(header)
    for family in FAMILIES:
        row = f"{family:<8}"
        valid = []
        for instrument in columns:
            pair = values[family].get(instrument)
            if pair:
                node, returns = pair
                row += f" {node.count:>9} {node.frequency:>8.2%} {node.persistence1:>9.2%} {returns.return5:>8.2%} {returns.return10:>8.2%} {returns.return20:>8.2%}"
                if node.count >= PER_INSTRUMENT_MIN_COUNT:
                    valid.append((node, returns))
            else:
                row += f" {0:>9} {0.0:>8.2%} {0.0:>9.2%} {0.0:>8.2%} {0.0:>8.2%} {0.0:>8.2%}"
        row += f" {len(valid):>5} {mean([item[0].persistence1 for item in valid]):>10.2%} {mean([item[1].return5 for item in valid]):>8.2%} {mean([item[1].return10 for item in valid]):>8.2%} {mean([item[1].return20 for item in valid]):>8.2%}"
        lines.append(row)


def append_aggregate_connectivity(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> None:
    lines.extend(["\nAggregate Connectivity Table", "============================"])
    header = f"{'Family':<8}"
    for instrument in columns:
        header += f" {('In_' + instrument):>9} {('WIn_' + instrument):>9} {('Out_' + instrument):>9} {('WOut_' + instrument):>9} {('Attr_' + instrument):>9} {('Src_' + instrument):>9} {('Hub_' + instrument):>9}"
    header += f" {'MeanAttr':>9} {'MeanSrc':>9} {'MeanHub':>9}"
    lines.append(header)
    for family in FAMILIES:
        row = f"{family:<8}"
        scores = []
        for instrument in columns:
            study = next((item for item in studies if item.instrument == instrument), None)
            if study:
                graph = study.populations["Full Population"]
                incoming = graph.inbound[family]
                outgoing = graph.outbound[family]
                score = graph.scores[family]
                scores.append(score)
                row += f" {incoming.mean_lift:>9.4f} {incoming.weighted_lift:>9.4f} {outgoing.mean_lift:>9.4f} {outgoing.weighted_lift:>9.4f} {score.attractor:>9.4f} {score.source:>9.4f} {score.hub:>9.4f}"
            else:
                row += "    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000"
        row += f" {mean([item.attractor for item in scores]):>9.4f} {mean([item.source for item in scores]):>9.4f} {mean([item.hub for item in scores]):>9.4f}"
        lines.append(row)


def append_aggregate_cycles(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> list[tuple[str, dict]]:
    output = []
    lines.extend(["\nAggregate Cycle Table", "====================="])
    header = f"{'Cycle':<12}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('OER_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9}"
    header += f" {'Valid':>5} {'MeanOER':>9} {'MeanCont':>9} {'MeanFail':>9}"
    lines.append(header)
    for cycle in CYCLE_PATHS:
        values = {}
        for study in studies:
            item = study.populations["Full Population"].base.pathways.get((3, cycle))
            if item:
                values[study.instrument] = item
        if sum(item.count for item in values.values()) < AGGREGATE_MIN_COUNT:
            continue
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        row = f"{cycle:<12}"
        for instrument in columns:
            item = values.get(instrument)
            row += (
                f" {item.count:>9} {format_oer(item.oer):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%}"
                if item else f" {0:>9} {'N/A':>9} {0.0:>8.2%} {0.0:>8.2%}"
            )
        metrics = {
            "valid": len(valid),
            "oer": mean([item.oer for item in valid if item.oer is not None]),
            "cont": mean([item.outcome.continuation_rate for item in valid]),
            "fail": mean([item.outcome.failure_rate for item in valid]),
            "count": sum(item.count for item in values.values()),
        }
        row += f" {metrics['valid']:>5} {metrics['oer']:>9.4f} {metrics['cont']:>8.2%} {metrics['fail']:>8.2%}"
        lines.append(row)
        output.append((cycle, metrics))
    return output


def append_aggregate_classes(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> list[tuple[str, dict]]:
    output = []
    lines.extend(["\nAggregate Constructive / Destructive Table", "=========================================="])
    header = f"{'TransitionClass':<32}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Prob_' + instrument):>9} {('Lift_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9}"
    header += f" {'MeanLift':>9} {'MeanCont':>9} {'MeanFail':>9}"
    lines.append(header)
    for transition in CLASS_TRANSITIONS:
        values = {study.instrument: study.populations["Full Population"].class_edges[transition] for study in studies}
        row = f"{transition:<32}"
        for instrument in columns:
            item = values.get(instrument)
            row += (
                f" {item.count:>9} {item.probability:>8.2%} {format_oer(item.lift):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%}"
                if item else f" {0:>9} {0.0:>8.2%} {'N/A':>9} {0.0:>8.2%} {0.0:>8.2%}"
            )
        metrics = {
            "lift": mean([lift_value(item) for item in values.values()]),
            "cont": mean([item.outcome.continuation_rate for item in values.values()]),
            "fail": mean([item.outcome.failure_rate for item in values.values()]),
        }
        row += f" {metrics['lift']:>9.4f} {metrics['cont']:>8.2%} {metrics['fail']:>8.2%}"
        lines.append(row)
        output.append((transition, metrics))
    return output


def append_ranked(lines: list[str], title: str, values: list[str]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.extend(values[:TOP_LIMIT])


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    lines = [
        "APVA Family Graph Study v0.1 - Cross-Instrument Aggregate",
        "========================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        "Study 30 family precedence reused exactly: A, then B, then C, then D, then N.",
        f"Cycle aggregate threshold: {AGGREGATE_MIN_COUNT}; replication threshold: Count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
    ]
    append_aggregate_nodes(lines, studies, columns)
    append_aggregate_connectivity(lines, studies, columns)
    cycles = append_aggregate_cycles(lines, studies, columns)
    classes = append_aggregate_classes(lines, studies, columns)
    scores = {
        family: [
            study.populations["Full Population"].scores[family]
            for study in studies
            if study.populations["Full Population"].nodes[family].count >= PER_INSTRUMENT_MIN_COUNT
        ]
        for family in FAMILIES
    }
    returns = {
        family: [
            study.populations["Full Population"].returns[family]
            for study in studies
            if study.populations["Full Population"].nodes[family].count >= PER_INSTRUMENT_MIN_COUNT
        ]
        for family in FAMILIES
    }
    b_differences = []
    for family in FAMILIES:
        for study in studies:
            full = study.populations["Full Population"].scores[family]
            pop_b = study.populations["Population B"].scores[family]
            b_differences.append((family, study.instrument, pop_b.attractor - full.attractor))
    append_ranked(lines, "Strongest Attractors", [f"Family {family}: MeanAttractor={mean([item.attractor for item in values]):.4f}" for family, values in sorted(scores.items(), key=lambda pair: -mean([item.attractor for item in pair[1]]))])
    append_ranked(lines, "Strongest Sources", [f"Family {family}: MeanSource={mean([item.source for item in values]):.4f}" for family, values in sorted(scores.items(), key=lambda pair: -mean([item.source for item in pair[1]]))])
    append_ranked(lines, "Strongest Hubs", [f"Family {family}: MeanHub={mean([item.hub for item in values]):.4f}" for family, values in sorted(scores.items(), key=lambda pair: -mean([item.hub for item in pair[1]]))])
    append_ranked(lines, "Highest Persistence Families", [f"Family {family}: MeanPersistence1={mean([item.persistence1 for item in values]):.2%}" for family, values in sorted(((family, [study.populations['Full Population'].nodes[family] for study in studies]) for family in FAMILIES), key=lambda pair: -mean([item.persistence1 for item in pair[1]]))])
    append_ranked(lines, "Highest Return Families", [f"Family {family}: MeanReturn5={mean([item.return5 for item in values]):.2%}, MeanReturn10={mean([item.return10 for item in values]):.2%}, MeanReturn20={mean([item.return20 for item in values]):.2%}" for family, values in sorted(returns.items(), key=lambda pair: -mean([item.return20 for item in pair[1]]))])
    eligible_cycles = [(cycle, item) for cycle, item in cycles if item["valid"] >= 2]
    append_ranked(lines, "Strongest Cycles by OER", [f"{cycle}: MeanOER={item['oer']:.4f}, AggregateCount={item['count']}" for cycle, item in sorted(eligible_cycles, key=lambda pair: -pair[1]["oer"])])
    append_ranked(lines, "Best Cycles by Continuation", [f"{cycle}: MeanContinuation={item['cont']:.2%}, MeanOER={item['oer']:.4f}" for cycle, item in sorted(eligible_cycles, key=lambda pair: -pair[1]["cont"])])
    append_ranked(lines, "Worst Cycles by Failure", [f"{cycle}: MeanFailure={item['fail']:.2%}, MeanOER={item['oer']:.4f}" for cycle, item in sorted(eligible_cycles, key=lambda pair: -pair[1]["fail"])])
    append_ranked(lines, "Best Constructive / Destructive Transitions", [f"{transition}: MeanContinuation={item['cont']:.2%}, MeanLift={item['lift']:.4f}" for transition, item in sorted(classes, key=lambda pair: -pair[1]["cont"])])
    append_ranked(lines, "Worst Constructive / Destructive Transitions", [f"{transition}: MeanFailure={item['fail']:.2%}, MeanLift={item['lift']:.4f}" for transition, item in sorted(classes, key=lambda pair: -pair[1]["fail"])])
    append_ranked(lines, "Population-B Strongest Differences", [f"Family {family} | {instrument}: AttractorDeltaVsFull={delta:.4f}" for family, instrument, delta in sorted(b_differences, key=lambda item: -abs(item[2]))])
    lines.extend(["\nResearch Notes", "=============="])
    lines.append("- Graph scores combine transition lift and persistence mechanically.")
    lines.append("- Return rates measure first same-family reappearance inside 5, 10, and 20 original bars.")
    lines.append("- Population B differences compare attractor score against the full population.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"FamilyGraph_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
