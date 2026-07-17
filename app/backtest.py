"""Leakage-safe expanding-window evaluation primitives."""
from dataclasses import dataclass
from .domain import EnsembleModel, RaceFeatures
@dataclass(frozen=True)
class HistoricalRow: cycle:int; features:RaceFeatures; actual_margin:float; available_cycle:int

def walk_forward(rows:list[HistoricalRow], initial_cycles:set[int]):
    frozen=[]; trained=set(initial_cycles)
    for cycle in sorted(set(r.cycle for r in rows if r.cycle not in initial_cycles)):
        test=[r for r in rows if r.cycle==cycle]
        assert all(r.available_cycle<=cycle for r in test), 'future information cannot enter snapshot'
        assert cycle not in trained, 'held-out cycle leaked into training'
        for r in test:
            fc=EnsembleModel().forecast(r.features); frozen.append({'cycle':cycle,'race_id':r.features.race_id,'probability':fc.dem_probability,'actual_margin':r.actual_margin,'training_cycles':sorted(trained)})
        trained.add(cycle) # only after frozen predictions have been scored
    return frozen
