from random import Random

from app.backtest import (CHAMPION_CANDIDATES, metrics,
                          select_chamber_champions, walk_forward)
from app.features import FeatureRow, PollLookup, ResultLookup, StateLean


def _rows(cycles, per_cycle=40, seed=1):
    rng = Random(seed)
    rows = []
    for cycle in cycles:
        for i in range(per_cycle):
            prior = rng.uniform(-30, 30)
            # [intercept, prior_margin, has_prior, prior_winner, state_lean,
            #  has_state_lean, environment, midterm_environment, gb, has_gb,
            #  poll, has_polls]
            x = [1.0, prior, 1.0, 1.0 if prior > 0 else -1.0, 0.0, 0.0, 1.0,
                 1.0 if cycle % 4 == 2 else 0.0, 0.0, 0.0, 0.0, 0.0]
            rows.append(FeatureRow(
                seat_key=f"house-XX-{i:02d}", cycle=cycle, chamber="house",
                state="XX", district=f"{i:02d}", x=x,
                actual_margin=prior * 0.85 + rng.gauss(0, 5),
                poll_count=0, has_prior=True,
                detail={"as_of": f"{cycle}-11-08"}))
    return rows


def test_walk_forward_trains_only_on_earlier_cycles():
    scored, evaluated = walk_forward(_rows([2014, 2016, 2018, 2020, 2022]), "house")
    assert evaluated == [2020, 2022]  # first three cycles are training-only
    for s in scored:
        assert max(s["training_cycles"]) < s["cycle"]


def test_walk_forward_produces_reasonable_skill():
    scored, _ = walk_forward(_rows([2012, 2014, 2016, 2018, 2020, 2022], per_cycle=60), "house")
    summary = metrics(scored)
    assert summary["n_races"] == 180  # cycles 2018/2020/2022 held out in turn
    assert summary["winner_accuracy"] > 0.8
    assert 0 < summary["brier"] < 0.25


def _result(cycle, seat_key, margin, state, chamber="house"):
    return {"cycle": cycle, "chamber": chamber, "state": state, "district": None,
            "seat_key": seat_key, "dem_margin": margin,
            "source": "medsl-constituency-returns"}


def test_champion_selection_sweeps_a_real_grid_and_picks_valid_spec():
    # The champion is chosen empirically from a grid, not hard-coded. Build a
    # multi-cycle synthetic universe and confirm select_chamber_champions
    # evaluates the full grid and returns one of its members per chamber.
    rng = Random(3)
    rows = []
    for cycle in range(2008, 2026, 2):
        for i in range(30):
            for ch, pre in (("house", "house"), ("senate", "senate")):
                m = rng.uniform(-25, 25)
                rows.append(_result(cycle, f"{pre}-S{i:02d}", m, f"S{i:02d}", ch))
    results = ResultLookup(rows)
    champs = select_chamber_champions(results, PollLookup([]), StateLean(results))
    for chamber in ("house", "senate"):
        assert champs[chamber]["name"] in CHAMPION_CANDIDATES
        # every grid member that produced predictions is scored
        assert len(champs[chamber]["scoreboard"]) >= len(CHAMPION_CANDIDATES) - 1
        best = champs[chamber]["name"]
        board = champs[chamber]["scoreboard"]
        assert board[best] == min(board.values())  # picked the lowest log loss


def test_metrics_hand_check():
    scored = [
        {"probability": 1.0, "dem_won": 1, "predicted_margin": 10.0, "actual_margin": 8.0,
         "low80": 2.0, "high80": 18.0, "low95": -2.0, "high95": 22.0},
        {"probability": 0.0, "dem_won": 1, "predicted_margin": -10.0, "actual_margin": 2.0,
         "low80": -18.0, "high80": -2.0, "low95": -22.0, "high95": 2.0},
    ]
    summary = metrics(scored)
    assert summary["brier"] == 0.5
    assert summary["winner_accuracy"] == 0.5
    assert summary["margin_mae"] == 7.0
    assert summary["coverage80"] == 0.5
    assert summary["coverage95"] == 1.0
