"""Poll-free track-record model (research claim T-004) and its inputs."""
from random import Random

import pytest

from app import redistricting, track_record as tr
from app.ingest import ballot_2026, cook_pvi


# --- Cook PVI parsing: the article changed format three times --------------

PVI_2010_STYLE = """==List of PVIs==
===By congressional district===
{| class="wikitable sortable"
|-
| Alaska
| [[Alaska's At-large congressional district|At-large]]
| {{party shading/Republican}} | <span style="display:none">086</span>R+13
|-
| California
| [[California's 28th congressional district|28th]]
| {{party shading/Democratic}} | <span style="display:none">123</span>D+20
|}
===By state===
{| class="wikitable sortable"
|-
| [[Alabama]]
| {{party shading/Republican}} | <span style="display:none">086</span>R+13
|}
"""
PVI_2018_STYLE = """==By congressional district==
|-
| {{ushr|California|26|X}}
| {{party shading/Democratic}} | {{sort|092|D+7}}
|-
| {{ushr|Ohio|9|X}}
| EVEN
"""
PVI_2026_STYLE = """==By congressional district==
|-
| {{ushr|Texas|9|X}}
| {{Shading PVI|R|9}}
|-
| {{ushr|Texas|34|X}}
| {{Shading PVI|R|value=3}}
|-
| {{ushr|Wyoming|AL|X}}
| {{Shading PVI|R|23}}
|}
==By state==
{| class="wikitable"
|-
| [[West Virginia]]
| {{Shading PVI|R|21}}
|-
| [[Wyoming]]
| {{Shading PVI|R|23}}
|}
==See also==
[[Wyoming's at-large congressional district]]
"""


def test_pvi_parser_handles_every_article_format():
    old = cook_pvi.parse_districts(PVI_2010_STYLE)
    assert old == {("AK", 1): -13.0, ("CA", 28): 20.0}
    assert cook_pvi.parse_states(PVI_2010_STYLE) == {"AL": -13.0}
    mid = cook_pvi.parse_districts(PVI_2018_STYLE)
    assert mid == {("CA", 26): 7.0, ("OH", 9): 0.0}
    new = cook_pvi.parse_districts(PVI_2026_STYLE)
    assert new == {("TX", 9): -9.0, ("TX", 34): -3.0, ("WY", 1): -23.0}
    # "West Virginia" must not be read as Virginia; the table's last row counts
    assert cook_pvi.parse_states(PVI_2026_STYLE) == {"WV": -21.0, "WY": -23.0}


def test_bundled_pvi_vintages_cover_2026_on_the_new_maps():
    districts, states = cook_pvi.load_vintages()
    assert sum(1 for (c, _s, _d) in districts if c == 2026) == 435
    assert sum(1 for (c, _s) in states if c == 2026) == 50
    # redrawn for 2026: the Memphis 9th was split, Texas' 9th became R-leaning
    assert districts[(2026, "TN", 9)] < 0 < districts[(2024, "TN", 9)]
    assert districts[(2026, "TX", 9)] < 0 < districts[(2024, "TX", 9)]


# --- 2026 ballot status ------------------------------------------------------

HOUSE_ROWS = """
|-
!{{ushr|NE|2|X}}
|{{shading PVI|D|3}}
|{{sortname|Don|Bacon}}
|style="background-color:#FFB6B6" |Republican
|[[2016 United States House of Representatives elections in Nebraska|2016]]
|style="background:#DDDDDD" |Incumbent retiring<ref name="BaconNE"/>
|nowrap |{{plainlist}}
*{{Party stripe|Democratic Party (US)}}Denise Powell (Democratic)
{{endplainlist}}
|-
!rowspan=2 |{{ushr|TX|33|X}}
|rowspan=2 {{shading PVI|D|18}}
|{{sortname|Jasmine|Crockett}}
|style="color:black;background-color:#B0CEFF" |Democratic
|[[2022 United States House of Representatives elections in Texas|2022]]
|Incumbent renominated
|rowspan=2 |{{plainlist}}
*{{Party stripe|Republican Party (US)}}Patrick Gillespie (Republican)
{{endplainlist}}
"""
SENATE_ROWS = """
|-
! [[2026 United States Senate special election in Ohio|Ohio]]<br />(Class 3)
| {{Shading PVI|R|5}}
| [[Jon Husted]]
| {{Party shading/Republican}} | Republican
| 2025 {{small|(appointed)}}
| data-sort-value=0 | Interim appointee nominated
| nowrap | {{Plainlist |
*{{Party stripe|Democratic Party (United States)}}[[Sherrod Brown]] (Democratic)
}}
|-
! [[2026 United States Senate election in Maine|Maine]]
| {{Shading PVI|D|4}}
| {{sortname|Susan|Collins}}
| {{Party shading/Republican}} | Republican
| data-sort-value=1 | Incumbent renominated
| nowrap | {{Plainlist |
*{{Party stripe|Democratic Party (US)}}Someone (Democratic)
}}
"""


def test_ballot_status_parser():
    house = {r["seat_key"]: r for r in ballot_2026.parse_house(HOUSE_ROWS)}
    assert house["house-NE-02"]["incumbent_running"] == 0
    assert house["house-NE-02"]["incumbent_party"] == "R"
    # rowspan rows parse too; the candidate list's parties never leak in
    assert house["house-TX-33"]["incumbent_running"] == 1
    assert house["house-TX-33"]["incumbent_party"] == "D"
    senate = {r["seat_key"]: r for r in ballot_2026.parse_senate(SENATE_ROWS)}
    # appointed senators are not "incumbents" in the fitted sense (see RUNNING)
    assert senate["senate-OH-special"]["incumbent_running"] == 0
    assert senate["senate-ME"]["incumbent_running"] == 1


# --- incumbency and redistricting ---------------------------------------------

def test_incumbency_flags_follow_prior_winners():
    from app.incumbency import IncumbencyLookup, surname
    assert surname("ROBERT C. \"BOBBY\" SCOTT JR.") == "SCOTT"
    lookup = IncumbencyLookup.__new__(IncumbencyLookup)
    lookup._flags = {}
    lookup._derive({
        (2020, "house-XX-01"): {"D": ("SMITH", 60.0), "R": ("JONES", 40.0)},
        (2022, "house-XX-01"): {"D": ("SMITH", 55.0), "R": ("BROWN", 45.0)},
        # Smith moved districts after a redraw: still an incumbent
        (2022, "house-XX-02"): {"D": ("SMITH", 30.0), "R": ("GREEN", 70.0)},
        (2022, "house-XX-03"): {"D": ("NEW", 30.0), "R": ("OTHER", 70.0)},
    }, "house")
    assert lookup.value(2022, "house-XX-01") == 1.0
    assert lookup.value(2022, "house-XX-02") == 1.0
    assert lookup.value(2022, "house-XX-03") == 0.0
    assert lookup.value(2026, "house-ZZ-01") is None


def test_map_changes_cover_census_and_mid_decade_redraws():
    assert redistricting.map_changed("PA", 2016, 2018)       # court map
    assert not redistricting.map_changed("PA", 2018, 2020)
    assert redistricting.map_changed("WI", 2020, 2022)       # census
    assert not redistricting.map_changed("WI", 2022, 2024)
    assert redistricting.map_changed("NY", 2022, 2024)       # 2024 redraw
    assert redistricting.map_changed("OH", 2024, 2026)       # 2026 redraw
    assert not redistricting.map_changed("OH", None, 2026)


# --- the model -----------------------------------------------------------------

def _history(seed=3, cycles=range(2008, 2026, 2), n=240):
    """Synthetic House history where PVI and incumbency drive outcomes."""
    rng = Random(seed)
    lean = [rng.gauss(0, 12) for _ in range(n)]
    history = {}
    for cycle in cycles:
        env, mid = tr.environment(cycle)
        national = rng.gauss(0, 3)
        rows = []
        for i, pvi in enumerate(lean):
            inc = rng.choice([1.0, -1.0, 0.0])
            y = 2.0 * pvi + 4.0 * inc + national + rng.gauss(0, 5)
            rows.append(tr.SeatRow(seat_key=f"house-ZZ-{i:02d}", chamber="house",
                                   state=("AA", "BB", "CC")[i % 3], district=str(i + 1),
                                   cycle=cycle, y=y, prior=None, pvi=pvi, lean=0.0,
                                   inc=inc, env=env, mid=mid))
        history[cycle] = rows
    return history


def test_walk_forward_is_out_of_sample_and_picks_the_right_system():
    history = _history()
    records = tr.walk_forward(history)
    # the first cycle has no training data and is never scored
    assert min(c for c, _ in records) > min(history)
    # changing a LATER cycle's outcomes cannot change an earlier prediction
    altered = {c: rows for c, rows in history.items()}
    altered[2024] = [tr.SeatRow(**{**r.__dict__, "y": -r.y}) for r in history[2024]]
    again = tr.walk_forward(altered)
    key = next(k for k in records if k[0] == 2020)
    assert again[key].preds == records[key].preds
    ranking = tr.rank_systems(list(records.values()))
    assert ranking[0] in {"pvi_inc", "full", "ensemble"}


def test_systems_train_only_where_their_input_exists():
    history = _history()
    rows = [r for rs in history.values() for r in rs]
    # add PVI-less rows whose outcomes would badly distort a PVI fit
    noise = [tr.SeatRow(**{**r.__dict__, "pvi": None, "y": 40.0 * r.inc}) for r in rows[:300]]
    clean = tr.fit_systems(rows)["pvi_inc"]
    mixed = tr.fit_systems(rows + noise)["pvi_inc"]
    assert clean.weights == pytest.approx(mixed.weights)
    # and a system missing its defining input is never used for that race
    row = tr.SeatRow(seat_key="house-ZZ-99", chamber="house", state="AA", district="99",
                     cycle=2026, y=None, pvi=None)
    assert tr.pick_for(["pvi_inc", "pvi", "ensemble"], row) == "ensemble"


def test_local_uncertainty_tracks_each_seats_own_record():
    records = []
    for cycle in (2018, 2020, 2022, 2024):
        for i in range(60):
            miss = 15.0 if i == 0 else 1.0          # seat 0 is always badly missed
            sign = 1 if (cycle // 2) % 2 else -1
            records.append(tr.Scored(cycle, f"house-ZZ-{i:02d}", "AA", 5.0 + sign * miss,
                                     {c: 5.0 for c in tr.CANDIDATES}, pvi=1.0, prior=1.0))
    unc = tr.Uncertainty(records, "pvi_inc", "house")
    noisy = unc.local(2026, "house-ZZ-00", "AA")
    steady = unc.local(2026, "house-ZZ-05", "AA")
    assert noisy["sigma"] > steady["sigma"] >= tr.MIN_SIGMA
    assert noisy["national"] == steady["national"]  # the shared part is shared


def test_calibration_is_slope_only():
    rng = Random(5)
    rows = []
    for _ in range(400):
        z = rng.gauss(0, 1.5)
        rows.append((z * 10.0, 10.0, 1.0 if rng.random() < 1 / (1 + 2.718 ** (-3 * z)) else -1.0))
    a, b = tr.fit_calibration(rows)
    assert a == 0.0 and b > 1.6          # sharper than the raw normal reading
    assert tr.fit_calibration(rows[:10]) is None
    assert tr.probability(0.0, 10.0, (a, b)) == pytest.approx(0.5)


def test_poll_free_gate():
    from app import gates
    ok = {"races": 470, "with_pvi": 470, "with_ballot_status": 470}
    assert gates.check_poll_free_inputs(ok)["passed"]
    with pytest.raises(gates.GateFailure):
        gates.check_poll_free_inputs({**ok, "with_pvi": 400})
    with pytest.raises(gates.GateFailure):
        gates.check_poll_free_inputs({**ok, "with_ballot_status": 100})
