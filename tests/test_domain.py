from app.domain import EnsembleModel, RaceFeatures, rating, quality_grade
from app.backtest import HistoricalRow, walk_forward
from app.simulation import simulate_control

def test_rating_and_quality_boundaries():
 assert rating(.97)=='Safe Democratic' and rating(.50)=='Toss-up' and rating(.03)=='Likely Republican'
 assert quality_grade(4, 2, True, True)=='A'
def test_house_and_senate_are_distinct():
 h=RaceFeatures('h','house','TX',0,candidate_edge=3); s=RaceFeatures('s','senate','TX',0,candidate_edge=3)
 assert EnsembleModel().forecast(s).expected_margin > EnsembleModel().forecast(h).expected_margin
def test_walk_forward_excludes_then_learns():
 rows=[HistoricalRow(2018,RaceFeatures('a','house','TX',1),1,2018),HistoricalRow(2020,RaceFeatures('b','house','TX',1),1,2020)]
 out=walk_forward(rows,{2016}); assert out[0]['training_cycles']==[2016]; assert 2018 in out[1]['training_cycles']
def test_control_reproducible():
 fs=[EnsembleModel().forecast(RaceFeatures(str(i),'house','TX',0)) for i in range(5)]
 assert simulate_control(fs,'house',100,seed=3)==simulate_control(fs,'house',100,seed=3)
