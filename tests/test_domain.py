from app.domain import normal_cdf, quality_grade, rating


def test_rating_boundaries():
    assert rating(.97) == "Safe Democratic"
    assert rating(.50) == "Toss-up"
    assert rating(.03) == "Likely Republican"
    assert rating(.005) == "Safe Republican"


def test_quality_grades():
    assert quality_grade(4, 2, True, True) == "A"
    assert quality_grade(0, None, True, False, boundary_certain=False) == "D"


def test_normal_cdf():
    assert abs(normal_cdf(0) - .5) < 1e-12
    assert normal_cdf(3) > .99
