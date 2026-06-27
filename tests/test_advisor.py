from advisor import overall_Recommendation


def test_overall_recommendation_buy_when_weighted_score_is_positive():
    signal, score, breakdown = overall_Recommendation(
        currentPrice=120,
        shortMA=110,
        longMA=100,
        momentum=0.05,
        trendStrength=0.05,
        volatility=0.01,
        sharpeRatio=1.0,
    )

    assert signal == "BUY"
    assert score > 0
    assert breakdown["moving_avg"] == "BUY"


def test_overall_recommendation_sell_when_weighted_score_is_negative():
    signal, score, breakdown = overall_Recommendation(
        currentPrice=80,
        shortMA=90,
        longMA=100,
        momentum=-0.05,
        trendStrength=-0.05,
        volatility=0.05,
        sharpeRatio=-0.2,
    )

    assert signal == "SELL"
    assert score < 0
    assert breakdown["moving_avg"] == "SELL"


def test_overall_recommendation_current_zero_score_behavior_is_sell():
    signal, score, breakdown = overall_Recommendation(
        currentPrice=120,
        shortMA=110,
        longMA=100,
        momentum=-0.05,
        trendStrength=0.0,
        volatility=0.05,
        sharpeRatio=0.1,
    )

    assert score == 0
    assert signal == "SELL"
    assert breakdown["moving_avg"] == "BUY"
    assert breakdown["momentum"] == "SELL"
