from audit_submission import same_values


def test_audit_compares_nested_metrics_and_tolerates_float_roundoff():
    measured = {"rate": 0.3, "ci": [0.1, 0.5], "detail": {"n": 50}, "optional": None}
    saved = {**measured, "rate": 0.3 + 1e-12, "extra": "allowed"}
    assert same_values(measured, saved)


def test_audit_rejects_changed_scores_missing_keys_and_invalid_numbers():
    assert not same_values({"rate": 0.3}, {"rate": 0.9})
    assert not same_values({"n": 50}, {})
    assert not same_values([0.1, 0.5], [0.1])
    assert not same_values(0.3, "0.3")
    assert not same_values(0.3, float("nan"))
