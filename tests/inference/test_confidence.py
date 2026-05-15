"""Tests for codingjepa.inference.confidence (#87)."""

from __future__ import annotations

import math

from codingjepa.inference.confidence import calibrate


class TestCalibrateEmpty:
    def test_returns_empty_list(self) -> None:
        assert calibrate([]) == []


class TestCalibrateSingle:
    def test_single_value_returns_one(self) -> None:
        result = calibrate([0.9])
        assert result == [1.0]

    def test_single_zero_returns_one(self) -> None:
        assert calibrate([0.0]) == [1.0]

    def test_single_negative_returns_one(self) -> None:
        assert calibrate([-0.5]) == [1.0]


class TestCalibrateUniform:
    def test_equal_cosines_uniform(self) -> None:
        result = calibrate([0.5, 0.5, 0.5])
        expected = 1.0 / 3
        for v in result:
            assert abs(v - expected) < 1e-9

    def test_equal_cosines_sum_to_one(self) -> None:
        result = calibrate([0.8, 0.8, 0.8, 0.8])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_zeros_uniform(self) -> None:
        result = calibrate([0.0, 0.0])
        for v in result:
            assert abs(v - 0.5) < 1e-9


class TestCalibrateOrder:
    def test_higher_cosine_higher_confidence(self) -> None:
        result = calibrate([0.9, 0.7, 0.5])
        assert result[0] > result[1] > result[2]

    def test_sum_to_one(self) -> None:
        result = calibrate([0.9, 0.8, 0.7])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_two_values_ordering(self) -> None:
        result = calibrate([0.8, 0.2])
        assert result[0] > result[1]

    def test_probabilities_in_unit_interval(self) -> None:
        result = calibrate([0.9, 0.5, 0.1, 0.0])
        for v in result:
            assert 0.0 <= v <= 1.0


class TestCalibrateNumericalStability:
    def test_large_negative_no_nan(self) -> None:
        # Extremely negative cosines should not produce NaN
        result = calibrate([-100.0, -200.0, -300.0])
        for v in result:
            assert not math.isnan(v)
            assert not math.isinf(v)

    def test_large_negative_no_nan_sum(self) -> None:
        result = calibrate([-1000.0, -2000.0, -3000.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_very_close_values_no_nan(self) -> None:
        result = calibrate([0.9999, 0.9998, 0.9997])
        for v in result:
            assert not math.isnan(v)

    def test_temperature_sharpening(self) -> None:
        # With tau < 0.1, the distribution should be more concentrated on the max
        result_sharp = calibrate([0.9, 0.7], tau=0.01)
        result_soft = calibrate([0.9, 0.7], tau=1.0)
        # Sharper temperature → larger gap between first and second
        assert result_sharp[0] - result_sharp[1] > result_soft[0] - result_soft[1]
