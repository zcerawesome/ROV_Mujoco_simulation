import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rov_core import CoreController  # noqa: E402


def test_proportional_term_scales_with_error():
    c = CoreController(0.5, 0.0, 0.0, target=2.0)
    c.record_measurement(1.0)
    assert c.compute_filtered(0.1) == pytest.approx(0.5)
    c2 = CoreController(0.5, 0.0, 0.0, target=2.0)
    c2.record_measurement(1.5)
    assert c2.compute_filtered(0.1) == pytest.approx(0.25)


def test_no_derivative_kick_on_first_sample():
    c = CoreController(0.0, 0.0, 10.0, target=1.0)
    c.record_measurement(0.0)
    assert c.compute_filtered(0.1) == pytest.approx(0.0)


def test_filter_seeds_with_first_measurement():
    c = CoreController(1.0, 0.0, 0.0)
    c.record_measurement(3.0)
    assert c.get_measurement() == pytest.approx(3.0)
    c.record_measurement(1.0)
    assert c.get_measurement() == pytest.approx(2.0)   # EMA alpha = 0.5


def test_output_clamped_and_integral_does_not_wind_up_when_saturated():
    c = CoreController(0.0, 1.0, 0.0, target=10.0)
    c.record_measurement(0.0)
    for _ in range(100):
        out = c.compute_filtered(0.1)
    assert out == pytest.approx(1.0)
    assert c.integral <= 1.0 + 1e-9
