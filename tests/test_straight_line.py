import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from straight_line_test import SCENARIOS, run_scenario  # noqa: E402


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_straight_line(name):
    result = run_scenario(SCENARIOS[name])
    assert result.passed, "; ".join(result.failures)
