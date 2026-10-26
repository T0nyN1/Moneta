"""_utils 测试：max_abs、to_date、is_leap_year、user_stacklevel。"""

import calendar
import random
from datetime import date, datetime

import numpy as np
import pytest

from moneta._utils import is_leap_year, max_abs, to_date, user_stacklevel

MIN, MAX = -(2 ** 63), 2 ** 63 - 1


class TestMaxAbs:
    def test_examples(self):
        assert max_abs(np.array([MIN, 5])) == 2 ** 63  # -2**63 取绝对值也不溢出
        assert max_abs(np.array([MAX, MIN + 1])) == MAX
        assert max_abs(np.array([-3, 7])) == 7 and max_abs(np.array([[-9], [2]])) == 9
        assert max_abs(np.array([], dtype=np.int64)) == 0 and max_abs(np.zeros((0, 3), dtype=np.int64)) == 0
        assert max_abs(np.int64(-4)) == 4 and max_abs(np.asarray(MIN)) == 2 ** 63
        assert type(max_abs(np.array([1]))) is int

    def test_random_against_python(self):
        rng = random.Random(1)
        for _ in range(500):
            xs = [rng.choice([MIN, MAX, 0, rng.randint(MIN, MAX)]) for _ in range(rng.randint(1, 6))]
            assert max_abs(np.array(xs, dtype=np.int64)) == max(abs(x) for x in xs)


class TestToDate:
    def test_accepted_inputs(self):
        d = date(2024, 1, 2)
        assert to_date(d) is d
        assert to_date(datetime(2024, 1, 2, 15, 30)) == d and type(to_date(datetime(2024, 1, 2))) is date
        assert to_date("2024-01-02") == d and to_date("  2024-01-02\t") == d
        assert to_date(None, optional=True) is None
        assert to_date("2024-01-02", optional=True) == d

    @pytest.mark.parametrize("bad", ["2024-13-01", "2024-02-30", "invalid-date", "", "2024/01/02"])
    def test_invalid_text(self, bad):
        with pytest.raises(ValueError, match="不是合法的 ISO 日期"):
            to_date(bad)

    @pytest.mark.parametrize("bad", [20240102, None, 1.5, b"2024-01-02"])
    def test_invalid_type(self, bad):
        with pytest.raises(TypeError, match="必须是 date、datetime 或 ISO 字符串"):
            to_date(bad)

    def test_label_in_messages(self):
        with pytest.raises(ValueError, match="^acquired_date 不是合法的 ISO 日期"):
            to_date("x", "acquired_date")
        with pytest.raises(TypeError, match="^as_of_date 必须是"):
            to_date(None, "as_of_date")
        with pytest.raises(TypeError, match="^日期 必须是"):  # 默认标签
            to_date(None)

    def test_optional_only_allows_none(self):
        with pytest.raises(ValueError, match="^as_of_date 不是合法的 ISO 日期"):
            to_date("2024-02-30", "as_of_date", optional=True)
        with pytest.raises(TypeError, match="^as_of_date 必须是"):
            to_date(20240102, "as_of_date", optional=True)


class TestIsLeapYear:
    @pytest.mark.parametrize("year, want", [(2000, True), (2024, True), (1600, True), (1900, False), (2023, False),
                                            (2100, False), (4, True), (1, False)])
    def test_examples(self, year, want):
        assert is_leap_year(year) is want

    def test_matches_calendar(self):
        assert all(is_leap_year(y) == calendar.isleap(y) for y in range(1, 3001))


def _call_from_moneta_frames(depth):
    """模拟 depth 层 moneta 包内部的调用帧，最内层调用 user_stacklevel。"""
    namespace = {"__name__": "moneta.fake", "user_stacklevel": user_stacklevel}
    exec("def level0():\n    return user_stacklevel()\n", namespace)
    for i in range(1, depth):
        exec(f"def level{i}():\n    return level{i - 1}()\n", namespace)
    return namespace[f"level{depth - 1}"]()


class TestUserStacklevel:
    def test_called_from_user_code(self):
        assert user_stacklevel() == 1  # 直接调用者就是测试代码

    @pytest.mark.parametrize("depth", [1, 2, 5])
    def test_skips_moneta_frames(self, depth):
        assert _call_from_moneta_frames(depth) == depth + 1  # 跳过 depth 层 moneta 帧，指向测试代码

    def test_points_warnings_at_user_code(self):
        from moneta import Money, PrecisionLossWarning
        with pytest.warns(PrecisionLossWarning) as record:
            Money("1.005", "USD")  # 告警穿过 Money、FixedArray 的内部帧，指向这一行
        assert record[0].filename == __file__
