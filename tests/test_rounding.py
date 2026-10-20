"""_rounding 测试：舍入模式枚举、check_rounding 与舍入进位裁决 decide_bump。

decide_bump 两种基准坐标系：
- Floor Base：q = floor(x)，带符号，bump 作用于 q，返回 bool（或 bool 数组）；
- Truncation Base：t = trunc(|x|)，bump 作用于 |x| 的截断值，返回 uint64 的 0 / 1。
参照实现 ref_round 直接按定义对 Fraction 取整，并与 Decimal 的舍入规则互相印证。
"""

import random
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP, Decimal
from fractions import Fraction

import numpy as np
import pytest

import moneta._rounding as rounding_module
from moneta._rounding import RoundingMode, check_rounding, decide_bump, round_quotient

MODES = [m.value for m in RoundingMode]
DECIMAL_MODES = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN, "down": ROUND_DOWN, "up": ROUND_UP,
                 "floor": ROUND_FLOOR, "ceil": ROUND_CEILING}
HALF = Fraction(1, 2)


def ref_round(x: Fraction, mode: str) -> int:
    """x 按 mode 取整到整数（精确参照）。"""
    floor = x.numerator // x.denominator
    frac = x - floor
    if frac == 0 or mode == "floor":
        return floor
    if mode == "ceil":
        return floor + 1
    if mode == "down":
        return floor + 1 if x < 0 else floor
    if mode == "up":
        return floor if x < 0 else floor + 1
    if frac != HALF:
        return floor + 1 if frac > HALF else floor
    if mode == "half_up":
        return floor + 1 if x > 0 else floor
    return floor if floor % 2 == 0 else floor + 1  # half_even


def floor_base_round(x: Fraction, mode: str) -> int:
    q = x.numerator // x.denominator
    r = x - q
    bump = decide_bump(mode, is_negative=np.bool_(q < 0), odd_q=np.bool_(q % 2 != 0), has_fraction=np.bool_(r != 0),
                       over_half=np.bool_(r > HALF), exact_half=np.bool_(r == HALF), floor_base=True)
    return q + int(bump)


def trunc_base_round(x: Fraction, mode: str) -> int:
    t = abs(x).numerator // abs(x).denominator
    r = abs(x) - t
    bump = decide_bump(mode, is_negative=np.bool_(x < 0), odd_q=np.bool_(t % 2 != 0), has_fraction=np.bool_(r != 0),
                       over_half=np.bool_(r > HALF), exact_half=np.bool_(r == HALF), floor_base=False)
    magnitude = t + int(bump)
    return magnitude if x >= 0 else -magnitude


# ===========================================================================
# 1. RoundingMode
# ===========================================================================


class TestRoundingMode:
    def test_members_in_order(self):
        assert MODES == ["half_up", "half_even", "down", "up", "floor", "ceil"]
        assert [m.name for m in RoundingMode] == ["HALF_UP", "HALF_EVEN", "DOWN", "UP", "FLOOR", "CEIL"]

    def test_str_compatible(self):
        for m in RoundingMode:
            assert isinstance(m, str) and m == m.value and RoundingMode(m.value) is m
        assert "half_even" in set(RoundingMode) and RoundingMode.UP == "up"

    def test_reexported_from_fixed_array(self):
        import moneta
        from moneta import fixed_array
        assert fixed_array.RoundingMode is RoundingMode and "RoundingMode" in fixed_array.__all__
        assert moneta.RoundingMode is RoundingMode

    def test_exports(self):
        assert rounding_module.__all__ == ["RoundingMode", "_ROUNDINGS", "check_rounding", "decide_bump",
                                           "round_quotient"]
        assert rounding_module._ROUNDINGS == tuple(MODES)


# ===========================================================================
# 2. check_rounding
# ===========================================================================


class TestCheckRounding:
    @pytest.mark.parametrize("mode", list(RoundingMode))
    def test_enum_and_str(self, mode):
        assert check_rounding(mode) == mode.value and type(check_rounding(mode)) is str
        assert check_rounding(mode.value) == mode.value and type(check_rounding(mode.value)) is str

    @pytest.mark.parametrize("bad", ["nearest", "HALF_UP", "Half_Up", " half_up", "", None, 1, 0.5, True, b"up",
                                     ["half_up"], ("half_up",)])
    def test_invalid(self, bad):
        with pytest.raises(ValueError, match="rounding 只能是"):
            check_rounding(bad)


# ===========================================================================
# 3. decide_bump
# ===========================================================================


class TestDecideBump:
    @pytest.mark.parametrize("mode", MODES)
    def test_quarter_grid_both_bases(self, mode):
        # 整数部分 × 小数部分 {0, 1/4, 1/2, 3/4} 的全部组合，正负对称
        for whole in range(-4, 5):
            for frac in (Fraction(0), Fraction(1, 4), HALF, Fraction(3, 4)):
                x = whole + frac if whole >= 0 else whole - frac
                assert floor_base_round(x, mode) == ref_round(x, mode), (mode, x)
                assert trunc_base_round(x, mode) == ref_round(x, mode), (mode, x)

    def test_random_against_decimal(self):
        rng = random.Random(1)
        for _ in range(5000):
            x = Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.choice([1, 2, 3, 4, 7, 8, 10, 1000]))
            mode = rng.choice(MODES)
            want = int((Decimal(x.numerator) / Decimal(x.denominator)).to_integral_value(rounding=DECIMAL_MODES[mode]))
            assert ref_round(x, mode) == want
            assert floor_base_round(x, mode) == want and trunc_base_round(x, mode) == want, (x, mode)

    @pytest.mark.parametrize("mode", MODES)
    def test_vectorized_matches_scalar(self, mode):
        xs = [whole + frac if whole >= 0 else whole - frac
              for whole in range(-3, 4) for frac in (Fraction(0), Fraction(1, 3), HALF, Fraction(2, 3))]
        q = np.array([x.numerator // x.denominator for x in xs], dtype=np.int64)
        r = [x - int(qi) for x, qi in zip(xs, q)]
        bump = decide_bump(mode, is_negative=q < 0, odd_q=q % 2 != 0, has_fraction=np.array([f != 0 for f in r]),
                           over_half=np.array([f > HALF for f in r]), exact_half=np.array([f == HALF for f in r]),
                           floor_base=True)
        assert (q + bump).tolist() == [ref_round(x, mode) for x in xs]

    def test_return_types(self):
        kw = dict(is_negative=np.array([False, True]), odd_q=np.array([True, False]),
                  has_fraction=np.array([True, True]), over_half=np.array([False, True]),
                  exact_half=np.array([True, False]))
        for mode in MODES:
            floor_bump = decide_bump(mode, floor_base=True, **kw)
            if mode == "floor":
                assert floor_bump is False
            else:
                assert np.asarray(floor_bump).dtype == np.bool_
            trunc_bump = np.asarray(decide_bump(mode, floor_base=False, **kw))
            assert trunc_bump.dtype == np.uint64 and set(trunc_bump.ravel().tolist()) <= {0, 1}

    def test_truncation_base_is_default(self):
        # 默认 floor_base=False：down 永不进位（作用于 |q|）
        assert decide_bump("down", is_negative=True, odd_q=False, has_fraction=True, over_half=True,
                           exact_half=False) == np.uint64(0)

    @pytest.mark.parametrize("floor_base", [True, False])
    @pytest.mark.parametrize("bad", ["unknown", "HALF_UP", ""])
    def test_invalid_mode(self, floor_base, bad):
        with pytest.raises(ValueError, match="rounding 只能是"):
            decide_bump(bad, is_negative=False, odd_q=False, has_fraction=True, over_half=False, exact_half=False,
                        floor_base=floor_base)


# ===========================================================================
# 4. round_quotient
# ===========================================================================


class TestRoundQuotient:
    @pytest.mark.parametrize("mode", MODES)
    def test_against_reference_on_grid(self, mode):
        for m in range(-25, 26):
            for d in (-7, -4, -2, -1, 1, 2, 4, 7):
                q, r = divmod(m, d)
                want = ref_round(Fraction(m, d), mode)
                got = round_quotient(q, r, d, mode)
                assert got == want, (m, d, mode)

    def test_vectorized(self):
        ms = np.arange(-25, 26, dtype=np.int64)
        ds = np.array([-7, -4, -2, -1, 1, 2, 4, 7], dtype=np.int64)
        q, r = np.divmod(ms[:, None], ds[None, :])
        for mode in MODES:
            got = round_quotient(q, r, ds[None, :], mode)
            expected = [[ref_round(Fraction(int(m), int(d)), mode) for d in ds] for m in ms]
            assert got.tolist() == expected

    def test_floor_short_circuit(self):
        assert round_quotient(10, 1, 2, "floor") == 10
        assert round_quotient(-10, 1, 2, "floor") == -10
        q, r = np.divmod(np.array([7, -7]), 2)
        assert round_quotient(q, r, 2, "floor").tolist() == [3, -4]
