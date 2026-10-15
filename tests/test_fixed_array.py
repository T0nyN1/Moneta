"""FixedArray 的完整行为约定。标准答案来自 Decimal、Fraction 与 Python 整数。

已确认的设计：
- float 一律先取最短十进制写法：标量直接解析，数组逐位试探并用往返检验（不设容差），
  两者结果一致；显式 scale 时再按 half_up 舍入到该 scale，与字符串输入的结果相同。
- 运算符、比较与 mul_round / matmul / divide / round_to_step / is_on_step 只接受数值白名单内的
  操作数（FixedArray、int、float、Decimal、Fraction、数值 ndarray 及只含数值的序列）；字符串、
  布尔、带单位对象一律拒绝，运算符返回 NotImplemented。FixedArray 不认识任何上层类。
- NumPy 函数不能直接作用在 FixedArray 上，也不能隐式转成数组，要 float 请调
  ``to_float()``；``np.sum`` / ``np.mean`` 例外。
- 溢出：全程 int64，不升级成 Python 整数。加减、乘法、求和、均值与矩阵乘法均做真实溢出
  判定：只有最终结果超出 int64 才报错；mantissa 始终是只读 int64 数组。
- ``*`` / ``@``：精确结果，scale 为两侧之和。``mul_round`` / ``matmul``：默认 half_up 舍入到
  max(两侧 scale)，``rounding=None`` 表示精确且不允许缩小 scale。
- ``/``：默认 scale = max(两侧 scale)，half_up；``divide(other, scale, rounding)`` 指定精度。

组织：按源码顺序分节——舍入模式、构造、属性、内部工具、算术、规约、形状、精度、比较、
NumPy 协议、导出、展示，最后是已知缺陷。每节含示例测试与固定种子的随机对拍。
曾经确认的缺陷放在 TestRegressions 中防止回归；新发现的缺陷先用 ``xfail(strict=True)``
固定预期行为，修复后会变成 XPASS 报错，届时删除标记并移入 TestRegressions。
"""

import random
import subprocess
import sys
import types
import warnings
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP, Decimal, \
    localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import moneta.fixed_array as fixed_array_module
from moneta import Money, Quantity
from moneta.fixed_array import FixedArray as F, RoundingMode

MIN, MAX = -(2 ** 63), 2 ** 63 - 1
MODES = [m.value for m in RoundingMode]
DECIMAL_MODES = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN, "down": ROUND_DOWN, "up": ROUND_UP,
                 "floor": ROUND_FLOOR, "ceil": ROUND_CEILING}
SRC = str(Path(__file__).resolve().parents[1] / "src")
AXIS_ERROR = getattr(getattr(np, "exceptions", np), "AxisError")  # NumPy < 1.25 没有 np.exceptions


# ===========================================================================
# 参照实现与工具
# ===========================================================================


def ref_round(x: Fraction, mode: str) -> int:
    """x 按 mode 取整到整数（精确参照）。"""
    floor = x.numerator // x.denominator
    frac = x - floor
    if frac == 0:
        return floor
    if mode == "floor":
        return floor
    if mode == "ceil":
        return floor + 1
    if mode == "down":
        return floor + 1 if x < 0 else floor
    if mode == "up":
        return floor if x < 0 else floor + 1
    half = Fraction(1, 2)
    if frac != half:
        return floor + 1 if frac > half else floor
    if mode == "half_up":
        return floor + 1 if x > 0 else floor
    return floor if floor % 2 == 0 else floor + 1  # half_even


def fits(v: int) -> bool:
    return MIN <= v <= MAX


def value(fa: F) -> Fraction:
    """0 维 FixedArray 的精确值。"""
    return Fraction(int(fa.mantissa), 10 ** fa.scale)


def raw(m, scale: int) -> F:
    """直接由尾数构造（不经解析）。"""
    return F._from_mantissa(np.asarray(m, dtype=np.int64), scale)


def decimals(fa: F):
    """FixedArray → Decimal（0 维）或 Decimal 列表。"""
    if fa.ndim == 0:
        return Decimal(int(fa.mantissa)).scaleb(-fa.scale)
    return [Decimal(int(m)).scaleb(-fa.scale) for m in fa.mantissa.ravel()]


def quantized_units(x: Decimal, scale: int, mode: str = "half_up") -> int:
    """x 舍入到 scale 位小数后的尾数。"""
    with localcontext() as ctx:
        ctx.prec = 100
        return int(x.scaleb(scale).to_integral_value(rounding=DECIMAL_MODES[mode]))


def random_mantissa(rng: random.Random) -> int:
    """偏向边界的随机尾数：int64 极值、小数、各数量级。"""
    if rng.random() < 0.1:
        return rng.choice([MIN, MAX, MIN + 1, MAX - 1, 0, 1, -1, 5, -5])
    digits = rng.choice([1, 3, 9, 10, 12, 15, 18, 19])
    return max(MIN, min(MAX, rng.randint(-(10 ** digits), 10 ** digits)))


def run_isolated(code: str) -> str:
    """在独立进程中运行代码（用于检验不会卡死），返回 stdout。"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20,
                            env={"PYTHONPATH": SRC})
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ===========================================================================
# 1. 舍入模式
# ===========================================================================


class TestRoundingMode:
    def test_members_and_str_compat(self):
        assert MODES == ["half_up", "half_even", "down", "up", "floor", "ceil"]
        assert RoundingMode.HALF_EVEN == "half_even" and isinstance(RoundingMode.UP, str)

    def test_check_rounding(self):
        for mode in RoundingMode:
            assert F._check_rounding(mode) == mode.value
            assert F._check_rounding(mode.value) == mode.value
        for bad in ("nearest", "HALF_UP", "", None, 1):
            with pytest.raises(ValueError, match="rounding 只能是"):
                F._check_rounding(bad)

    @pytest.mark.parametrize("mode", list(RoundingMode))
    def test_every_rounding_api_accepts_enum(self, mode):
        x = F("2.5")
        want = ref_round(Fraction(5, 2), mode.value)
        assert int(x.rescale(0, mode).mantissa) == want
        assert int(x.mul_round(1, scale=0, rounding=mode).mantissa) == want
        assert int(F([["2.5"]]).matmul(F([[1]]), scale=0, rounding=mode).mantissa[0, 0]) == want
        assert int(F(5).divide(2, scale=0, rounding=mode).mantissa) == want
        assert int(F(["2", "3"]).mean(scale=0, rounding=mode).mantissa) == want
        assert x.round_to_step(1, mode) == F(want, scale=1)


# ===========================================================================
# 2. 构造
# ===========================================================================


class TestConstructInt:
    def test_scale_default_zero(self):
        a = F(5)
        assert (int(a.mantissa), a.scale, a.ndim) == (5, 0, 0)

    def test_explicit_scale(self):
        a = F([1, -2], scale=2)
        assert (a.mantissa.tolist(), a.scale) == ([100, -200], 2)
        b = F(12, scale=2)
        assert (int(b.mantissa), b.scale, b.ndim) == (1200, 2, 0)

    def test_numpy_integer_kinds(self):
        assert F(np.array([1, 2], dtype=np.uint8)).mantissa.tolist() == [1, 2]
        assert F(np.int32(-7)).mantissa == -7
        assert F(np.array([MAX], dtype=np.uint64)).mantissa.tolist() == [MAX]
        assert F([]).shape == (0,)

    @pytest.mark.parametrize("value, scale", [(np.array([2 ** 62]), 1), (np.array([2 ** 63], dtype=np.uint64), 0),
                                              (2 ** 63, None), (-(2 ** 63) - 1, None), (1, 19)])
    def test_overflow(self, value, scale):
        with pytest.raises(OverflowError):
            F(value, scale=scale)


class TestConstructFloat:
    # 显式 scale：先取最短写法，再按 half_up 舍入
    def test_explicit_scale_scalar(self):
        assert (int(F(1.5, scale=2).mantissa), F(1.5, scale=2).scale) == (150, 2)
        assert int(F(1.234, scale=2).mantissa) == 123
        assert int(F(0.125, scale=2).mantissa) == 13
        assert int(F(-0.125, scale=2).mantissa) == -13
        assert F(0.125, scale=2) == F("0.125").rescale(2)

    def test_explicit_scale_array_matches_scalar(self):
        values = [1.5, 1.234, -0.07, 12.78, 0.125]
        arr = F(np.array(values), scale=2)
        assert arr.mantissa.tolist() == [int(F(v, scale=2).mantissa) for v in values]

    @pytest.mark.parametrize("value", [1e19, 1e300, np.array([2.0 ** 62])])
    def test_explicit_scale_overflow(self, value):
        with pytest.raises(OverflowError):
            F(value, scale=1)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
    def test_arrays_match_scalar_shortest_repr_random(self, dtype):
        """数组（不指定 / 指定 scale）与逐个标量的最短写法一致，含大数、恰好一半与各种 dtype。"""
        rng = random.Random(77)
        with localcontext() as ctx:
            ctx.prec = 100
            for _ in range(800):
                xs = []
                for _ in range(rng.randint(1, 10)):
                    c = rng.random()
                    if c < 0.3:
                        x = round(rng.uniform(-1e9, 1e9), rng.randint(0, 9))
                    elif c < 0.5:
                        x = (rng.randint(-10 ** 8, 10 ** 8) + 0.5) / 10 ** rng.randint(0, 6)  # 恰好一半
                    elif c < 0.7:
                        x = rng.choice([0.125, 2.675, 1.005, -2.5, 1 / 3, 1e-20, 1e15 + 0.5, 123456789.987654321])
                    else:
                        x = float(f"{rng.randint(1, 10 ** 15)}e{rng.randint(-20, 5)}") * rng.choice([1, -1])
                    xs.append(x)
                with np.errstate(over="ignore"):
                    arr = np.array(xs).astype(dtype)
                arr = arr[np.isfinite(arr)]
                try:
                    a = F(arr)
                except OverflowError:
                    pass
                else:
                    assert a.mantissa.tolist() == [int(F(x).rescale(a.scale).mantissa) for x in arr], arr
                t = rng.randint(0, 10)
                if arr.size and float(np.abs(arr.astype(np.float64)).max()) * 10.0 ** t >= 2.0 ** 63:
                    continue
                want = [int(Decimal(np.format_float_positional(x, unique=True, trim="-")).scaleb(t)
                            .quantize(Decimal(1), rounding=ROUND_HALF_UP)) for x in arr]
                assert F(arr, scale=t).mantissa.tolist() == want, (arr, t)

    def test_float32_large_integers_use_shortest_repr(self):
        # float32 超过 2^24 后相邻值间距 ≥ 2，rint(x) 能往返但不是最短写法
        x = np.float32(-613739648.0)
        assert str(F(np.array([x, np.float32(0.5)]))) == "[-613739650.0, 0.5]"
        assert F(np.array([x]), scale=1).mantissa.tolist() == [-6137396500]
        assert F(np.array([x])) == F(x)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_subnormals(self, dtype):
        info = np.finfo(dtype)
        xs = np.array([info.smallest_subnormal, -info.smallest_subnormal * 3, info.tiny / 2], dtype=dtype)
        with localcontext() as ctx:
            ctx.prec = 400
            for t in (1, 10, 50, 300):
                if float(np.abs(xs.astype(np.float64)).max()) * 10.0 ** t >= 2.0 ** 63:
                    continue
                want = [int(Decimal(np.format_float_positional(x, unique=True, trim="-")).scaleb(t)
                            .quantize(Decimal(1), rounding=ROUND_HALF_UP)) for x in xs]
                assert F(xs, scale=t).mantissa.tolist() == want

    def test_explicit_scale_noisy_values_vectorized(self):
        rng = np.random.default_rng(42)
        xs = rng.random(20000) * 2000 - 1000  # 计算得到的浮点数：每个都需要舍入
        got = F(xs, scale=2).mantissa.tolist()
        assert got == [int(F(float(x), scale=2).mantissa) for x in xs]

    def test_round_float_to_scale(self):
        # 远离 k + 1/2：直接由 float 计算；贴近 k + 1/2、过大或次正规：逐个精确处理
        vals = np.array([1.234, -1.236, 2.675, 0.125, -0.125, 1e15 + 0.5, 5e-324])
        assert F._round_float_to_scale(vals, 2).tolist() == [123, -124, 268, 13, -13, 100000000000000050, 0]
        with pytest.raises(OverflowError):
            F._round_float_to_scale(np.array([9e18]), 1)

    def test_align_mantissas(self):
        m = np.array([1, -25, 0], dtype=np.int64)
        assert F._align_mantissas(m, np.array([0, 1, 3]), 3).tolist() == [1000, -2500, 0]
        assert F._align_mantissas(np.array([0, 5]), np.array([0, 20]), 25).tolist() == [0, 500000]  # 0 可位移任意位
        with pytest.raises(OverflowError):
            F._align_mantissas(np.array([1]), np.array([0]), 19)
        with pytest.raises(OverflowError):
            F._align_mantissas(np.array([MAX // 10 + 1]), np.array([0]), 1)

    def test_explicit_scale_large_values_use_shortest_repr(self):
        # x·10^s 超过 2^53 的元素无法用 float 试探，改用最短写法再对齐到目标 scale
        a = F(np.array([1e15 + 0.5, 1.0]), scale=2)
        assert a.mantissa.tolist() == [100000000000000050, 100]

    def test_explicit_scale_random_equals_decimal_half_up(self):
        rng = random.Random(12)
        for _ in range(3000):
            s = rng.randint(0, 6)
            x = round(rng.uniform(-1e6, 1e6), rng.randint(0, 8))
            want = int(Decimal(repr(x)).scaleb(s).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            assert int(F(x, scale=s).mantissa) == want, (x, s)
            assert int(F(np.array([x, 1.5]), scale=s).mantissa[0]) == want, (x, s)

    # 未知 scale 的标量：最短写法
    def test_scalar_shortest_repr(self):
        assert decimals(F(1.1)) == Decimal("1.1") and F(1.1).scale == 1
        assert decimals(F(-0.005)) == Decimal("-0.005")
        assert F(0.1 + 0.2).scale == 17  # 最短写法是 0.30000000000000004
        assert F(np.float32(1.1)).scale == 1
        assert F(np.array(1.1, dtype=np.float32)).scale == 1  # 0 维数组同样按 float32 精度
        assert (int(F(1.0).mantissa), F(1.0).scale) == (1, 0)

    # 未知 scale 的数组：逐位试探 + 往返检验
    def test_array_probe(self):
        b = F(np.array([12.78, 1.15, 100.0]))
        assert (b.mantissa.tolist(), b.scale) == ([1278, 115, 10000], 2)
        c = F(np.array([1.0, 2.0]))
        assert (c.mantissa.tolist(), c.scale) == ([1, 2], 0)
        assert F(np.array([], dtype=np.float64)).scale == 0

    def test_array_probe_only_failing_elements(self):
        # 少数元素需要更多小数位时，只对未通过的元素继续试探
        values = [1.0] * 30 + [0.125, 1.5]
        a = F(np.array(values))
        assert a.scale == 3 and a.mantissa.tolist() == [1000] * 30 + [125, 1500]

    def test_array_round_trip_keeps_noise(self):
        a = F(np.array([0.1 + 0.2, 1.25]))
        assert (a.mantissa.tolist(), a.scale) == ([30000000000000004, 125000000000000000], 17)

    @pytest.mark.parametrize("value", [0.1 + 0.2, 1 / 3, 12.78, -0.07, 1e-5, 123456.789, 1.0, 0.0, -0.0, 2.5e-10])
    def test_array_matches_scalar(self, value):
        scalar, array = F(value), F(np.array([value]))
        assert (int(array.mantissa[0]), array.scale) == (int(scalar.mantissa), scalar.scale)

    def test_array_large_magnitude_keeps_decimals(self):
        a = F(np.array([1.5e12 + 0.01]))
        assert (a.mantissa.tolist(), a.scale) == ([150000000000001], 2)
        b = F(np.array([3e13 + 0.25]))
        assert (b.mantissa.tolist(), b.scale) == ([3000000000000025], 2)

    def test_array_float32(self):
        a = F(np.array([12.78, 1.15, -0.5], dtype=np.float32))
        assert (a.mantissa.tolist(), a.scale) == ([1278, 115, -50], 2)

    def test_array_beyond_float_exact_range(self):
        # 需要 22 位以上小数的元素改用最短写法逐个解析
        a = F(np.array([2.5e-25, 0.0]))
        assert (a.mantissa.tolist(), a.scale) == ([25, 0], 26)
        b = F(np.array([2.5e-25] * 20))  # 整列试探到 22 位仍未通过
        assert (b.mantissa.tolist(), b.scale) == ([25] * 20, 26)

    def test_array_2d_shape_kept(self):
        a = F(np.array([[1.5, 2.25], [3.0, -4.125]]))
        assert (a.shape, a.scale, a.mantissa.tolist()) == ((2, 2), 3, [[1500, 2250], [3000, -4125]])

    @pytest.mark.parametrize("values", [
        [100000.0, 1 / 3],  # 试探中途最大值放大越界
        [8.92790820483521e-26, 0.00027510801534, -4.453e-10],  # 对齐时位数差超过 18
        [1.6234040368018e-21, -9e-15, 0.0006441646],  # 对齐时乘积越界
        [-8.590709051494e-05, 6.411e-24, -8.07043246904e-10],  # 对齐时乘积越界
        [-1.33446048939e-21, 1.94767651803e-25, 6.53137899815e-12],  # 逐个解析的元素对齐越界
    ])
    def test_array_alignment_overflow(self, values):
        with pytest.raises(OverflowError):
            F(np.array(values))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.array([1.0, np.inf]), np.array([np.nan])])
    def test_non_finite(self, value):
        with pytest.raises(ValueError):
            F(value)
        with pytest.raises(ValueError):
            F(value, scale=2)


class TestConstructExact:
    @pytest.mark.parametrize("text, mantissa, scale", [
        ("12.78", 1278, 2), ("-0.005", -5, 3), ("1.20", 120, 2), ("1e2", 100, 0), (" 7 ", 7, 0), ("0", 0, 0),
        ("0.00", 0, 2), ("-0.000", 0, 3), ("0e5", 0, 0), ("1.5E-3", 15, 4), (str(MAX), MAX, 0),
    ])
    def test_str(self, text, mantissa, scale):
        a = F(text)
        assert (int(a.mantissa), a.scale) == (mantissa, scale)

    def test_decimal_and_fraction(self):
        assert (int(F(Decimal("3.14")).mantissa), F(Decimal("3.14")).scale) == (314, 2)
        assert (int(F(Fraction(1, 8)).mantissa), F(Fraction(1, 8)).scale) == (125, 3)
        assert (int(F(Fraction(1, 5)).mantissa), F(Fraction(1, 5)).scale) == (2, 1)
        assert (int(F(Fraction(-3, 40)).mantissa), F(Fraction(-3, 40)).scale) == (-75, 3)
        assert (int(F(Fraction(6, 1)).mantissa), F(Fraction(6, 1)).scale) == (6, 0)
        with pytest.raises(ValueError, match="不是有限小数"):
            F(Fraction(1, 3))

    def test_arrays(self):
        z = F(["0.00", "1"])
        assert (z.mantissa.tolist(), z.scale) == ([0, 100], 2)
        a = F(["1.5", "2.25"])
        assert (a.mantissa.tolist(), a.scale) == ([150, 225], 2)
        repeated = F(np.array(["1.1", "2.2", "1.1"] * 1000))
        assert repeated.mantissa.tolist() == [11, 22, 11] * 1000

    def test_decimal_trailing_zeros_order_independent(self):
        for values in ([Decimal("1.1"), Decimal("1.100")], [Decimal("1.100"), Decimal("1.1")]):
            a = F(np.array(values, dtype=object))
            assert (a.scale, a.mantissa.tolist()) == (3, [1100, 1100])

    def test_mixed_object_arrays(self):
        a = F(np.array([1.5, "2"], dtype=object))
        assert (a.mantissa.tolist(), a.scale) == ([15, 20], 1)
        b = F(np.array([Fraction(1, 4), 2, Decimal("0.5"), np.int16(3), np.float32(0.5)], dtype=object))
        assert (b.mantissa.tolist(), b.scale) == ([25, 200, 50, 300, 50], 2)
        c = F(np.array([Decimal("1"), Decimal("1"), 2], dtype=object))  # 重复值只解析一次
        assert c.mantissa.tolist() == [1, 1, 2]

    def test_explicit_scale_pads_or_rejects(self):
        assert F(["1.5", "2.25"], scale=3).mantissa.tolist() == [1500, 2250]
        assert int(F("1.5", scale=3).mantissa) == 1500
        with pytest.raises(ValueError, match="超出指定 scale"):
            F("1.234", scale=2)
        with pytest.raises(ValueError):
            F(["1.234"], scale=2)

    @pytest.mark.parametrize("value", ["abc", "nan", "inf", "-Infinity", "", ["1", "x"],
                                       np.array([Decimal("1"), "x"], dtype=object)])
    def test_unparseable(self, value):
        with pytest.raises(ValueError):
            F(value)

    @pytest.mark.parametrize("value, scale", [("9" * 20, None), (["1e30"], None), (["10"], 18), ("10", 18),
                                              ("1e20", None), (["0.0000000001", "1000000000"], None)])
    def test_overflow(self, value, scale):
        with pytest.raises(OverflowError):
            F(value, scale=scale)

    def test_scale_18_fits(self):
        assert int(F("1.5", scale=18).mantissa) == 15 * 10 ** 17

    def test_huge_exponent_fails_fast(self):
        code = "from moneta.fixed_array import FixedArray as F\ntry:\n    F('1e999999999')\nexcept OverflowError:\n    print('ok')"
        assert run_isolated(code) == "ok"

    def test_zero_with_huge_scale_is_fast(self):
        code = ("from moneta.fixed_array import FixedArray as F\n"
                "a = F(['0e-999999999'])\n"
                "assert a.scale == 999999999 and a.mantissa.tolist() == [0]\n"
                "try:\n    F(['0e-999999999', '1'])\nexcept OverflowError:\n    print('ok')")
        assert run_isolated(code) == "ok"


class TestConstructValidation:
    @pytest.mark.parametrize("scale", [-1, 2.0, "2", True, np.bool_(True), np.int64(-1)])
    def test_scale_must_be_non_negative_int(self, scale):
        for make in (lambda: F(1, scale=scale), lambda: F(1.5, scale=scale), lambda: F("1", scale=scale),
                     lambda: F(F("1"), scale=scale), lambda: F("1").rescale(scale),
                     lambda: F("1").divide(3, scale=scale), lambda: F("1").mul_round(3, scale=scale),
                     lambda: F([[1]]).matmul(F([[1]]), scale=scale), lambda: F([1]).mean(scale=scale)):
            with pytest.raises(ValueError, match="scale 必须是非负整数"):
                make()

    def test_scale_accepts_numpy_integer_and_none(self):
        assert F(1, scale=np.int64(2)).scale == 2
        assert F(1, scale=None).scale == 0

    @pytest.mark.parametrize("value", [True, np.bool_(False), np.array([True, False]), np.array([True], dtype=object)])
    def test_bool_rejected(self, value):
        with pytest.raises(TypeError):
            F(value)

    @pytest.mark.parametrize("value", [1 + 2j, np.array([1j]), np.array(["2026-01-01"], dtype="datetime64[D]"),
                                       np.array([1], dtype="timedelta64[s]"), b"1"])
    def test_unsupported_dtype(self, value):
        with pytest.raises(TypeError, match="不支持从"):
            F(value)


class TestCopyAndImmutability:
    def test_copy_constructor(self):
        a = F("1.5")
        b = F(a)
        assert b == a and b.scale == 1
        assert int(F(F("1.25"), scale=1).mantissa) == 13  # half_up
        assert int(F(F("1.5"), scale=3).mantissa) == 1500

    def test_mantissa_readonly(self):
        for fa in (F(["1.5"]), F(F(["1.5"])), F(["1.5"]) + F(["1"]), F(["1.5"]) * 2, F(np.array([1.5]), scale=2)):
            with pytest.raises(ValueError):
                fa.mantissa[0] = 999

    def test_copy_rules(self):
        # 公开构造：不与调用方的数组共享内存
        src = np.array([1, 2], dtype=np.int64)
        a = F(src)
        assert not np.may_share_memory(a.mantissa, src) and src.flags.writeable
        src[0] = 99
        assert a.mantissa.tolist() == [1, 2]
        # 复制构造：尾数只读，直接共享
        assert np.may_share_memory(F(a).mantissa, a.mantissa)
        # 内部零拷贝：共享内存，但不锁调用方的数组
        raw_src = np.array([5], dtype=np.int64)
        c = F._from_mantissa(raw_src, 1)
        assert np.may_share_memory(c.mantissa, raw_src) and raw_src.flags.writeable

    def test_every_path_yields_readonly_int64_array(self):
        cases = [F(12), F(12, scale=2), F([1, 2]), F(1.5), F(1.5, scale=2), F(np.array([1.5])), F("1.5"),
                 F(["1.5"]), F(Fraction(1, 8)), F(np.array([Decimal("1")], dtype=object)), F(F("1.5")),
                 F(F("1.25"), scale=1), F._from_mantissa(np.int64(3), 0), F._from_mantissa(3, 0)]
        for fa in cases:
            assert isinstance(fa.mantissa, np.ndarray) and fa.dtype == np.int64
            assert not fa.mantissa.flags.writeable

    def test_from_mantissa_rejects_non_integer(self):
        for bad in (np.array([1.5]), np.array([1], dtype=object), True):
            with pytest.raises(TypeError):
                F._from_mantissa(bad, 0)
        with pytest.raises(OverflowError):
            F._from_mantissa(2 ** 63, 0)

    def test_set_is_the_only_writer(self):
        a = F.__new__(F)
        a._set(np.int64(7), 2)
        assert (int(a.mantissa), a.scale, a.mantissa.flags.writeable) == (7, 2, False)

    def test_slots_and_unhashable(self):
        with pytest.raises(AttributeError):
            F("1").extra = 1
        with pytest.raises(TypeError):
            hash(F("1"))


# ===========================================================================
# 3. 属性
# ===========================================================================


class TestProperties:
    def test_basic(self):
        a = F([["1.5", "2"], ["3", "4"]])
        assert (a.shape, a.ndim, a.size, a.dtype, a.scale) == ((2, 2), 2, 4, np.int64, 1)
        assert a.mantissa.tolist() == [[15, 20], [30, 40]]

    def test_scalar(self):
        a = F("-0.5")
        assert (a.shape, a.ndim, a.size, a.scale, int(a.mantissa)) == ((), 0, 1, 1, -5)

    def test_results_stay_int64(self):
        a = F(["1.25", "-2.50", "3.75"])
        results = [a.rescale(1), a / 3, a.divide(7, scale=2), a.mean(), a.sum(), a @ F(["1", "2", "3"]),
                   a.matmul(F(["1", "2", "3"])), a.mul_round(F("0.5")), a.round_to_step(F("0.5")),
                   a + a, a - a, a * a, a * 3, -a, abs(a), a % F("0.3"), F(np.array([1.5, 2.5]), scale=1)]
        for r in results:
            assert r.dtype == np.int64


# ===========================================================================
# 4. 内部工具
# ===========================================================================


class TestOperandWhitelist:
    @pytest.mark.parametrize("ok", [1, np.int32(2), np.uint8(3), 1.5, np.float32(0.5), Decimal("0.25"),
                                    Fraction(1, 8), F("1"), [1, Decimal("2")], (1, 2), [[1], [2]], [],
                                    np.array([1, 2]), np.array([1.5]), np.array([Decimal("1"), 2], dtype=object)])
    def test_is_numeric_operand_accepts(self, ok):
        assert F._is_numeric_operand(ok) is True
        assert isinstance(F._coerce(ok), F)
        assert isinstance(F("1") + ok, F)

    @pytest.mark.parametrize("bad", ["1", b"1", True, np.bool_(True), None, object(), {"a": 1}, {1},
                                     np.array(["1"]), np.array([b"1"]), np.array(["1"], dtype=object),
                                     np.array([True]), ["1", 2], (1, "2"), [1, [2, "3"]], np.array([1 + 2j]),
                                     1 + 2j, Quantity(1, unit=""), Money("1", "USD")])
    def test_is_numeric_operand_rejects(self, bad):
        assert F._is_numeric_operand(bad) is False
        assert F._coerce(bad, strict=False) is None
        with pytest.raises(TypeError, match="只能与数值类型"):
            F._coerce(bad)

    @pytest.mark.parametrize("bad", ["1", b"1", True, None, object(), np.array(["1"]), np.array(["1"], dtype=object),
                                     ["1", 2], (1, "2"), np.array([1 + 2j])])
    def test_arithmetic_rejects(self, bad):
        for op in (lambda: F("1") + bad, lambda: bad + F("1"), lambda: F("1") - bad, lambda: bad - F("1"),
                   lambda: F("1") * bad, lambda: F("1") @ bad, lambda: F("1") / bad, lambda: F("1") // bad,
                   lambda: F("1") % bad, lambda: divmod(F("1"), bad)):
            with pytest.raises(TypeError):
                op()

    @pytest.mark.parametrize("bad", ["1", True, None, Quantity(1, unit="")])
    def test_comparison_rejects(self, bad):
        assert (F("1") == bad) is False
        assert (F("1") != bad) is True
        for op in (lambda: F("1") < bad, lambda: F("1") <= bad, lambda: F("1") > bad, lambda: F("1") >= bad):
            with pytest.raises(TypeError):
                op()

    @pytest.mark.parametrize("bad", ["0.05", True, [1, "2"], np.array(["0.05"], dtype=object)])
    def test_explicit_methods_reject(self, bad):
        for call in (lambda: F("1").mul_round(bad), lambda: F([1]).matmul(bad), lambda: F("1").divide(bad),
                     lambda: F("1").round_to_step(bad), lambda: F("1").is_on_step(bad)):
            with pytest.raises(TypeError):
                call()

    def test_unknown_classes_get_reflected_operation(self):
        class Custom:
            def __rmul__(self, other):
                return "rmul"

            def __radd__(self, other):
                return "radd"

        assert F("10.5") * Custom() == "rmul"
        assert F("10.5") + Custom() == "radd"
        with pytest.raises(TypeError, match="unsupported operand type"):
            F("10.5") * object()

    def test_unit_objects_never_degrade(self):
        # str() 恰好能解析成数字（unit 为空）也不能当成无量纲数参与运算
        with pytest.raises(TypeError):
            F("1") + Quantity(2, unit="")
        assert F("2") * Money("3", "USD") == Money("6", "USD")  # 交给 Money.__rmul__
        with pytest.raises(TypeError, match="Money"):
            F("1") < Money("2", "USD")  # 交给 Money.__gt__，由 Money 报错

    def test_invalid_numeric_values_raise_their_own_error(self):
        with pytest.raises(ValueError):
            F("1") + Fraction(1, 3)
        with pytest.raises(ValueError):
            F("1") + float("nan")
        with pytest.raises(ValueError):
            F("1") + [1, [2]]  # 不规则嵌套
        with pytest.raises(OverflowError):
            F("1") + 10 ** 30
        assert (F("1") == float("nan")) is False
        assert (F("1") == 10 ** 30) is False


class TestIntegerHelpers:
    def test_as_int64(self):
        assert F._as_int64(5).dtype == np.int64 and F._as_int64(5).ndim == 0
        assert F._as_int64(np.array([1, 2], dtype=np.uint16)).dtype == np.int64
        arr = np.array([1, 2], dtype=np.int64)
        assert F._as_int64(arr) is arr  # 已是 int64 不复制
        assert F._as_int64(np.array([MAX], dtype=np.uint64)).tolist() == [MAX]
        for bad in (True, 1.5, np.array([1.5]), np.array([1], dtype=object), "1"):
            with pytest.raises(TypeError, match="只接受整数"):
                F._as_int64(bad)
        for big in (2 ** 63, MIN - 1, np.array([2 ** 63], dtype=np.uint64)):
            with pytest.raises(OverflowError):
                F._as_int64(big)

    def test_check_scale(self):
        assert F._check_scale(0) == 0 and F._check_scale(np.int8(3)) == 3
        assert type(F._check_scale(np.int64(3))) is int

    def test_max_abs(self):
        assert F._max_abs(np.array([MIN, 5])) == 2 ** 63
        assert F._max_abs(np.array([-3, 7])) == 7
        assert F._max_abs(np.array([], dtype=np.int64)) == 0
        assert F._max_abs(np.int64(-4)) == 4

    def test_mul_pow10(self):
        x = np.array([1, -2], dtype=np.int64)
        assert F._mul_pow10(x, 0) is x
        assert F._mul_pow10(x, 3).tolist() == [1000, -2000]
        assert F._mul_pow10(np.array([MAX // 10]), 1).tolist() == [MAX // 10 * 10]
        with pytest.raises(OverflowError):
            F._mul_pow10(np.array([MAX // 10 + 1]), 1)
        with pytest.raises(OverflowError):
            F._mul_pow10(np.array([1]), 19)

    def test_check_division(self):
        F._check_division(np.array([1]), np.array([2]))
        with pytest.raises(ZeroDivisionError):
            F._check_division(np.array([1, 2]), np.array([1, 0]))
        with pytest.raises(OverflowError):
            F._check_division(np.array([MIN]), np.array([-1]))

    def test_align_scales(self):
        a, b = F("1.5"), F("0.25")
        x, y = F._align_scales(a, b)
        assert (x.scale, y.scale, int(x.mantissa), int(y.mantissa)) == (2, 2, 150, 25)
        assert F._align_scales(a, a) == (a, a)
        with pytest.raises(OverflowError):
            F._align_scales(raw(MAX, 0), F("0.1"))


class TestDecideBump:
    """统一舍入裁决：Truncation Base（作用于 |q|）与 Floor Base（作用于有符号 q）。"""

    @pytest.mark.parametrize("mode", MODES)
    def test_against_reference_on_halves_grid(self, mode):
        # 以 x = 整数部分 + 小数部分 的全部组合检验两种坐标系
        for whole in (-3, -2, -1, 0, 1, 2):
            for frac in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
                x = whole + frac if whole >= 0 else whole - frac
                want = ref_round(x, mode)
                # Floor Base：q = floor(x)
                q = x.numerator // x.denominator
                r = x - q
                bump = F._decide_bump(mode, is_negative=np.bool_(q < 0), odd_q=np.bool_(q % 2 != 0),
                                      has_fraction=np.bool_(r != 0), over_half=np.bool_(r > Fraction(1, 2)),
                                      exact_half=np.bool_(r == Fraction(1, 2)), floor_base=True)
                assert q + int(bump) == want, (mode, x)
                # Truncation Base：作用于 |x| 的截断值
                t = abs(x).numerator // abs(x).denominator
                rt = abs(x) - t
                bump_t = F._decide_bump(mode, is_negative=np.bool_(x < 0), odd_q=np.bool_(t % 2 != 0),
                                        has_fraction=np.bool_(rt != 0), over_half=np.bool_(rt > Fraction(1, 2)),
                                        exact_half=np.bool_(rt == Fraction(1, 2)), floor_base=False)
                magnitude = t + int(bump_t)
                assert (magnitude if x >= 0 else -magnitude) == want, (mode, x)

    @pytest.mark.parametrize("floor_base", [True, False])
    def test_invalid_mode(self, floor_base):
        with pytest.raises(ValueError, match="rounding 只能是"):
            F._decide_bump("unknown", is_negative=False, odd_q=False, has_fraction=True, over_half=False,
                           exact_half=False, floor_base=floor_base)


class TestDivmodRound:
    """_divmod_round(m, d, mode)：round(m / d)，m、d 为 int64 范围内的整数或数组，d 可正可负。"""

    @pytest.mark.parametrize("mode", MODES)
    def test_small_grid_array_and_scalar(self, mode):
        ms = np.arange(-25, 26, dtype=np.int64)
        ds = np.array([-7, -4, -2, -1, 1, 2, 4, 7], dtype=np.int64)
        result = F._divmod_round(ms[:, None], ds[None, :], mode)
        assert result.dtype == np.int64
        assert result.tolist() == [[ref_round(Fraction(int(m), int(d)), mode) for d in ds] for m in ms]
        for m in (-25, -7, -1, 0, 3, 14, 25):
            for d in (-4, -1, 2, 7):
                got = F._divmod_round(m, d, mode)
                assert type(got) is int and got == ref_round(Fraction(m, d), mode)

    @pytest.mark.parametrize("mode", MODES)
    def test_int64_extremes_are_exact(self, mode):
        ms = [MIN, MIN + 1, -3, -1, 0, 1, 3, MAX - 1, MAX]
        ds = [MIN, MIN + 1, -3, -2, -1, 1, 2, 3, MAX]
        pairs = [(m, d) for m in ms for d in ds if (m, d) != (MIN, -1)]
        result = F._divmod_round(np.array([m for m, _ in pairs]), np.array([d for _, d in pairs]), mode)
        assert result.tolist() == [ref_round(Fraction(m, d), mode) for m, d in pairs]

    def test_round_quotient_floor_short_circuit(self):
        q, r = np.divmod(np.array([7, -7]), 2)
        assert F._round_quotient(q, r, np.int64(2), "floor").tolist() == [3, -4]

    def test_errors(self):
        with pytest.raises(OverflowError):
            F._divmod_round(np.array([MIN]), np.array([-1]), "floor")
        with pytest.raises(OverflowError):
            F._divmod_round(MIN, -1, "half_up")
        with pytest.raises(ZeroDivisionError):
            F._divmod_round(5, 0, "half_up")
        with pytest.raises(ZeroDivisionError):
            F._divmod_round(np.array([1, 2]), np.array([1, 0]), "half_up")
        for d in (1, 10):
            with pytest.raises(ValueError):
                F._divmod_round(np.array([10]), d, "nearest")
        with pytest.raises(OverflowError):
            F._divmod_round(2 ** 63, 3, "half_up")
        assert F._divmod_round(np.array([5, -5, 15]), 10, "half_up").tolist() == [1, -1, 2]


class TestChunkedKernels:
    """分块内核与 Python 整数精确对拍。"""

    def test_chunked_divmod_round_random(self):
        rng = random.Random(21)
        for _ in range(2000):
            m = random_mantissa(rng)
            d = rng.choice([1, -1]) * rng.randint(1, 10 ** rng.choice([1, 6, 12, 17]))
            k, mode = rng.randint(0, 30), rng.choice(MODES)
            want = ref_round(Fraction(m * 10 ** k, d), mode)
            if fits(want):
                assert int(F._chunked_divmod_round(np.int64(m), np.int64(d), k, mode)) == want, (m, d, k, mode)
            else:
                with pytest.raises(OverflowError):
                    F._chunked_divmod_round(np.int64(m), np.int64(d), k, mode)

    def test_chunked_divmod_round_huge_divisor_random(self):
        # |d| > MAX // 10 时改走倍增内核（10r = 8r + 2r）
        rng = random.Random(26)
        for _ in range(2000):
            m = random_mantissa(rng)
            d = rng.choice([1, -1]) * rng.randint(MAX // 10 + 1, MAX)
            if rng.random() < 0.1:
                d = rng.choice([MIN, MAX, MIN + 1])
            k, mode = rng.randint(0, 25), rng.choice(MODES)
            want = ref_round(Fraction(m * 10 ** k, d), mode)
            if fits(want):
                assert int(F._chunked_divmod_round(np.int64(m), np.int64(d), k, mode)) == want, (m, d, k, mode)
            else:
                with pytest.raises(OverflowError):
                    F._chunked_divmod_round(np.int64(m), np.int64(d), k, mode)

    def test_doubling_step_10r_random(self):
        rng = random.Random(27)
        for _ in range(3000):
            d = rng.choice(
                [MIN, MAX, rng.randint(MIN, -1), rng.randint(1, MAX), rng.randint(-1000, -1), rng.randint(1, 1000)])
            mag = rng.randint(0, abs(d) - 1)
            r = mag if d > 0 else -mag  # 余数与除数同号，|r| < |d|
            digits, new_r = F._doubling_step_10r(np.array([r], dtype=np.int64), np.array([d], dtype=np.int64))
            want_digit, want_r = divmod(10 * r, d)
            assert (int(digits[0]), int(new_r[0])) == (want_digit, want_r), (r, d)

    def test_round_div_pow10_random(self):
        """round((q + r/d) / 10^k)：覆盖 k ≤ 18、k = 19（需与 ±5·10^18 比较）与 k ≥ 20。"""
        rng = random.Random(28)
        half = 5 * 10 ** 18
        for _ in range(20000):
            k = rng.choice([1, 9, 17, 18, 19, 19, 19, 20, 25])
            q = rng.choice([MIN, MAX, 0, -1, 1, half, -half, half - 1, -half + 1, half + 1, -half - 1,
                            rng.randint(MIN, MAX), rng.randint(-(10 ** min(k, 18)), 10 ** min(k, 18))])
            d = rng.choice([1, 2, 3, 7, -1, -2, -3, -7, 10 ** 18, -(10 ** 18), MAX, MIN])
            mag = min(rng.choice([0, 1, (abs(d) - 1) // 2, abs(d) // 2, abs(d) - 1, rng.randint(0, abs(d) - 1)]),
                      abs(d) - 1)
            r = mag if d > 0 else -mag
            mode = rng.choice(MODES)
            want = ref_round((q + Fraction(r, d)) / 10 ** k, mode)
            got = F._round_div_pow10(np.int64(q), np.int64(r), np.int64(d), k, mode)
            assert int(got) == want, (q, r, d, k, mode)

    def test_round_div_pow10_k0_delegates(self):
        q, r = np.divmod(np.array([7, -7]), 2)
        assert F._round_div_pow10(q, r, np.int64(2), 0, "half_even").tolist() == [4, -4]

    def test_chunked_divmod_round_errors(self):
        with pytest.raises(ZeroDivisionError):
            F._chunked_divmod_round(np.array([1]), np.array([0]), 3, "half_up")
        with pytest.raises(OverflowError, match="除法结果超出"):
            F._chunked_divmod_round(np.array([MAX]), np.array([1]), 1, "half_up")

    def test_chunked_mul_round_random(self):
        rng = random.Random(22)
        for _ in range(3000):
            a, b = random_mantissa(rng), random_mantissa(rng)
            shift, mode = rng.randint(-20, 45), rng.choice(MODES)
            want = ref_round(Fraction(a * b) * Fraction(10) ** -shift, mode)
            if fits(want):
                got = F._chunked_mul_round(np.int64(a), np.int64(b), shift, mode)
                assert int(got) == want, (a, b, shift, mode)
            else:
                with pytest.raises(OverflowError):
                    F._chunked_mul_round(np.int64(a), np.int64(b), shift, mode)

    @pytest.mark.parametrize("shift", [9, 18, 27, 54, 63, 81])
    @pytest.mark.parametrize("mode", MODES)
    def test_chunked_mul_round_block_boundaries(self, shift, mode):
        # shift 为 9 的倍数时，舍入位恰好落在 10^9 块边界；≥ 54 时商的所有块都被舍去
        for a, b in [(10 ** 18 + 5 * 10 ** 17, 7), (-(10 ** 18) - 5 * 10 ** 17, 3), (MAX, MAX), (MIN, 3),
                     (5, 10 ** 17), (1, 1), (-1, 1)]:
            want = ref_round(Fraction(a * b, 10 ** shift), mode)
            if fits(want):
                assert int(F._chunked_mul_round(np.int64(a), np.int64(b), shift, mode)) == want
            else:
                with pytest.raises(OverflowError):
                    F._chunked_mul_round(np.int64(a), np.int64(b), shift, mode)

    def test_chunked_mul_round_broadcast(self):
        got = F._chunked_mul_round(np.array([[1], [2]]), np.array([10 ** 18, -(10 ** 18)]), 18, "half_up")
        assert got.tolist() == [[1, -1], [2, -2]]

    def test_chunked_sum_random(self):
        rng = random.Random(23)
        for _ in range(1500):
            rows = [[random_mantissa(rng) for _ in range(rng.randint(1, 5))]] * rng.randint(1, 4)
            arr = np.array(rows, dtype=np.int64)
            for axis in (None, 0, 1, (0, 1)):
                if axis in (None, (0, 1)):
                    want = [sum(map(sum, rows))]
                elif axis == 0:
                    want = [sum(col) for col in zip(*rows)]
                else:
                    want = [sum(r) for r in rows]
                if all(map(fits, want)):
                    assert np.ravel(F._chunked_sum(arr, axis=axis)).tolist() == want
                else:
                    with pytest.raises(OverflowError):
                        F._chunked_sum(arr, axis=axis)

    def test_chunked_sum_boundaries(self):
        assert int(F._chunked_sum(np.array([MAX, 0]))) == MAX
        assert int(F._chunked_sum(np.array([MIN, 0]))) == MIN
        assert F._chunked_sum(np.array([[MAX], [MIN]]), axis=1, keepdims=True).tolist() == [[MAX], [MIN]]
        for bad in ([MAX, 1], [MIN, -1]):
            with pytest.raises(OverflowError):
                F._chunked_sum(np.array(bad))

    def test_chunked_matmul_random(self):
        rng = random.Random(24)
        for _ in range(1500):
            n, k, p = rng.randint(1, 3), rng.randint(1, 5), rng.randint(1, 3)
            a = [[random_mantissa(rng) for _ in range(k)] for _ in range(n)]
            b = [[random_mantissa(rng) for _ in range(p)] for _ in range(k)]
            want = [sum(a[i][t] * b[t][j] for t in range(k)) for i in range(n) for j in range(p)]
            if all(map(fits, want)):
                got = F._chunked_matmul(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
                assert got.ravel().tolist() == want
            else:
                with pytest.raises(OverflowError):
                    F._chunked_matmul(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))

    def test_chunked_matmul_long_inner_dimension(self):
        # 内积长度 2^18 的极端块值，结果放得下
        rng = np.random.default_rng(25)
        pool = np.array([-1, MIN, MAX, -(2 ** 42), 2 ** 42 - 1, (2 ** 21 - 1) << 21, -(2 ** 21)], dtype=np.int64)
        k = 2 ** 18
        a = pool[rng.integers(0, len(pool), k)]
        b = pool[rng.integers(0, len(pool), k)].copy()
        b[rng.random(k) < 0.99999] = 0
        want = int(np.dot(a.astype(object), b.astype(object)))
        assert fits(want)
        assert int(F._chunked_matmul(a, b)) == want


# ===========================================================================
# 5. 加减与符号
# ===========================================================================


class TestAddSub:
    def test_align_scales(self):
        r = F("1.5") + F("0.25")
        assert (int(r.mantissa), r.scale) == (175, 2)
        assert decimals(F("1.5") - F("0.25")) == Decimal("1.25")

    def test_mixed_operands(self):
        assert decimals(F("1.5") + 1) == Decimal("2.5")
        assert decimals(1 + F("1.5")) == Decimal("2.5")
        assert decimals(3 - F("0.5")) == Decimal("2.5")
        assert decimals(F("1.5") + Decimal("0.01")) == Decimal("1.51")
        assert decimals(F("1.5") - Fraction(1, 4)) == Decimal("1.25")
        assert ([1, 2] + F(["0.5", "0.5"])).mantissa.tolist() == [15, 25]
        assert (np.array([10, 10]) - F(["2", "4"])).mantissa.tolist() == [8, 6]

    def test_broadcast(self):
        r = F([["1"], ["2"]]) + F(["0.1", "0.2", "0.3"])
        assert r.shape == (2, 3) and r.mantissa.tolist() == [[11, 12, 13], [21, 22, 23]]

    def test_against_decimal(self):
        rng = np.random.default_rng(1)
        a_texts = [f"{x:.3f}" for x in rng.uniform(-1e6, 1e6, 200)]
        b_texts = [f"{x:.5f}" for x in rng.uniform(-1e6, 1e6, 200)]
        a, b = F(a_texts), F(b_texts)
        assert decimals(a + b) == [Decimal(x) + Decimal(y) for x, y in zip(a_texts, b_texts)]
        assert decimals(a - b) == [Decimal(x) - Decimal(y) for x, y in zip(a_texts, b_texts)]

    def test_random_overflow_is_exact(self):
        rng = random.Random(4)
        for _ in range(3000):
            a, b = random_mantissa(rng), random_mantissa(rng)
            for op, want in ((lambda x, y: x + y, a + b), (lambda x, y: x - y, a - b)):
                if fits(want):
                    assert int(op(raw(a, 2), raw(b, 2)).mantissa) == want
                else:
                    with pytest.raises(OverflowError):
                        op(raw(a, 2), raw(b, 2))

    def test_overflow(self):
        with pytest.raises(OverflowError):
            F(np.array([MAX])) + F(np.array([1]))
        with pytest.raises(OverflowError):
            F(np.array([MIN])) - F(np.array([1]))
        with pytest.raises(OverflowError):
            5 - raw(MIN, 0)
        with pytest.raises(OverflowError):
            F(np.array([10 ** 17])) + F(["0.01"])  # 对齐 scale 时溢出
        r = F(np.array([MAX, -MAX])) + F(np.array([-1, 1]))
        assert r.mantissa.tolist() == [MAX - 1, -MAX + 1]

    def test_neg_abs(self):
        a = F(["-1.5", "2"])
        assert (-a).mantissa.tolist() == [15, -20] and (-a).scale == 1
        assert abs(a).mantissa.tolist() == [15, 20]
        assert int((-raw(MAX, 0)).mantissa) == -MAX
        for op in (lambda x: -x, abs):
            with pytest.raises(OverflowError):
                op(F(np.array([MIN])))


# ===========================================================================
# 6. 乘法：mul_round / *
# ===========================================================================


class TestMulRound:
    def test_star_is_exact_with_scales_added(self):
        r = F("1.5") * F("0.25")
        assert (int(r.mantissa), r.scale) == (375, 3)
        r = F("1.5") * 3
        assert (int(r.mantissa), r.scale) == (45, 1)
        assert 3 * F("1.5") == r
        assert decimals(F("1.1") * 1.1) == Decimal("1.21")  # float 操作数按最短写法
        r = np.array([1, 2]) * F(["0.1", "0.2"])
        assert isinstance(r, F) and (r.mantissa.tolist(), r.scale) == ([1, 4], 1)

    def test_star_against_decimal(self):
        rng = np.random.default_rng(2)
        a_texts = [f"{x:.3f}" for x in rng.uniform(-1e4, 1e4, 200)]
        b_texts = [f"{x:.4f}" for x in rng.uniform(-10, 10, 200)]
        assert decimals(F(a_texts) * F(b_texts)) == [Decimal(x) * Decimal(y) for x, y in zip(a_texts, b_texts)]

    def test_star_random_overflow_is_exact(self):
        rng = random.Random(3)
        for _ in range(3000):
            a, b = random_mantissa(rng), random_mantissa(rng)
            if fits(a * b):
                got = raw(a, 2) * raw(b, 3)
                assert (int(got.mantissa), got.scale) == (a * b, 5)
            else:
                with pytest.raises(OverflowError):
                    raw(a, 2) * raw(b, 3)

    def test_star_sparse_large_numbers_no_false_positive(self):
        # 上界 10^12 × 10^7 超出快速通道，但每个元素的积都放得下
        c = F(["10000000000.00", "0.01"]) * F(["0.00000001", "0.10000000"])
        assert (c.mantissa.tolist(), c.scale) == ([1000000000000, 10000000], 10)
        assert (F(np.array([2 ** 31])) * F(np.array([2 ** 31]))).dtype == np.int64
        for a, b in ((2 ** 40, 2 ** 40), (2 ** 62, 4), (MAX, 2), (10 ** 18, 10 ** 18)):
            with pytest.raises(OverflowError):
                raw(a, 0) * raw(b, 0)

    def test_default_scale_and_unrounded(self):
        a, b = F("2.50"), F("1.250")
        res = a.mul_round(b)  # half_up 舍入到 max(2, 3) = 3
        assert res.scale == 3 and str(res.to_decimal()) == "3.125"
        exact = a.mul_round(b, rounding=None)  # 精确：scale 2 + 3
        assert exact.scale == 5 and exact == a * b
        assert a.mul_round(b, scale=7, rounding=None).mantissa == 31250000  # 放大 scale 允许
        with pytest.raises(ValueError, match="rounding=None 时不支持缩小 scale"):
            a.mul_round(b, scale=2, rounding=None)
        with pytest.raises(ValueError, match="rounding 只能是"):
            a.mul_round(b, rounding="nearest")

    def test_large_amount_times_precise_rate(self):
        # 100 亿 USD × 8 位汇率：中间积超出 int64，但舍入回 2 位小数后放得下
        res = F("10000000000.00").mul_round(F("0.13821700"), scale=2)
        assert (res.scale, int(res.mantissa)) == (2, 138217000000)
        amounts = F(["1000000000.00", "2000000000.00", "-500000000.00"])
        rates = F(["0.10000000", "0.20000000", "0.50000000"])
        assert amounts.mul_round(rates, scale=2).mantissa.tolist() == [10000000000, 40000000000, -25000000000]

    @pytest.mark.parametrize("mode", MODES)
    def test_rounding_modes_on_ties(self, mode):
        for text in ("2.50", "3.50", "-2.50", "-3.50", "2.49", "-2.51"):
            want = ref_round(Fraction(Decimal(text)), mode)
            assert int(F(text).mul_round(F("1.00"), scale=0, rounding=mode).mantissa) == want

    def test_random_against_fraction(self):
        rng = random.Random(1)
        for _ in range(4000):
            a, b = random_mantissa(rng), random_mantissa(rng)
            sa, sb, target = rng.randint(0, 10), rng.randint(0, 10), rng.randint(0, 22)
            mode = rng.choice(MODES)
            want = ref_round(Fraction(a * b) * Fraction(10) ** (target - sa - sb), mode)
            if fits(want):
                got = raw(a, sa).mul_round(raw(b, sb), scale=target, rounding=mode)
                assert (int(got.mantissa), got.scale) == (want, target), (a, sa, b, sb, target, mode)
            else:
                with pytest.raises(OverflowError):
                    raw(a, sa).mul_round(raw(b, sb), scale=target, rounding=mode)

    def test_vectorized_matches_scalar(self):
        rng = random.Random(2)
        for _ in range(200):
            n = rng.randint(1, 8)
            a = [rng.randint(-(10 ** 12), 10 ** 12) for _ in range(n)]
            b = [rng.randint(-(10 ** 12), 10 ** 12) for _ in range(n)]
            mode = rng.choice(MODES)
            got = raw(a, 4).mul_round(raw(b, 8), scale=2, rounding=mode)
            assert got.mantissa.tolist() == [ref_round(Fraction(x * y, 10 ** 10), mode) for x, y in zip(a, b)]

    def test_shift_edge_cases(self):
        assert F("0.0000000001").mul_round(F("0.0000000001"), scale=0, rounding="up") == F(1)  # shift 20
        assert raw(1, 63).mul_round(raw(1, 0), scale=0, rounding="up") == F(1)  # shift 63
        assert raw(1, 63).mul_round(raw(1, 0), scale=0, rounding="half_up") == F(0)
        assert raw(3, 0).mul_round(raw(4, 0), scale=5).mantissa == 1_200_000  # 放大
        assert raw(0, 0).mul_round(raw(10 ** 18, 0), scale=25).mantissa == 0  # 放大超过 10^18
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert F("0").mul_round(F("5"), scale=3) == F("0.000")

    def test_overflow_true_positive(self):
        with pytest.raises(OverflowError):
            F(str(2 ** 60)).mul_round(F(str(2 ** 10)), scale=0)
        with pytest.raises(OverflowError):
            raw(MAX, 0).mul_round(raw(3, 0), scale=0)
        with pytest.raises(OverflowError):
            raw(10 ** 10, 0).mul_round(raw(10 ** 10, 0), scale=1)
        with pytest.raises(OverflowError):
            raw(10 ** 18, 0).mul_round(raw(10, 0), scale=19)
        with pytest.raises(OverflowError):
            raw(1, 0).mul_round(raw(1, 0), scale=19)

    def test_overflow_only_after_bump(self):
        # MAX·10 + 5 = 3689348814741910323 × 25：截断后恰为 MAX，进位才越界
        assert int(raw(3689348814741910323, 1).mul_round(raw(25, 0), scale=0, rounding="down").mantissa) == MAX
        with pytest.raises(OverflowError):
            raw(3689348814741910323, 1).mul_round(raw(25, 0), scale=0, rounding="up")
        # 2**63·10 + 5 = 1370885 × 67280421310721：负数截断后为 MIN，远离零进位才越界
        assert int(raw(-1370885, 1).mul_round(raw(67280421310721, 0), scale=0, rounding="down").mantissa) == MIN
        with pytest.raises(OverflowError):
            raw(-1370885, 1).mul_round(raw(67280421310721, 0), scale=0, rounding="up")


# ===========================================================================
# 7. 矩阵乘法：matmul / @
# ===========================================================================


class TestMatmul:
    def test_at_is_exact(self):
        r = F([["0.1", "0.2"], ["0.3", "0.4"]]) @ F(["0.5", "0.25"])
        assert (r.mantissa.tolist(), r.scale) == ([100, 250], 3)
        v = F(["1", "2"]) @ F(["3", "4"])  # 1-D @ 1-D → 0 维
        assert (v.ndim, int(v.mantissa)) == (0, 11)
        assert (np.array([[1, 2]]) @ F([[3], [4]])).mantissa.tolist() == [[11]]  # __rmatmul__
        assert ([[1, 2]] @ F([[3], [4]])).mantissa.tolist() == [[11]]

    def test_at_against_decimal(self):
        rng = np.random.default_rng(3)
        a = [[f"{x:.2f}" for x in row] for row in rng.uniform(-1e3, 1e3, (20, 6))]
        w = [f"{x:.4f}" for x in rng.uniform(-1, 1, 6)]
        expected = [sum(Decimal(x) * Decimal(y) for x, y in zip(row, w)) for row in a]
        assert decimals(F(a) @ F(w)) == expected

    def test_matmul_scale_and_rounding(self):
        a = F([["0.10", "0.20"], ["0.30", "0.40"]])
        w = F(["0.500", "0.250"])
        assert (a.matmul(w).scale, a.matmul(w).mantissa.tolist()) == (3, [100, 250])  # max(2, 3)
        assert a.matmul(w, scale=2, rounding="down").mantissa.tolist() == [10, 25]
        exact = a.matmul(w, rounding=None)
        assert exact.scale == 5 and np.all(exact == a @ w)
        assert a.matmul(w, scale=6, rounding=None).mantissa.tolist() == [100000, 250000]
        with pytest.raises(ValueError, match="rounding=None 时不支持缩小 scale"):
            a.matmul(w, scale=2, rounding=None)
        with pytest.raises(ValueError, match="rounding 只能是"):
            a.matmul(w, rounding="nearest")

    @pytest.mark.parametrize("mode", MODES)
    def test_matmul_rounding_modes(self, mode):
        a = F([["1.5", "1"], ["-1.5", "-1"], ["0.25", "0"]])
        r = a.matmul(F(["1", "0"]), scale=0, rounding=mode)
        assert r.mantissa.tolist() == [ref_round(Fraction(3, 2), mode), ref_round(Fraction(-3, 2), mode),
                                       ref_round(Fraction(1, 4), mode)]

    def test_exact_overflow_detection(self):
        with pytest.raises(OverflowError):
            F(np.array([MAX, MAX])) @ F([1, 1])
        assert (F(np.array([MAX, MAX])) @ F([1, -1])).mantissa == 0  # 正负抵消不误报
        hedged = F(np.array([[MAX, 1], [0, 1]])) @ F(np.array([[1, 0], [-1, 1]]))
        assert hedged.mantissa.tolist() == [[MAX - 1, 1], [-1, 1]]

    def test_random_against_python_int(self):
        rng = random.Random(10)
        for _ in range(1500):
            n, k, p = rng.randint(1, 3), rng.randint(1, 5), rng.randint(1, 3)
            a = [[random_mantissa(rng) for _ in range(k)] for _ in range(n)]
            b = [[random_mantissa(rng) for _ in range(p)] for _ in range(k)]
            want = [sum(a[i][t] * b[t][j] for t in range(k)) for i in range(n) for j in range(p)]
            fa, fb = raw(a, 1), raw(b, 2)
            if all(map(fits, want)):
                got = fa @ fb
                assert (got.mantissa.ravel().tolist(), got.scale) == (want, 3)
            else:
                with pytest.raises(OverflowError):
                    fa @ fb

    def test_shape_errors_from_numpy(self):
        with pytest.raises(ValueError):
            F([[1, 2]]) @ F([[1, 2]])
        with pytest.raises(ValueError):
            F("1") @ F("2")


# ===========================================================================
# 8. 除法：divide / / // % divmod
# ===========================================================================


class TestDivide:
    def test_default_scale_is_max_of_operands(self):
        assert (F("1") / 3).scale == 0 and int((F("1") / 3).mantissa) == 0
        assert decimals(F("1.23") / F("2")) == Decimal("0.62")
        assert decimals(F("1.2") / F("2.000")) == Decimal("0.600")
        a = F("1.23456789012")
        assert (a / 2).scale == 11 and (a / 2) == (a / F(2))
        assert F("1.5").divide(7) == F("1.5") / 7

    def test_reflected(self):
        assert 10 / F("2.5") == F("4.0")
        assert (np.array([1, 3]) / F("2.0")).mantissa.tolist() == [5, 15]

    def test_sign_and_ties(self):
        assert decimals(F("1.00000000") / -3) == Decimal("-0.33333333")
        assert decimals(F("-2.00000000") / F("3")) == Decimal("-0.66666667")
        assert int((F("1.00000000") / 200_000_000).mantissa) == 1  # 0.5e-8 进位
        assert int((F("1.00000000") / -200_000_000).mantissa) == -1

    def test_against_decimal(self):
        rng = np.random.default_rng(4)
        a_texts = [f"{x:.2f}" for x in rng.uniform(-1e4, 1e4, 200)]
        b_texts = [f"{x:.3f}" for x in rng.uniform(0.5, 50, 200) * rng.choice([-1, 1], 200)]
        result = F(a_texts) / F(b_texts)
        assert result.scale == 3
        assert result.mantissa.tolist() == [quantized_units(Decimal(x) / Decimal(y), 3) for x, y in
                                            zip(a_texts, b_texts)]

    def test_divide_with_scale_and_rounding(self):
        assert decimals(F("1").divide(3, scale=2)) == Decimal("0.33")
        assert decimals(F("2").divide(3, scale=2, rounding="down")) == Decimal("0.66")
        assert decimals(F("2").divide(3, scale=2, rounding="up")) == Decimal("0.67")
        assert decimals(F("1.234").divide(1, scale=1)) == Decimal("1.2")  # 目标 scale 低于自然 scale
        with pytest.raises(ValueError):
            F("1").divide(3, rounding="nearest")

    @pytest.mark.parametrize("mode", MODES)
    def test_long_division_against_decimal(self, mode):
        rng = np.random.default_rng(11)
        m = rng.integers(-10 ** 17, 10 ** 17, 300) // 10 ** rng.integers(0, 17, 300)
        d = (rng.integers(1, 10 ** 15, 300) // 10 ** rng.integers(0, 15, 300) + 1) * rng.choice([-1, 1], 300)
        for scale in (None, 0, 3, 8, 12):
            target = 4 if scale is None else scale
            expected = [quantized_units(Decimal(int(x)).scaleb(-4) / Decimal(int(y)).scaleb(-2), target, mode)
                        for x, y in zip(m, d)]
            idx = np.flatnonzero([abs(e) < 2 ** 62 for e in expected])
            got = raw(m[idx], 4).divide(raw(d[idx], 2), scale=scale, rounding=mode)
            assert got.mantissa.tolist() == [expected[i] for i in idx]

    def test_random_against_fraction(self):
        rng = random.Random(5)
        for _ in range(3000):
            a, b = random_mantissa(rng), random_mantissa(rng) or 7
            sa, sb, target = rng.randint(0, 8), rng.randint(0, 8), rng.randint(0, 12)
            mode = rng.choice(MODES)
            want = ref_round(Fraction(a, 10 ** sa) / Fraction(b, 10 ** sb) * 10 ** target, mode)
            try:
                got = raw(a, sa).divide(raw(b, sb), scale=target, rounding=mode)
            except OverflowError:
                assert not fits(want), (a, sa, b, sb, target, mode)
                continue
            assert (int(got.mantissa), got.scale) == (want, target), (a, sa, b, sb, target, mode)

    def test_wide_intermediate(self):
        # 被除数乘 10^k 会越界，但结果放得下：长除法精确算出
        assert int(F(np.array([10 ** 15])).divide(10 ** 9, scale=8).mantissa[0]) == 10 ** 14
        assert int(F("123456.789012").divide(7, scale=14).mantissa) == quantized_units(Decimal("123456.789012") / 7, 14)
        with pytest.raises(OverflowError):
            F(np.array([10 ** 12])) / F("0.0001")  # 结果本身越界

    @pytest.mark.parametrize("mode", MODES)
    def test_min_by_minus_one_to_lower_scale(self, mode):
        # 结果还要再除以 10^k，放得下；np.divmod(MIN, -1) 本身算不了，按对称性单独处理
        nums = raw([MIN, MIN, 7, -7], 16)
        dens = raw([-1, 3, -1, -1], 0)
        got = nums.divide(dens, scale=8, rounding=mode)
        want = [ref_round(Fraction(n, 10 ** 16) / d * 10 ** 8, mode) for n, d in
                ((MIN, -1), (MIN, 3), (7, -1), (-7, -1))]
        assert got.mantissa.tolist() == want
        with pytest.raises(OverflowError):
            raw(MIN, 0).divide(raw(-1, 0), scale=0)  # 目标 scale 等于自然 scale：结果本身越界

    def test_long_division_overflow_only_after_bump(self):
        # 6456360425798343065 × 10 = 7·MAX + 1：截断后恰为 MAX，余 1/7，远离零进位才越界
        assert int(raw(6456360425798343065, 0).divide(raw(7, 0), scale=1, rounding="down").mantissa) == MAX
        with pytest.raises(OverflowError, match="舍入后超出"):
            raw(6456360425798343065, 0).divide(raw(7, 0), scale=1, rounding="up")

    def test_errors(self):
        for divisor in (0, F("0"), F(["1", "0"])):
            with pytest.raises(ZeroDivisionError):
                F(["1", "2"]) / divisor
        with pytest.raises(ZeroDivisionError):
            F("1").divide(0, scale=2)
        with pytest.raises(ZeroDivisionError):
            F("1.00").divide(0, scale=0)  # shift ≤ 0 分支也先报除以 0
        with pytest.raises(OverflowError):
            F(MAX).divide(F("0.1"), scale=0)


class TestFloorDivMod:
    def test_python_semantics(self):
        a, b = F(["10.01", "-10.01"]), F("3")
        q, r = divmod(a, b)
        assert q.tolist() == [3, -4] and decimals(r) == [Decimal("1.01"), Decimal("1.99")]
        assert np.all(b * q + r == a)
        assert (a // b).tolist() == [3, -4]
        assert decimals(a % b) == [Decimal("1.01"), Decimal("1.99")]

    def test_scalar_quotient_is_int(self):
        q, r = divmod(F("10.01"), F("3"))
        assert type(q) is int and q == 3 and decimals(r) == Decimal("1.01")
        assert type(F("7") // 2) is int

    def test_reflected(self):
        x = F("2.5")
        assert 10 // x == 4 and 10 % x == F("0") and divmod(10, x) == (4, F("0"))
        assert (np.array([10, 10]) // F(["2", "4"])).tolist() == [5, 2]
        assert (np.array([7]) % F("2")).mantissa.tolist() == [1]

    def test_random_against_fraction(self):
        rng = random.Random(8)
        for _ in range(2000):
            a, b = rng.randint(-(10 ** 15), 10 ** 15), rng.randint(-(10 ** 6), 10 ** 6) or 3
            x, y = raw(a, 3), raw(b, 1)
            q, r = divmod(Fraction(a, 1000), Fraction(b, 10))
            assert x // y == q and value(x % y) == r
            assert divmod(x, y)[0] == q and value(divmod(x, y)[1]) == r

    def test_errors(self):
        with pytest.raises(OverflowError):
            F(np.array([MIN])) // -1
        with pytest.raises(OverflowError):
            divmod(F(np.array([MIN])), -1)
        for op in (lambda a, b: a // b, lambda a, b: a % b, divmod):
            with pytest.raises(ZeroDivisionError):
                op(F([1, 2]), F([1, 0]))
            with pytest.raises(ZeroDivisionError):
                op(F("1"), 0)


class TestReflectedOperators:
    @pytest.mark.parametrize("name", ["__radd__", "__rsub__", "__rmul__", "__rmatmul__", "__rtruediv__",
                                      "__rfloordiv__", "__rmod__", "__rdivmod__"])
    @pytest.mark.parametrize("bad", ["1", True, None])
    def test_return_not_implemented_for_non_numeric(self, name, bad):
        assert getattr(F("2"), name)(bad) is NotImplemented

    @pytest.mark.parametrize("name", ["__add__", "__sub__", "__mul__", "__matmul__", "__truediv__",
                                      "__floordiv__", "__mod__", "__divmod__", "__lt__", "__le__", "__gt__",
                                      "__ge__", "__eq__", "__ne__"])
    def test_forward_return_not_implemented_for_non_numeric(self, name):
        assert getattr(F("2"), name)("1") is NotImplemented

    def test_numeric_left_operands(self):
        x = F("2.5")
        assert 10 - x == F("7.5")
        assert Decimal("1.5") + x == F("4")
        assert Fraction(1, 2) * x == F("1.25")


# ===========================================================================
# 9. 规约：sum / mean
# ===========================================================================


class TestSum:
    def test_basic(self):
        a = F([["1.5", "2"], ["3", "4.25"]])
        assert decimals(a.sum()) == Decimal("10.75")
        assert decimals(a.sum(axis=0)) == [Decimal("4.5"), Decimal("6.25")]
        assert a.sum(axis=1, keepdims=True).shape == (2, 1)
        assert decimals(np.sum(a)) == Decimal("10.75")
        assert np.all(np.sum(a, axis=0) == a.sum(axis=0))

    def test_axes_and_dtype(self):
        base = np.arange(24).reshape(2, 3, 4)
        a = raw(base, 1)
        for axis in (None, 0, -1, (0, 2)):
            for keepdims in (False, True):
                r = a.sum(axis=axis, keepdims=keepdims)
                assert r.dtype == np.int64 and r.scale == 1
                assert np.array_equal(r.mantissa, base.sum(axis=axis, keepdims=keepdims))
        assert int(F(np.array([], dtype=np.int64)).sum().mantissa) == 0

    def test_exact_overflow(self):
        with pytest.raises(OverflowError):
            F(np.array([MAX, 1])).sum()
        with pytest.raises(OverflowError):
            F(np.array([MIN, -1])).sum()
        assert F(np.array([MAX, 1, -1])).sum().mantissa == MAX  # 正负抵消不误报
        assert F(np.array([MIN, -1, 1])).sum().mantissa == MIN
        assert F(np.array([[MAX, -MAX], [1, 1]])).sum(axis=1).mantissa.tolist() == [0, 2]
        assert F(np.array([MAX // 2, MAX // 2, -(MAX // 2), -(MAX // 2)])).sum().mantissa == 0

    def test_counts_per_cell(self):
        a = F(np.array([[MAX // 2, MAX // 2], [1, 1]]))
        assert a.sum(axis=0).mantissa.tolist() == [MAX // 2 + 1, MAX // 2 + 1]
        with pytest.raises(OverflowError):
            a.sum()

    def test_random_against_python_int(self):
        rng = random.Random(9)
        for _ in range(1500):
            rows = [[random_mantissa(rng) for _ in range(rng.randint(1, 5))]] * rng.randint(1, 4)
            fa = raw(rows, 2)
            for axis in (None, 0, 1):
                want = ([sum(map(sum, rows))] if axis is None else
                        [sum(col) for col in zip(*rows)] if axis == 0 else [sum(r) for r in rows])
                if all(map(fits, want)):
                    assert np.ravel(fa.sum(axis=axis).mantissa).tolist() == want
                else:
                    with pytest.raises(OverflowError):
                        fa.sum(axis=axis)

    def test_rejects_out_and_dtype(self):
        a = F(["1", "2"])
        with pytest.raises(TypeError):
            np.sum(a, out=np.zeros(()))
        with pytest.raises(TypeError):
            np.sum(a, dtype=np.float64)
        with pytest.raises(TypeError):
            a.sum(dtype=np.int64)


class TestMean:
    def test_basic(self):
        assert int(F(["1.0", "2.0", "4.0"]).mean().mantissa) == 23  # 2.333 → 2.3
        assert F([["1", "2"], ["3", "5"]]).mean(axis=0).mantissa.tolist() == [2, 4]  # 3.5 → 4
        assert int(F(["-1", "-2"]).mean().mantissa) == -2  # -1.5 → -2（远离零）
        assert decimals(np.mean(F(["1.5", "2.5"]))) == Decimal("2.0")
        assert np.all(np.mean(F([[1, 2], [3, 5]]), axis=0) == F([2, 4]))
        assert F(["1", "2"]).mean().dtype == np.int64
        assert F("2.5").mean() == F("2.5")  # 0 维

    def test_scale_and_rounding(self):
        assert F(["1", "2"]).mean(scale=2) == F("1.50")
        assert F(["1.23", "1.25"]).mean(scale=1).mantissa == 12
        assert F(["1.23", "1.27"]).mean(scale=1).mantissa == 13
        assert F(["1.23", "1.27"]).mean(scale=0).mantissa == 1
        assert F(["1.0", "2.0"]).mean(scale=0, rounding="floor").mantissa == 1
        assert F(["1.0", "2.0"]).mean(scale=0, rounding="ceil").mantissa == 2
        assert F(["2.0", "3.0"]).mean(scale=0, rounding="half_even").mantissa == 2
        assert F(np.arange(10)).mean(scale=18) == F("4.5")  # 计数 × 10^shift 超出 int64：分段长除法

    def test_axes(self):
        base = np.arange(24).reshape(2, 3, 4) * 7 - 50
        a = raw(base, 0)
        for axis in (None, 0, 1, -1, (0, 2), (-1, 0), ()):
            for keepdims in (False, True):
                r = a.mean(axis=axis, keepdims=keepdims, scale=6, rounding="half_even")
                ref = base.sum(axis=axis, keepdims=keepdims)
                count = base.size // max(np.size(ref), 1) if axis != () else 1
                want = np.vectorize(lambda s: ref_round(Fraction(int(s), count) * 10 ** 6, "half_even"),
                                    otypes=[object])(ref)
                assert r.shape == np.shape(ref) and np.ravel(r.mantissa).tolist() == np.ravel(want).tolist()

    def test_sum_overflow_does_not_matter(self):
        # 商余分别累加：和溢出但均值放得下时不报错
        assert F(np.array([MAX, MAX - 2])).mean().mantissa == MAX - 1
        assert F([MAX, MAX]).mean().mantissa == MAX
        assert raw([[MIN, MIN, MIN]], 0).mean(axis=1).mantissa.tolist() == [MIN]

    def test_overflow_when_raising_scale(self):
        with pytest.raises(OverflowError):
            F(np.array([MAX, MAX - 2])).mean(scale=1)
        with pytest.raises(OverflowError):
            F([1]).mean(scale=19)
        # 放大后整数部分加小数部分越界
        with pytest.raises(OverflowError):
            raw([[922337203685477581] * 9 + [922337203685477580]], 0).mean(axis=1, scale=1)
        # 放大后恰为 MAX，进位才越界
        data = [[922337203685477581] * 22 + [922337203685477580] * 8]
        assert raw(data, 0).mean(axis=1, scale=1, rounding="down").mantissa.tolist() == [MAX]
        with pytest.raises(OverflowError):
            raw(data, 0).mean(axis=1, scale=1, rounding="up")

    def test_random_against_fraction(self):
        rng = random.Random(3)
        for _ in range(3000):
            shape = rng.choice([(3,), (2, 3), (4, 1), (1,), (5,)])
            data = np.array([random_mantissa(rng) for _ in range(int(np.prod(shape)))], dtype=np.int64).reshape(shape)
            s, t, mode = rng.randint(0, 6), rng.randint(0, 12), rng.choice(MODES)
            axis = rng.choice([None, 0, -1] + ([1, (0, 1)] if len(shape) == 2 else []))
            sums = np.ravel(np.asarray(data.astype(object).sum(axis=axis), dtype=object))
            count = data.size // len(sums)
            want = [ref_round(Fraction(int(v), count) * Fraction(10) ** (t - s), mode) for v in sums]
            fa = raw(data, s)
            if all(map(fits, want)):
                got = fa.mean(axis=axis, scale=t, rounding=mode)
                assert np.ravel(got.mantissa).tolist() == want, (data.tolist(), s, t, mode, axis)
            else:
                with pytest.raises(OverflowError):
                    fa.mean(axis=axis, scale=t, rounding=mode)

    def test_empty_inputs(self):
        empty_rows = F(np.zeros((0, 3), dtype=np.int64))
        assert empty_rows.mean(axis=1).shape == (0,)  # 每格 3 个元素，只是没有格子
        with pytest.raises(ZeroDivisionError):
            empty_rows.mean(axis=0)
        with pytest.raises(ZeroDivisionError):
            empty_rows.mean()

    def test_argument_errors(self):
        a = F([[1, 2], [3, 4]])
        with pytest.raises(ZeroDivisionError):
            F(np.array([], dtype=np.int64)).mean()

        # 对空数组沿非空维度求均值（count > 0），行为与 NumPy 一致，返回对应形状的空 FixedArray
        empty_2d = F(np.zeros((0, 3), dtype=np.int64), scale=2)
        res = empty_2d.mean(axis=1)
        assert res.shape == (0,)
        assert res.scale == 2
        res_kd = empty_2d.mean(axis=1, keepdims=True)
        assert res_kd.shape == (0, 1)

        # 真正对 0 个元素求均值时（count == 0），抛出 ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            empty_2d.mean(axis=0)
        with pytest.raises(ZeroDivisionError):
            empty_2d.mean(axis=None)

        with pytest.raises(AXIS_ERROR):
            a.mean(axis=2)
        with pytest.raises(AXIS_ERROR):
            a.mean(axis=(0, -3))
        with pytest.raises(ValueError, match="duplicate"):
            a.mean(axis=(0, -2))
        with pytest.raises(TypeError):
            a.mean(axis=(0, 1.0))
        with pytest.raises(TypeError):
            a.mean(axis=[0])
        with pytest.raises(TypeError):
            np.mean(a, out=np.zeros(()))
        with pytest.raises(TypeError):
            a.mean(dtype=np.float64)
        with pytest.raises(ValueError, match="rounding 只能是"):
            a.mean(rounding="nearest")


# ===========================================================================
# 10. 形状
# ===========================================================================


class TestShape:
    def test_indexing_and_len(self):
        a = F(["1.5", "2.5", "3.5"])
        assert int(a[1].mantissa) == 25 and a[1].ndim == 0 and a[1].scale == 1
        assert a[1:].mantissa.tolist() == [25, 35]
        assert a[np.array([True, False, True])].mantissa.tolist() == [15, 35]
        assert len(a) == 3 and [int(x.mantissa) for x in a] == [15, 25, 35]
        with pytest.raises(TypeError):
            len(F("1"))
        with pytest.raises(IndexError):
            a[3]

    def test_python_protocols(self):
        assert bool(F([5])) is True and bool(F([[0]])) is False  # 恰好 1 个元素：按值判断
        for ambiguous in (F([1, 2]), F(np.array([], dtype=np.int64))):
            with pytest.raises(ValueError, match="ambiguous"):
                bool(ambiguous)
        rows = list(F([["1.5", "2"], ["3", "4"]]))
        assert len(rows) == 2 and np.all(rows[1] == F(["3", "4"])) and rows[1].scale == 1
        a = F("1.5")
        assert +a is a

    def test_reshape_transpose(self):
        a = F(["1", "2", "3", "4"]).reshape(2, 2)
        assert a.scale == 0 and a.T.mantissa.tolist() == [[1, 3], [2, 4]]
        assert a.transpose(1, 0).mantissa.tolist() == [[1, 3], [2, 4]]
        assert F(["1.5"] * 6).reshape((3, 2)).shape == (3, 2)
        assert not a.T.mantissa.flags.writeable
        with pytest.raises(ValueError):
            a.reshape(3)


# ===========================================================================
# 11. 精度调整：rescale / round_to_step / is_on_step
# ===========================================================================


class TestRescale:
    @pytest.mark.parametrize("mode", MODES)
    def test_against_decimal(self, mode):
        texts = ["-1.25", "-1.24", "-1.15", "-0.05", "0", "0.05", "1.15", "1.24", "1.25", "1.26", "-1.26"]
        result = F(texts).rescale(1, rounding=mode)
        assert result.scale == 1
        assert result.mantissa.tolist() == [quantized_units(Decimal(t), 1, mode) for t in texts]

    def test_up_and_same(self):
        a = F("1.5")
        assert int(a.rescale(3).mantissa) == 1500
        assert a.rescale(1) is a

    def test_random_against_fraction(self):
        rng = random.Random(6)
        for _ in range(3000):
            m, s = random_mantissa(rng), rng.randint(0, 30)
            new = rng.randint(0, s + 18)
            mode = rng.choice(MODES)
            want = ref_round(Fraction(m, 10 ** s) * Fraction(10) ** new, mode)
            if fits(want):
                got = raw(m, s).rescale(new, mode)
                assert (int(got.mantissa), got.scale) == (want, new)
            else:
                with pytest.raises(OverflowError):
                    raw(m, s).rescale(new, mode)

    @pytest.mark.parametrize("m, mode, want", [
        (5 * 10 ** 18, "half_up", 1), (5 * 10 ** 18, "half_even", 0), (5 * 10 ** 18 + 1, "half_even", 1),
        (5 * 10 ** 18 - 1, "half_up", 0), (-5 * 10 ** 18, "half_up", -1), (-5 * 10 ** 18, "half_even", 0),
        (-5 * 10 ** 18 + 1, "half_up", 0), (MIN, "half_up", -1), (MAX, "down", 0), (MIN, "ceil", 0), (1, "up", 1),
    ])
    def test_drop_19_digits_boundaries(self, m, mode, want):
        assert int(raw(m, 19).rescale(0, mode).mantissa) == want

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("m", [MAX, -MAX, MIN, 1, -1, 0])
    def test_drop_more_than_19_digits(self, m, mode):
        # |m| < 10^19 ≤ 10^k：结果只能是 -1、0、1
        assert int(raw(m, 25).rescale(0, mode).mantissa) == ref_round(Fraction(m, 10 ** 25), mode)

    def test_errors(self):
        with pytest.raises(ValueError):
            F("1.5").rescale(-1)
        for new_scale in (1, 2, 3):  # 缩小、不变、放大都要校验 rounding
            with pytest.raises(ValueError, match="rounding 只能是"):
                F("1.55").rescale(new_scale, rounding="nearest")
        with pytest.raises(OverflowError):
            F([10 ** 17]).rescale(3)


class TestRoundToStep:
    def test_scalar_steps(self):
        assert F("10.23").round_to_step(F("0.05")) == F("10.25")
        assert F("10.22").round_to_step(F("0.05")) == F("10.20")
        assert F("10.225").round_to_step(F("0.05")) == F("10.250")
        assert F("10.23").round_to_step(0.05) == F("10.25")
        assert F("10.23").round_to_step(Decimal("0.05")) == F("10.25")
        assert F("10.23").round_to_step(Fraction(1, 20)) == F("10.25")
        assert F(345).round_to_step(200) == F(400)
        assert F(245).round_to_step(200) == F(200)
        assert F("10.123").round_to_step(F("0.025")) == F("10.125")

    @pytest.mark.parametrize("mode", MODES)
    def test_rounding_modes(self, mode):
        for x in (100, -100, 300, -300, 150, 99):
            assert F(x).round_to_step(200, mode) == F(ref_round(Fraction(x, 200), mode) * 200)

    def test_scale_preservation(self):
        r = F(10).round_to_step(F("0.05"))
        assert r == F("10.00") and r.scale == 2
        r2 = F("345.00").round_to_step(200)
        assert r2 == F("400.00") and r2.scale == 2

    def test_array_and_broadcasting(self):
        arr = F(["10.21", "10.23", "10.24"])
        assert np.all(arr.round_to_step(F("0.05")) == F(["10.20", "10.25", "10.25"]))
        assert np.all(arr.round_to_step(F(["0.05", "0.10", "0.25"])) == F(["10.20", "10.20", "10.25"]))

    def test_random_against_fraction(self):
        rng = random.Random(7)
        for _ in range(2000):
            m, step_m, step_s = rng.randint(-(10 ** 12), 10 ** 12), rng.randint(1, 500), rng.randint(0, 4)
            mode = rng.choice(MODES)
            step = Fraction(step_m, 10 ** step_s)
            want = ref_round(Fraction(m, 10 ** 4) / step, mode) * step
            got = raw(m, 4).round_to_step(raw(step_m, step_s), rounding=mode)
            assert value(got) == want and got.is_on_step(raw(step_m, step_s)) is True

    def test_overflow(self):
        with pytest.raises(OverflowError):
            raw(MAX, 0).round_to_step(10, rounding="up")

    def test_is_on_step(self):
        assert F("10.25").is_on_step(F("0.05")) is True
        assert F("10.23").is_on_step(F("0.05")) is False
        assert F(400).is_on_step(200) is True and F(345).is_on_step(200) is False
        res = F(["10.20", "10.23", "10.25"]).is_on_step(F("0.05"))
        assert np.array_equal(res, [True, False, True])
        assert np.array_equal(F("0.10").is_on_step(F(["0.05", "0.03"])), [True, False])

    def test_invalid_step(self):
        for call in (lambda s: F(10).round_to_step(s), lambda s: F(10).is_on_step(s)):
            for bad in (0, -5, F(["1", "0"])):
                with pytest.raises(ValueError, match="步长 step 必须为正数"):
                    call(bad)
        with pytest.raises(ValueError, match="rounding 只能是"):
            F(10).round_to_step(5, rounding="invalid")


# ===========================================================================
# 12. 比较
# ===========================================================================


class TestCompare:
    def test_align_and_broadcast(self):
        assert F("1.10") == F("1.1")
        assert (F(["1", "2"]) < F(["2", "1"])).tolist() == [True, False]
        assert (F(["1", "2"]) <= 1).tolist() == [True, False]
        assert (F(["1", "2"]) >= F("2")).tolist() == [False, True]
        assert F("1") != F("2")
        assert F("2.0") > 1
        assert F("0.3") == 0.3 and F("0.25") == Fraction(1, 4) and F("1.5") == Decimal("1.50")
        assert (F([1, 2]) == [1, 3]).tolist() == [True, False]
        assert (F([1, 2]) != np.array([1, 3])).tolist() == [False, True]

    def test_scalar_results_are_numpy_bool(self):
        assert isinstance(F("1") == F("1"), np.bool_)
        assert isinstance(F("1") < 2, np.bool_)

    def test_random_against_fraction(self):
        rng = random.Random(11)
        for _ in range(2000):
            a, b = random_mantissa(rng), random_mantissa(rng)
            sa, sb = rng.randint(0, 2), rng.randint(0, 2)
            x, y = Fraction(a, 10 ** sa), Fraction(b, 10 ** sb)
            fx, fy = raw(a, sa), raw(b, sb)
            try:
                results = (fx < fy, fx <= fy, fx == fy, fx != fy, fx > fy, fx >= fy)
            except OverflowError:
                continue  # 对齐 scale 时放大溢出
            assert tuple(map(bool, results)) == (x < y, x <= y, x == y, x != y, x > y, x >= y)

    def test_non_numeric(self):
        assert (F("1") == "1") is False and (F("1") != "1") is True
        assert (F("1.5") == np.array(["1.5"], dtype=object)) is False
        with pytest.raises(TypeError):
            F("1") < "2"

    def test_shape_mismatch_is_a_real_error(self):
        with pytest.raises(ValueError):
            F(["1", "2"]) == F(["1", "2", "3"])
        with pytest.raises(ValueError):
            F(["1", "2"]) < F(["1", "2", "3"])

    def test_alignment_overflow_propagates_for_ordering(self):
        with pytest.raises(OverflowError):
            raw(MAX, 0) < F("0.1")


# ===========================================================================
# 13. NumPy 协议
# ===========================================================================


class TestNumpyProtocol:
    def test_ufuncs_rejected(self):
        a = F(["0.1"])
        for op in (lambda: np.add(a, 1), lambda: np.multiply(np.array([1]), a), lambda: np.sin(a),
                   lambda: np.array([1 + 2j]) + a):
            with pytest.raises(TypeError):
                op()

    def test_no_implicit_array(self):
        a = F(["0.1"])
        for op in (lambda: np.asarray(a), lambda: np.concatenate([a, a]), lambda: np.array(a, dtype=float),
                   lambda: F([F("1"), F("2")]), lambda: a.__array__()):
            with pytest.raises(TypeError):
                op()

    def test_numpy_reductions_dispatch(self):
        a = F([["1", "2"], ["3", "4"]])
        assert np.all(np.sum(a, axis=1, keepdims=True) == a.sum(axis=1, keepdims=True))
        assert np.all(np.mean(a, axis=0) == a.mean(axis=0))


# ===========================================================================
# 14. 导出
# ===========================================================================


class TestToFloat:
    def test_scalar_and_array(self):
        r = F(["0.1", "0.25"]).to_float()
        assert r.dtype == np.float64 and r.tolist() == [0.1, 0.25]
        s = F("0.1").to_float()
        assert type(s) is float and s == 0.1
        assert F([["1.5"]]).to_float().shape == (1, 1)

    def test_random_correctly_rounded(self):
        rng = random.Random(31)
        for _ in range(2000):
            m, s = random_mantissa(rng), rng.randint(0, 25)
            assert raw(m, s).to_float() == m / 10 ** s, (m, s)
        ms = [random_mantissa(rng) for _ in range(500)] + [MIN, MAX, 2 ** 53 + 1]
        assert raw(ms, 15).to_float().tolist() == [m / 10 ** 15 for m in ms]  # 含超过 2^53 的尾数：逐个精确计算


class TestToDecimal:
    def test_scalar(self):
        assert F("10.50").to_decimal() == Decimal("10.50")
        assert str(F("10.50").to_decimal()) == "10.50"  # 保留末尾 0
        assert F(5).to_decimal() == Decimal("5") and F("-0.005").to_decimal() == Decimal("-0.005")

    def test_array(self):
        r = F(["10.50", "20.00"]).to_decimal()
        assert r.dtype == object and r.tolist() == [Decimal("10.50"), Decimal("20.00")]
        r2d = F([["1.5", "2.0"], ["3.0", "4.0"]]).to_decimal()
        assert r2d.shape == (2, 2) and r2d[0, 0] == Decimal("1.5")

    def test_random_round_trip(self):
        rng = random.Random(13)
        for _ in range(1000):
            m, s = random_mantissa(rng), rng.randint(0, 18)
            if m == MIN:
                continue  # 见 TestKnownDefects.test_construct_int64_min_from_text
            d = raw(m, s).to_decimal()
            assert d == Fraction(m, 10 ** s) and F(d) == raw(m, s)


class TestToPandas:
    def test_series(self):
        pd = pytest.importorskip("pandas")
        s = F(["10.50", "20.00"]).to_pandas(name="price", as_float=True)
        assert isinstance(s, pd.Series) and s.name == "price" and s.dtype == np.float64
        assert s.tolist() == [10.5, 20.0]
        d = F(["10.50", "20.00"]).to_pandas(name="price", as_float=False)
        assert d.dtype == object and d.tolist() == [Decimal("10.50"), Decimal("20.00")]
        assert F(["1", "2"]).to_pandas(index=["a", "b"]).to_dict() == {"a": 1.0, "b": 2.0}

    def test_scalar(self):
        pytest.importorskip("pandas")
        s = F("10.50").to_pandas(name="p")
        assert len(s) == 1 and s.iloc[0] == 10.5 and s.name == "p"

    def test_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = F([["1.5", "2.0"], ["3.0", "4.0"]]).to_pandas(as_float=True)
        assert isinstance(df, pd.DataFrame) and df.shape == (2, 2) and df.iloc[0, 1] == 2.0
        df_dec = F([["1.5", "2.0"], ["3.0", "4.0"]]).to_pandas(as_float=False, index=["x", "y"])
        assert df_dec.loc["x", 1] == Decimal("2.0")

    def test_missing_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ImportError, match="需要安装 pandas"):
            F(["1"]).to_pandas()


class TestToPolars:
    def test_float(self):
        pl = pytest.importorskip("polars")
        s = F(["10.50", "20.00"]).to_polars(name="p")
        assert isinstance(s, pl.Series) and s.name == "p" and s.to_list() == [10.5, 20.0]
        assert F("10.50").to_polars().to_list() == [10.5]

    def test_decimal_dtype(self):
        pytest.importorskip("polars")
        s = F(["10.50", "20.00"]).to_polars(as_float=False)
        assert str(s.dtype).startswith("Decimal")
        assert F("10.50").to_polars(as_float=False).len() == 1

    def test_decimal_scale_limit(self):
        pytest.importorskip("polars")
        a = F("0." + "0" * 39 + "1")
        assert a.to_polars().to_list() == [1e-40]  # float 导出不受限制
        with pytest.raises(ValueError, match="最多支持 38 位小数"):
            a.to_polars(as_float=False)
        assert F("0." + "0" * 37 + "1").to_polars(as_float=False).to_list() == [Decimal("1e-38")]

    def test_rejects_2d(self):
        pytest.importorskip("polars")
        with pytest.raises(ValueError, match="仅支持 1 维或标量"):
            F([["1"]]).to_polars()

    def test_missing_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "polars", None)
        with pytest.raises(ImportError, match="需要安装 polars"):
            F(["1"]).to_polars()


# ===========================================================================
# 15. 展示
# ===========================================================================


class TestDisplay:
    def test_scalar(self):
        assert str(F("0.10")) == "0.10"
        assert str(F("-0.005")) == "-0.005"
        assert str(F("12")) == "12"
        assert repr(F("-0.005")) == "FixedArray(-0.005, scale=3)"
        assert str(raw(MIN, 2)) == "-92233720368547758.08"

    def test_array(self):
        assert str(F([1, 2])) == "[1, 2]"
        assert str(F(["0.1", "0.25"])) == "[0.10, 0.25]"
        assert repr(F(["0.1", "0.25"])) == "FixedArray([0.10, 0.25], scale=2)"
        assert str(F(np.array([2 ** 60 + 1]))) == "[1152921504606846977]"
        assert str(F([["1.5", "2"], ["3", "4"]])) == "[[1.5, 2.0],\n [3.0, 4.0]]"
        assert str(F(np.array([], dtype=np.int64))) == "[]"

    @pytest.mark.parametrize("m, scale, text", [(0, 0, "0"), (5, 3, "0.005"), (-5, 3, "-0.005"), (1234, 2, "12.34"),
                                                (-100, 2, "-1.00"), (MAX, 18, "9.223372036854775807")])
    def test_format_scalar(self, m, scale, text):
        assert F._format_scalar(m, scale) == text


# ===========================================================================
# 16. 缺陷回归测试（已修复）
# ===========================================================================


class TestRegressions:
    """曾经确认并已修复的缺陷，防止回归。"""

    # ---- 构造 ----
    def test_float_array_explicit_scale_does_not_need_full_precision(self):
        assert F(np.array([100000.0, 1 / 3]), scale=2).mantissa.tolist() == [10000000, 33]

    def test_money_from_float_array(self):
        assert Money(np.array([100000.0, 1 / 3]), "USD").units.tolist() == [10000000, 33]

    def test_tiny_float_to_low_scale(self):
        assert F(1e-20, scale=0) == F(0)

    @pytest.mark.parametrize("make", [lambda: F(0, scale=19), lambda: F([0, 0], scale=25), lambda: F("0", scale=19),
                                      lambda: F(0.0, scale=19), lambda: F("0").rescale(30)], ids=[
        "int", "int-array", "str", "float", "rescale"])
    def test_zero_with_scale_above_18(self, make):
        r = make()
        assert r.scale >= 19 and not r.mantissa.any()

    def test_construct_int64_min_from_text(self):
        assert int(F(str(MIN)).mantissa) == MIN
        assert int(F([str(MIN)]).mantissa[0]) == MIN
        assert int(F(Decimal(MIN)).mantissa) == MIN

    # ---- 精度调整 / 除法 ----
    @pytest.mark.parametrize("mode, want", [("half_up", 0), ("up", 1), ("floor", 0)])
    def test_rescale_drop_more_than_18_digits(self, mode, want):
        assert int(raw(1, 20).rescale(0, mode).mantissa) == want

    def test_matmul_round_drop_more_than_18_digits(self):
        assert raw([[1]], 20).matmul(F([[1]]), scale=0) == F([[0]])

    def test_divide_to_lower_scale_no_false_overflow(self):
        assert raw(10 ** 18, 18).divide(raw(10 ** 9, 0), scale=0) == F(0)

    def test_divide_by_huge_divisor(self):
        assert raw(1, 0).divide(raw(2 ** 62, 0), scale=1) == F(0)

    @pytest.mark.parametrize("m, want", [(MIN, MIN), (MAX - 7, MAX - 7)])
    def test_long_division_result_near_int64_bound(self, m, want):
        assert int(raw(m, 0).divide(raw(10, 0), scale=1).mantissa) == want

    # ---- 运行时警告 ----
    def test_product_equal_to_int64_min_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert int((raw(MIN, 0) * raw(1, 0)).mantissa) == MIN
            assert int((raw(-(2 ** 62), 0) * raw(2, 0)).mantissa) == MIN
            got = raw(-1370885, 1).mul_round(raw(67280421310721, 0), scale=0, rounding="down")  # 舍入路径
            assert int(got.mantissa) == MIN

    def test_mean_scalar_result_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert int(F(np.array([MIN] * 3)).mean().mantissa) == MIN

    # ---- 兼容性 ----
    def test_mean_works_without_numpy_exceptions_module(self, monkeypatch):
        class OldNumpy(types.ModuleType):
            def __getattr__(self, name):
                if name == "exceptions":
                    raise AttributeError("module 'numpy' has no attribute 'exceptions'")
                return getattr(np, name)

        monkeypatch.setattr(fixed_array_module, "np", OldNumpy("numpy"))
        assert F([1, 2]).mean(scale=1) == F("1.5")

    # ---- 导出 ----
    def test_to_polars_decimal_values(self):
        pytest.importorskip("polars")
        assert F(["10.50", "20.00"]).to_polars(as_float=False).to_list() == [Decimal("10.50"), Decimal("20.00")]
        assert F("10.50").to_polars(as_float=False).to_list() == [Decimal("10.50")]

    def test_to_float_is_correctly_rounded(self):
        m, s = 2310746717979154049, 15
        assert raw(m, s).to_float() == m / 10 ** s

    # ---- Python 协议 ----
    def test_truthiness_follows_numpy(self):
        assert bool(F("0")) is False
        assert bool(F("1.5")) is True
        with pytest.raises(ValueError):
            bool(F([0, 0]))

    def test_iterating_scalar_raises(self):
        with pytest.raises(TypeError):
            list(F("5"))

    def test_unary_plus(self):
        assert +F("1.5") == F("1.5")

    # ---- 第二轮：float 数组最短写法与 MIN / -1 ----
    def test_float_array_matches_scalar_shortest_repr(self):
        xs = [-632982710.3962541, 0.125, 808251333.22099]
        a = F(np.array(xs))
        assert a.mantissa.tolist() == [int(F(x).rescale(a.scale).mantissa) for x in xs]

    def test_float_array_explicit_scale_rounds_shortest_repr(self):
        xs = [-632982710.3962541, 0.125, 808251333.22099]
        assert F(np.array(xs), scale=8).mantissa.tolist() == [-63298271039625410, 12500000, 80825133322099000]

    def test_float_array_random_large_magnitudes_match_scalar(self):
        rng = random.Random(41)
        for _ in range(3000):
            xs = [round(rng.uniform(-1e9, 1e9), rng.randint(0, 7)) for _ in range(rng.randint(2, 6))]
            try:
                a = F(np.array(xs))
            except OverflowError:
                continue
            assert a.mantissa.tolist() == [int(F(x).rescale(a.scale).mantissa) for x in xs], xs

    def test_divide_min_by_minus_one_to_lower_scale(self):
        got = raw(MIN, 16).divide(F(-1), scale=8, rounding="up")
        assert int(got.mantissa) == 92233720369
