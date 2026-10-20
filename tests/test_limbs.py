"""_limbs 测试：Base 2^21 分块整数内核与 Python 任意精度整数精确对拍。

分块表示：值 = Σ limbs[i]·2^(21·i)，低位在前。规整后除最高块外每块落在 [0, 2^21)，
最高块带符号且 |·| < 2^42，至少 4 块。各函数的结果都按这个定义还原成 Python int 后比对。
"""

import random
from fractions import Fraction

import numpy as np
import pytest

import moneta._limbs as limbs_module
from moneta._limbs import (MASK, SHIFT, STEP_K, _MAX, _MIN, limbs_matmul, limbs_mul, limbs_round_div_pow10,
                           limbs_sum, limbs_to_int64, normalize_limbs, split_int64)
from moneta._rounding import RoundingMode

MIN, MAX = -(2 ** 63), 2 ** 63 - 1
MODES = [m.value for m in RoundingMode]
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


def fits(v: int) -> bool:
    return MIN <= v <= MAX


def random_mantissa(rng: random.Random) -> int:
    """偏向边界的随机 int64：极值、分块边界、各数量级。"""
    if rng.random() < 0.15:
        return rng.choice([MIN, MAX, MIN + 1, MAX - 1, 0, 1, -1, int(MASK), -int(MASK), 1 << 21, -(1 << 21),
                           1 << 42, -(1 << 42), (1 << 42) - 1])
    bits = rng.choice([1, 8, 20, 21, 22, 41, 42, 43, 62, 63])
    return max(MIN, min(MAX, rng.randint(-(1 << bits), 1 << bits)))


def value(limbs, index=()) -> int:
    """分块 → Python int（取第 index 个元素）。"""
    return sum(int(np.asarray(limb)[index]) << (SHIFT * i) for i, limb in enumerate(limbs))


def to_limbs(v: int, n: int):
    """Python int → n 块规整表示（0 维数组）。"""
    out = []
    for _ in range(n - 1):
        out.append(np.asarray(v & int(MASK), dtype=np.int64))
        v >>= SHIFT
    out.append(np.asarray(v, dtype=np.int64))
    return out


def limbs_for(v: int):
    return to_limbs(v, max(4, (abs(v).bit_length() + SHIFT) // SHIFT + 1))


def assert_normalized(limbs):
    assert len(limbs) >= 4
    for limb in limbs[:-1]:
        assert np.all((limb >= 0) & (limb <= MASK))
    top = np.asarray(limbs[-1])
    if top.size:
        assert int(top.max()) < 1 << 42 and int(top.min()) > -(1 << 42)


# ===========================================================================
# 1. 常量与导出
# ===========================================================================


class TestConstants:
    def test_values(self):
        assert (SHIFT, int(MASK), STEP_K) == (21, (1 << 21) - 1, 1 << 19)
        assert (_MAX, _MIN) == (MAX, MIN)
        assert 3 * STEP_K * (1 << 42) + (1 << 43) <= MAX  # 单段同一位权 3 项块积之和，加上进位

    def test_exports(self):
        assert "finalize_limbs" not in limbs_module.__all__ and not hasattr(limbs_module, "finalize_limbs")
        for name in limbs_module.__all__:
            assert hasattr(limbs_module, name)


# ===========================================================================
# 2. split_int64 / normalize_limbs
# ===========================================================================


class TestSplitAndNormalize:
    def test_split_random(self):
        rng = random.Random(1)
        xs = np.array([random_mantissa(rng) for _ in range(3000)], dtype=np.int64)
        parts = split_int64(xs)
        assert len(parts) == 3 and all(p.dtype == np.int64 for p in parts)
        assert np.all((parts[0] >= 0) & (parts[0] <= MASK)) and np.all((parts[1] >= 0) & (parts[1] <= MASK))
        assert np.all((parts[2] >= -(1 << 21)) & (parts[2] < 1 << 21))
        assert split_int64(np.array([MIN]))[2].tolist() == [-(1 << 21)]  # 最高块的下界由 MIN 取到
        for i, x in enumerate(xs.tolist()):
            assert value(parts, i) == x

    def test_normalize_random_unnormalized_input(self):
        rng = random.Random(2)
        for _ in range(5000):
            raw = [np.asarray(rng.randint(-(1 << 61), 1 << 61), dtype=np.int64) for _ in range(rng.randint(1, 7))]
            want = value(raw)
            norm = normalize_limbs(raw)
            assert_normalized(norm)
            assert value(norm) == want

    def test_dynamic_expansion(self):
        top = np.asarray(1 << 61, dtype=np.int64)
        norm = normalize_limbs([np.asarray(0, dtype=np.int64)] * 3 + [top])
        assert len(norm) > 4 and value(norm) == (1 << 61) << 63
        assert_normalized(norm)
        neg = normalize_limbs([np.asarray(0, dtype=np.int64)] * 3 + [-top])
        assert value(neg) == -((1 << 61) << 63) and int(neg[-1]) < 0

    def test_short_input_padded_to_four_limbs(self):
        for raw in ([np.asarray(-1, dtype=np.int64)], [np.asarray(5, dtype=np.int64)] * 2):
            norm = normalize_limbs(raw)
            assert len(norm) == 4 and value(norm) == value(raw)

    def test_arrays_and_empty(self):
        raw = [np.array([[MAX, -1], [0, MIN]]), np.array([[1, -1], [MAX, 0]])]
        norm = normalize_limbs(raw)
        for idx in np.ndindex(2, 2):
            assert value(norm, idx) == value(raw, idx)
        empty = normalize_limbs(split_int64(np.zeros((0, 3), dtype=np.int64)))
        assert len(empty) >= 4 and all(limb.shape == (0, 3) for limb in empty)


# ===========================================================================
# 3. limbs_to_int64
# ===========================================================================


class TestToInt64:
    @pytest.mark.parametrize("v", [0, 1, -1, MAX, MIN, MAX - 1, MIN + 1, 1 << 62, -(1 << 62), MASK, -MASK])
    def test_fits(self, v):
        for n in (4, 5, 7):  # 高位块数量不同：-1 的符号扩展必须在每一块上都成立
            got = limbs_to_int64(to_limbs(v, n))
            assert got.dtype == np.int64 and int(got) == v

    @pytest.mark.parametrize("v", [MAX + 1, MIN - 1, 1 << 64, -(1 << 64), 1 << 84, -(1 << 84), (1 << 63) + (1 << 21)])
    def test_overflow(self, v):
        for n in (4, 5, 7):
            with pytest.raises(OverflowError, match="超出 int64 范围"):
                limbs_to_int64(to_limbs(v, n))

    def test_random_around_bounds(self):
        rng = random.Random(3)
        for _ in range(5000):
            v = rng.choice([rng.randint(MIN - 1000, MIN + 1000), rng.randint(MAX - 1000, MAX + 1000),
                            rng.randint(-(1 << 130), 1 << 130), rng.randint(MIN, MAX)])
            limbs = limbs_for(v)
            if fits(v):
                assert int(limbs_to_int64(limbs)) == v
            else:
                with pytest.raises(OverflowError):
                    limbs_to_int64(limbs)

    def test_partial_overflow_in_array_raises(self):
        limbs = normalize_limbs([np.array([MAX, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 1])])
        with pytest.raises(OverflowError):
            limbs_to_int64(limbs)

    def test_short_input_is_normalized_first(self):
        assert limbs_to_int64(split_int64(np.array([MIN, MAX]))).tolist() == [MIN, MAX]
        assert limbs_to_int64(split_int64(np.zeros(0, dtype=np.int64))).shape == (0,)


# ===========================================================================
# 4. limbs_mul
# ===========================================================================


class TestMul:
    def test_random_against_python(self):
        rng = random.Random(4)
        a = [random_mantissa(rng) for _ in range(3000)]
        b = [random_mantissa(rng) for _ in range(3000)]
        limbs = limbs_mul(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
        assert_normalized(limbs)
        for i, (x, y) in enumerate(zip(a, b)):
            assert value(limbs, i) == x * y, (x, y)

    @pytest.mark.parametrize("x, y", [(MIN, MIN), (MIN, MAX), (MAX, MAX), (MIN, -1), (MIN, 1), (0, MIN),
                                      (int(MASK), int(MASK))])
    def test_extremes(self, x, y):
        assert value(limbs_mul(np.asarray(x), np.asarray(y))) == x * y

    def test_broadcast_and_empty(self):
        a, b = np.array([[1], [-2]]), np.array([MAX, MIN, 3])
        limbs = limbs_mul(a, b)
        assert limbs[0].shape == (2, 3)
        for i, j in np.ndindex(2, 3):
            assert value(limbs, (i, j)) == int(a[i, 0]) * int(b[j])
        empty = limbs_mul(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        assert limbs_to_int64(empty).shape == (0,)


# ===========================================================================
# 5. limbs_matmul
# ===========================================================================


class TestMatmul:
    def test_random_against_python(self):
        rng = random.Random(5)
        for _ in range(1500):
            n, k, m = rng.randint(1, 3), rng.randint(1, 6), rng.randint(1, 3)
            a = [[random_mantissa(rng) for _ in range(k)] for _ in range(n)]
            b = [[random_mantissa(rng) for _ in range(m)] for _ in range(k)]
            limbs = limbs_matmul(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
            assert_normalized(limbs)
            for i, j in np.ndindex(n, m):
                assert value(limbs, (i, j)) == sum(a[i][t] * b[t][j] for t in range(k))

    def test_vector_shapes(self):
        a, b = np.array([MIN, MAX, 3]), np.array([MIN, MIN, -1])
        assert value(limbs_matmul(a, b)) == MIN * MIN + MAX * MIN - 3
        mat = np.array([[1, 2, 3], [MAX, MAX, MAX]])
        limbs = limbs_matmul(mat, a)
        assert [value(limbs, (i,)) for i in range(2)] == [MIN + 2 * MAX + 9, MAX * (MIN + MAX + 3)]
        limbs = limbs_matmul(a, mat.T)
        assert [value(limbs, (i,)) for i in range(2)] == [MIN + 2 * MAX + 9, MAX * (MIN + MAX + 3)]
        batch = limbs_matmul(np.stack([mat, -mat]), np.stack([mat.T, mat.T]))
        assert value(batch, (1, 1, 1)) == -3 * MAX * MAX

    def test_long_inner_dimension_spans_segments(self):
        k = STEP_K * 2 + 5
        a = np.full(k, MIN, dtype=np.int64)
        assert value(limbs_matmul(a, a)) == k * MIN * MIN
        top = np.full(k, MAX, dtype=np.int64)  # 三块全满：同一位权 3 项块积都取到 ≈ 2^42，单段和最接近上界
        assert value(limbs_matmul(top, top)) == k * MAX * MAX and value(limbs_matmul(top, a)) == k * MAX * MIN
        rng = np.random.default_rng(6)
        b = rng.choice(np.array([MIN + 1, MAX, -1, 1, 0], dtype=np.int64), k)  # 不含 MIN：下面要取负
        assert value(limbs_matmul(a, b)) == MIN * int(b.astype(object).sum())
        mat = np.stack([a, -b])
        limbs = limbs_matmul(mat, np.stack([b, a], axis=1))
        assert value(limbs, (1, 0)) == -int((b.astype(object) * b.astype(object)).sum())

    def test_empty_and_errors(self):
        assert limbs_to_int64(limbs_matmul(np.zeros((0, 3), np.int64), np.zeros((3, 2), np.int64))).shape == (0, 2)
        assert limbs_to_int64(limbs_matmul(np.zeros((2, 0), np.int64), np.zeros((0, 2), np.int64))).tolist() == \
               [[0, 0], [0, 0]]
        with pytest.raises(ValueError):
            limbs_matmul(np.int64(5), np.int64(6))  # 标量不能做矩阵乘法
        with pytest.raises(ValueError):
            limbs_matmul(np.ones((2, 3), np.int64), np.ones((2, 3), np.int64))


# ===========================================================================
# 6. limbs_sum
# ===========================================================================


class TestSum:
    def test_random_against_python(self):
        rng = random.Random(7)
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
                    assert np.ravel(limbs_sum(arr, axis=axis)).tolist() == want
                else:
                    with pytest.raises(OverflowError):
                        limbs_sum(arr, axis=axis)

    def test_boundaries_and_keepdims(self):
        assert int(limbs_sum(np.array([MAX, 0]))) == MAX and int(limbs_sum(np.array([MIN, 0]))) == MIN
        assert int(limbs_sum(np.array([MAX, MAX, MIN, MIN + 1]))) == -1  # 中间和超出 int64，最终放得下
        assert int(limbs_sum(np.array([MAX, MAX, MIN, MIN]))) == -2
        assert limbs_sum(np.array([[MAX], [MIN]]), axis=1, keepdims=True).tolist() == [[MAX], [MIN]]
        for bad in ([MAX, 1], [MIN, -1], [MAX] * 3):
            with pytest.raises(OverflowError):
                limbs_sum(np.array(bad))

    def test_long_cancellation(self):
        x = np.full(10 ** 6, MAX, dtype=np.int64)
        x[1::2] = -MAX
        assert int(limbs_sum(x)) == 0 and int(limbs_sum(np.append(x, MAX))) == MAX

    def test_empty(self):
        assert limbs_sum(np.zeros((0, 3), dtype=np.int64), axis=1).shape == (0,)
        assert limbs_sum(np.zeros((0, 3), dtype=np.int64), axis=0).tolist() == [0, 0, 0]
        assert int(limbs_sum(np.zeros(0, dtype=np.int64))) == 0


# ===========================================================================
# 7. limbs_round_div_pow10
# ===========================================================================


class TestRoundDivPow10:
    def test_random_against_fraction(self):
        rng = random.Random(8)
        for _ in range(20000):
            v = rng.choice([rng.randint(-(1 << 140), 1 << 140), rng.randint(-(1 << 70), 1 << 70),
                            rng.randint(MIN - 5, MIN + 5), rng.randint(MAX - 5, MAX + 5), rng.randint(-999, 999)])
            k, mode = rng.randint(0, 45), rng.choice(MODES)
            got = limbs_round_div_pow10(limbs_for(v), k, mode)
            assert_normalized(got)
            assert value(got) == ref_round(Fraction(v, 10 ** k), mode), (v, k, mode)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("k", [1, 11, 12, 13, 24, 25, 36])
    def test_ties_and_sticky_across_segments(self, mode, k):
        # 恰好一半、一半多一点点（sticky 位在前面的段里）、一半少一点点
        p = 10 ** k
        for q in (0, 1, 2, -1, -2, 10 ** 20 + 1):
            for num in (q * p + p // 2, q * p + p // 2 + 1, q * p + p // 2 - 1, q * p, q * p + 1, q * p - 1):
                for v in (num, -num):
                    assert value(limbs_round_div_pow10(limbs_for(v), k, mode)) == ref_round(Fraction(v, p), mode)

    def test_vectorized(self):
        vs = [25, -25, 35, -35, 0, MAX, MIN]
        limbs = limbs_round_div_pow10(split_int64(np.array(vs, dtype=np.int64)), 1, "half_even")
        assert limbs_to_int64(limbs).tolist() == [ref_round(Fraction(v, 10), "half_even") for v in vs]

    def test_k_zero_only_normalizes(self):
        raw = [np.asarray(1 << 60, dtype=np.int64), np.asarray(-3, dtype=np.int64)]
        got = limbs_round_div_pow10(raw, 0, "half_up")
        assert_normalized(got)
        assert value(got) == value(raw)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="rounding 只能是"):
            limbs_round_div_pow10(split_int64(np.array([15])), 1, "nearest")
