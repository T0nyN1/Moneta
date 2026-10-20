"""基于二进制分块（Limbs）的高性能无溢出多精度整数算术模块。

本模块提供一套纯函数式的整数算术内核，采用 2^21 进制（Base 2^21）动态分块列表：
    Value = sum(limbs[i] * 2^(21 * i))

设计原则：
1. 纯函数式与无状态：不依赖外部张量或领域对象，仅操作 NumPy 数组与 limbs 列表。
2. 杜绝溢出回绕：中间计算维持在原生整数安全区间内，长除法带余数精确舍入，溢出检测基于位模式匹配。
3. 高性能 SIMD 位运算：采用二进制 Base 2^21，位移（>> 21）与掩码（& MASK）均为单周期指令。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from ._rounding import decide_bump

# 二进制 Base 2^21 参数
SHIFT = 21
BASE = np.int64(1 << SHIFT)
MASK = np.int64(BASE - 1)

# int64 边界常量
_MAX = (1 << 63) - 1
_MIN = -(1 << 63)

# 内积维分段步长：低两块在 [0, 2^21)、最高块在 [-2^21, 2^21)，块积 |·| ≤ 2^42，同一位权最多 3 项；
# 每段 |·| ≤ 3·K·2^42 = 1.5·2^62 < 2^63，加上进位（< 2^43）也不会溢出
STEP_K = 1 << 19  # 524,288


def split_int64(a: np.ndarray) -> List[np.ndarray]:
    """将 int64 数组分解为 21 位无符号低位块与带符号高位块构成的 limbs 列表（低位在前）。

    Parameters
    ----------
    a : np.ndarray
        输入的 int64 数组。

    Returns
    -------
    List[np.ndarray]
        拆分后的 limbs 列表，前两块位于 [0, 2^21)，最高块带符号、位于 [-2^21, 2^21)。
    """
    a_arr = np.asarray(a, dtype=np.int64)
    q1 = a_arr >> SHIFT
    return [a_arr & MASK, q1 & MASK, q1 >> SHIFT]


def normalize_limbs(limbs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """进位规整器：从低到高推波传递进位。

    采用 >> SHIFT 和 & MASK 进行单周期高性能进位与取余，完全取代 np.divmod。
    规整后除最高块外每块落在 [0, 2^21)，最高块带符号且绝对值 < 2^42。
    当最高块超出 2^42 时，自动向上动态追加新 limb（大数动态扩容）。
    输出至少包含 4 个 limb，以完整覆盖 63 位数值位及更高位。空数组安全。

    Parameters
    ----------
    limbs : Sequence[np.ndarray]
        未完全规整的 limbs 列表（各块可正可负，绝对值 ≤ 1.5·2^62，加上进位不超出 int64）。

    Returns
    -------
    List[np.ndarray]
        规整后规范化的 limbs 列表。
    """
    out: List[np.ndarray] = []
    carry: np.int64 = np.int64(0)
    for limb in limbs[:-1]:
        temp = limb + carry
        carry = temp >> SHIFT
        r = temp & MASK
        out.append(r)
    top = limbs[-1] + carry

    # 保证至少包含 3 个低位块（覆盖 63 位无符号有效位），方便后续位模式检查
    while len(out) < 3:
        r = top & MASK
        top = top >> SHIFT
        out.append(r)

    # 动态扩容：当最高块绝对值达到 2^42 时向上进位扩展新 limb（空数组安全）
    top_arr = np.asarray(top)
    while top_arr.size and (int(top_arr.max()) >= (1 << 42) or int(top_arr.min()) <= -(1 << 42)):
        r = top & MASK
        top = top >> SHIFT
        top_arr = np.asarray(top)
        out.append(r)

    out.append(top)
    return out


def limbs_mul(a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """逐元素乘法：计算 a * b 的精确大数表示（已规整的 limbs 列表）。

    将输入各拆分为 3 个 21 位块进行 3x3 块卷积相乘，得到 5 个中间项并进位规整。
    单项乘积 < 2^42，中间运算完全维持在原生 int64 内部，绝不溢出。

    Parameters
    ----------
    a, b : np.ndarray
        广播兼容的 int64 操作数数组。

    Returns
    -------
    List[np.ndarray]
        已规整的积大数 limbs 列表。
    """
    pa = split_int64(a)
    pb = split_int64(b)
    t: List[Any] = [None] * 5
    for i in range(3):
        for j in range(3):
            c = pa[i] * pb[j]
            t[i + j] = c if t[i + j] is None else t[i + j] + c
    return normalize_limbs(t)


def limbs_matmul(a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """矩阵乘法：计算 a @ b 的精确大数内积（已规整的 limbs 列表）。

    沿内积维度以 STEP_K (2^19 = 524,288) 项为粒度分段切片累加，
    每段累加 |·| ≤ 3·STEP_K·2^42 = 1.5·2^62 < 2^63，规整后再跨段累加，任意长度都不溢出。

    Parameters
    ----------
    a, b : np.ndarray
        满足矩阵乘法维度约束的 int64 数组。

    Returns
    -------
    List[np.ndarray]
        已规整的内积大数 limbs 列表。
    """
    if a.ndim == 0 or b.ndim == 0:
        np.matmul(a, b)  # 标量不能做矩阵乘法，由 NumPy 抛出形状异常

    terms = a.shape[-1]
    acc: Optional[List[np.ndarray]] = None

    for start in range(0, max(terms, 1), STEP_K):
        pa = split_int64(a[..., start:start + STEP_K])
        pb = split_int64(b[start:start + STEP_K] if b.ndim == 1 else b[..., start:start + STEP_K, :])
        t: List[Any] = [None] * 5
        for i in range(3):
            for j in range(3):
                c = np.asarray(np.matmul(pa[i], pb[j]))
                t[i + j] = c if t[i + j] is None else t[i + j] + c
        limbs = normalize_limbs(t)
        if acc is None:
            acc = limbs
        else:
            zero = np.zeros_like(limbs[0])
            n = max(len(acc), len(limbs))
            acc = normalize_limbs([
                (acc[i] if i < len(acc) else zero) + (limbs[i] if i < len(limbs) else zero)
                for i in range(n)
            ])

    return acc


def limbs_sum(arr: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
    """分块精确求和：支持任意 axis 和 keepdims，全程纯 int64 运算。

    将输入数组分解为 3 个 21 位块，沿指定轴求和后进位规整，
    只有真实结果超出 int64 范围时才抛出 OverflowError。
    单块累加在元素数量 <= 4.39 万亿内绝对不溢出。

    Parameters
    ----------
    arr : np.ndarray
        输入 int64 数组。
    axis : Any, optional
        求和轴。
    keepdims : bool, default False
        是否保留轴维度。

    Returns
    -------
    np.ndarray
        精确求和后的 int64 数组。
    """
    pa = split_int64(arr)
    s = [limb.sum(axis=axis, keepdims=keepdims) for limb in pa]
    return limbs_to_int64(normalize_limbs(s))


def limbs_round_div_pow10(limbs: Sequence[np.ndarray], k: int, mode: str = "half_up") -> List[np.ndarray]:
    """大数长除法与融合舍入：计算 round(V / 10^k)，V 为 limbs 列表。

    从最高 limb 向下带余数做竖式长除法（每轮除以至多 10^12），
    借助粘滞位（sticky bit）精确处理小数部分与 1/2 的关系，支持 6 大舍入模式。

    Parameters
    ----------
    limbs : Sequence[np.ndarray]
        输入的规整大数 limbs。
    k : int
        除以 10 的幂次 (k >= 0)。
    mode : str, default "half_up"
        舍入模式，取值为 "half_up", "half_even", "down", "up", "floor", "ceil" 之一。

    Returns
    -------
    List[np.ndarray]
        长除并舍入后的规整 limbs 列表。
    """
    if k == 0:
        return normalize_limbs(limbs)

    sticky = np.zeros(np.shape(limbs[0]), dtype=bool)
    cur_limbs = list(limbs)

    while True:
        c = min(k, 12)
        p = np.int64(10 ** c)
        q_top, rem = np.divmod(cur_limbs[-1], p)
        quotient = [q_top]
        for limb in reversed(cur_limbs[:-1]):
            q, rem = np.divmod(rem * BASE + limb, p)
            quotient.append(q)
        cur_limbs = quotient[::-1]
        k -= c
        if k == 0:
            break
        sticky = sticky | (rem != 0)

    half = p // 2
    bump = decide_bump(
        mode,
        is_negative=cur_limbs[-1] < 0,
        odd_q=(cur_limbs[0] & 1) == 1,
        has_fraction=(rem != 0) | sticky,
        over_half=(rem > half) | ((rem == half) & sticky),
        exact_half=(rem == half) & ~sticky,
        floor_base=True,
    )
    cur_limbs[0] = cur_limbs[0] + bump
    return normalize_limbs(cur_limbs)


def limbs_to_int64(limbs: Sequence[np.ndarray]) -> np.ndarray:
    """将 limbs 列表收口转换为 int64 原生数组。

    通过高位块的位模式匹配（正数全 0，负数全 1 符号扩展）严格检测溢出，
    不含任何标量乘法，彻底杜绝硬件溢出回绕。

    Parameters
    ----------
    limbs : Sequence[np.ndarray]
        规整后的大数 limbs 列表。

    Returns
    -------
    np.ndarray
        int64 类型的 NumPy 数组。

    Raises
    ------
    OverflowError
        真实数值超出 int64 表示范围 [-2^63, 2^63 - 1]。
    """
    if len(limbs) < 4:
        limbs = normalize_limbs(limbs)
    high = limbs[3:]
    is_zero = np.logical_and.reduce([h == 0 for h in high])
    is_neg = np.logical_and.reduce([h == MASK for h in high[:-1]] + [high[-1] == -1])
    if not np.all(is_zero | is_neg):
        raise OverflowError("结果超出 int64 范围")

    low63 = np.asarray((limbs[2] << 42) | (limbs[1] << 21) | limbs[0])
    with np.errstate(over="ignore"):
        res = np.where(is_neg, (low63.astype(np.uint64) | np.uint64(1 << 63)).view(np.int64), low63)
    return np.asarray(res, dtype=np.int64)


__all__ = ["SHIFT", "BASE", "MASK", "STEP_K", "split_int64", "normalize_limbs", "limbs_mul", "limbs_matmul",
           "limbs_sum", "limbs_round_div_pow10", "limbs_to_int64"]
