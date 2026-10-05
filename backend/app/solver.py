# -*- coding: utf-8 -*-
"""平面连杆运动学求解核心。

约束形式采用杆长平方差，避免开根号带来的导数问题：

    f_e(x) = (p_i - p_j)·(p_i - p_j) - L_e^2 = 0

驱动铰点 P（与固定铰点 O 相连的动铰点）的位置完全由驱动角确定，
因此它不作为未知量；其余每个动铰点 2 个未知数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import nnls

# ---------------------------------------------------------------------------
# 数值参数
# ---------------------------------------------------------------------------
RES_TOL = 1.0e-8            # 残差（长度平方量级）通过阈值，约 0.1 毫米（L~100 量级）
MAX_NEWTON_ITER = 60
ROOT_MATCH_TOL = 0.25       # 多解判同：坐标最大距离（机构长度单位）
CANDIDATE_MATCH_TOL = 0.25
JUMP_TOL = 0.12             # 相邻角点间最大位移 / 机构平均杆长 的阈值
SING_SV_TOL = 1.0e-3        # 最小奇异值（行归一化后）低于此值视为奇异


@dataclass
class Compiled:
    """编译后的机构数据。"""

    joints: list[dict]                 # 原始关节定义副本（含 id/type/x/y）
    links: list[dict]                  # 原始杆件定义副本
    driver: dict                       # 驱动定义
    joint_ids: list[str]               # 全部关节 id，顺序即前端使用顺序
    fixed: dict[str, np.ndarray]       # 固定铰点坐标
    moving: list[str]                  # 动铰点 id（含驱动铰点）
    free: list[str]                    # 自由未知动铰点 id（不含驱动铰点）
    free_index: dict[str, int]         # 自由铰点 -> 下标
    pivot: str                         # 固定驱动铰点 id
    crank: str                         # 驱动动铰点 id
    crank_len: float                   # 驱动杆长
    # 自由约束：两端均为自由铰点，或一端固定（非 pivot）一端自由
    rows: list[tuple[str, str, float]]  # (a, b, L)
    n_free: int
    mean_len: float

    def driver_position(self, theta: float) -> np.ndarray:
        o = self.fixed[self.pivot]
        return o + self.crank_len * np.array([math.cos(theta), math.sin(theta)])

    def positions(self, free_xy: np.ndarray, theta: float) -> dict[str, np.ndarray]:
        """把自由变量 + 驱动角组装为全部关节坐标。"""
        pos: dict[str, np.ndarray] = {jid: p.copy() for jid, p in self.fixed.items()}
        pos[self.crank] = self.driver_position(theta)
        for k, jid in enumerate(self.free):
            pos[jid] = np.array(free_xy[2 * k:2 * k + 2], dtype=float)
        return pos

    def init_free(self) -> np.ndarray:
        x = np.zeros(2 * self.n_free)
        for k, jid in enumerate(self.free):
            j = next(jj for jj in self.joints if jj["id"] == jid)
            x[2 * k] = float(j["x"])
            x[2 * k + 1] = float(j["y"])
        return x


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------
def compile_mech(payload: dict) -> Compiled:
    joints = payload["joints"]
    links = payload["links"]
    driver = payload["driver"]

    fixed = {j["id"]: np.array([float(j["x"]), float(j["y"])])
             for j in joints if j["type"] == "fixed"}
    moving = [j["id"] for j in joints if j["type"] == "moving"]
    free = [m for m in moving if m != driver["joint"]]

    pivot = driver["pivot"]
    crank = driver["joint"]
    # 驱动杆长由 pivot-crank 初始距离确定
    crank_len = float(np.linalg.norm(fixed[pivot] -
                                     next(np.array([float(j["x"]), float(j["y"])])
                                          for j in joints if j["id"] == crank)))

    rows: list[tuple[str, str, float]] = []
    used_lengths: list[float] = []
    for ln in links:
        a, b, L = ln["a"], ln["b"], float(ln["length"])
        if a == crank or b == crank:
            continue                       # 接驱动铰点的约束由驱动角满足，不进方程组
        if a in fixed and b in fixed:
            continue                       # 固定-固定杆仅在校验阶段处理
        if a in fixed:
            rows.append((a, b, L))
        elif b in fixed:
            rows.append((b, a, L))
        else:
            rows.append((a, b, L))
        used_lengths.append(L)
    rows.append((crank_len,))  # placeholder removed below
    rows.pop()

    mean_len = float(np.mean(used_lengths + [crank_len]))

    return Compiled(
        joints=[dict(j) for j in joints],
        links=[dict(l) for l in links],
        driver=dict(driver),
        joint_ids=[j["id"] for j in joints],
        fixed=fixed,
        moving=moving,
        free=free,
        free_index={jid: k for k, jid in enumerate(free)},
        pivot=pivot,
        crank=crank,
        crank_len=crank_len,
        rows=rows,
        n_free=len(free),
        mean_len=mean_len,
    )


# ---------------------------------------------------------------------------
# 残差 / 雅可比
# ---------------------------------------------------------------------------
def residual(comp: Compiled, x: np.ndarray, theta: float) -> np.ndarray:
    pos = comp.positions(x, theta)
    f = np.zeros(len(comp.rows))
    for e, (a, b, L) in enumerate(comp.rows):
        d = pos[a] - pos[b]
        f[e] = d @ d - L * L
    return f


def jacobian(comp: Compiled, x: np.ndarray, theta: float) -> np.ndarray:
    pos = comp.positions(x, theta)
    J = np.zeros((len(comp.rows), 2 * comp.n_free))
    for e, (a, b, _L) in enumerate(comp.rows):
        da = pos[a] - pos[b]
        db = -da
        for p, g in ((a, da), (b, db)):
            k = comp.free_index.get(p)
            if k is not None:
                J[e, 2 * k:2 * k + 2] = 2.0 * g
    return J


def _row_scales(J: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(J, axis=1), 1e-12)


def min_singular(comp: Compiled, x: np.ndarray, theta: float) -> float:
    """行归一化后的最小奇异值，用于奇异汇合判定。"""
    J = jacobian(comp, x, theta)
    scales = _row_scales(J)
    Jn = J / scales[:, None]
    if Jn.size == 0:
        return 1.0
    s = np.linalg.svd(Jn, compute_uv=False)
    return float(s[-1]) if len(s) else 1.0


def rank_ok(comp: Compiled, x: np.ndarray, theta: float) -> bool:
    """初态雅可比必须列满秩（机构非奇异且单自由度）。"""
    J = jacobian(comp, x, theta)
    ncol = 2 * comp.n_free
    if J.shape[0] < ncol:
        return False
    scales = _row_scales(J)
    Jn = J / scales[:, None]
    s = np.linalg.svd(Jn, compute_uv=False)
    if len(s) < ncol:
        return False
    return bool(s[-1] > 1.0e-6)


# ---------------------------------------------------------------------------
# 牛顿迭代
# ---------------------------------------------------------------------------
def newton(comp: Compiled, x0: np.ndarray, theta: float,
           max_iter: int = MAX_NEWTON_ITER) -> Optional[np.ndarray]:
    """阻尼牛顿法求解。x0 同时给出装配分支提示（就近收敛）。"""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        f = residual(comp, x, theta)
        if np.max(np.abs(f)) < RES_TOL:
            return x
        J = jacobian(comp, x, theta)
        try:
            dx, *_ = np.linalg.lstsq(J, -f, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dx)):
            return None
        # 阻尼线搜索：平方差之和下降
        base = np.max(np.abs(f))
        step = 1.0
        accepted = False
        for _tries in range(12):
            xn = x + step * dx
            if np.all(np.isfinite(xn)):
                fn = residual(comp, xn, theta)
                if np.max(np.abs(fn)) < base * (1.0 - 0.25 * step) + 1e-14:
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return None
        x = xn
    f = residual(comp, x, theta)
    return x if np.max(np.abs(f)) < RES_TOL else None


def tangent(comp: Compiled, x: np.ndarray, theta: float) -> Optional[np.ndarray]:
    """切向 dx/dtheta：Jx·x' + Jtheta = 0。

    驱动铰点按 dP/dtheta = L·(-sin, cos) 运动，把它对残差的贡献作为右端项。
    """
    pos = comp.positions(x, theta)
    cr = comp.crank
    dp = comp.crank_len * np.array([-math.sin(theta), math.cos(theta)])
    g = np.zeros(len(comp.rows))
    for e, (a, b, _L) in enumerate(comp.rows):
        if a == cr:
            g[e] = 2.0 * (pos[a] - pos[b]) @ dp
        elif b == cr:
            g[e] = 2.0 * (pos[b] - pos[a]) @ dp
    J = jacobian(comp, x, theta)
    try:
        xp, *_ = np.linalg.lstsq(J, -g, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return xp if np.all(np.isfinite(xp)) else None


# ---------------------------------------------------------------------------
# 初态校验残差（含驱动杆与固定-固定杆）
# ---------------------------------------------------------------------------
def closure_residual(payload: dict) -> tuple[float, list[str]]:
    """返回 (归一化最大长度误差, 各杆误差描述)。"""
    joints = {j["id"]: j for j in payload["joints"]}

    def pos_of(jid: str) -> np.ndarray:
        j = joints[jid]
        return np.array([float(j["x"]), float(j["y"])])

    worst = 0.0
    details: list[str] = []
    for ln in payload["links"]:
        d = float(np.linalg.norm(pos_of(ln["a"]) - pos_of(ln["b"])))
        err = abs(d - float(ln["length"]))
        worst = max(worst, err)
        if err > 1e-6:
            details.append(f"杆 {ln['a']}–{ln['b']}：给定 {ln['length']:g}，实际 {d:.4g}（差 {err:.3g}）")
    return worst, details


# ---------------------------------------------------------------------------
# 多起点枚举全部解
# ---------------------------------------------------------------------------
def _seeds(comp: Compiled, x_hint: np.ndarray, theta: float, radius: float) -> np.ndarray:
    """围绕提示解构造一组起点：本解、轴对齐扰动、随机扰动、网格散布。"""
    rng = np.random.defaultRNG if False else np.random.default_rng(20261004)
    n = comp.n_free
    seeds = [np.array(x_hint, dtype=float)]

    # 1) 每个铰点沿若干方向的较大扰动
    dirs = [np.array(v, float) for v in
            [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]]
    for k in range(n):
        for d in dirs:
            v = np.array(x_hint, dtype=float)
            v[2 * k:2 * k + 2] += radius * d / np.linalg.norm(d)
            seeds.append(v)

    # 2) 随机扰动（多尺度）
    for scale in (0.35, 0.7, 1.2):
        for _ in range(10 * n + 8):
            v = np.array(x_hint, dtype=float) + rng.normal(0.0, radius * scale, 2 * n)
            seeds.append(v)

    # 3) 以驱动铰点 / 固定铰点为中心的宽散布
    pos = comp.positions(x_hint, theta)
    centers = [pos[comp.crank]] + list(comp.fixed.values())
    for c in centers:
        for _ in range(4):
            v = np.zeros(2 * n)
            for k, jid in enumerate(comp.free):
                ang = rng.uniform(0, 2 * math.pi)
                rr = radius * rng.uniform(0.2, 1.4)
                v[2 * k:2 * k + 2] = c + rr * np.array([math.cos(ang), math.sin(ang)])
            seeds.append(v)

    return np.array(seeds)


def enumerate_roots(comp: Compiled, x_hint: np.ndarray, theta: float,
                    match_tol: float = ROOT_MATCH_TOL) -> list[np.ndarray]:
    """枚举该驱动角下的全部实解（坐标距离去重）。"""
    radius = max(comp.mean_len, 1e-6)
    roots: list[np.ndarray] = []

    def consider(x: Optional[np.ndarray]) -> None:
        if x is None or not np.all(np.isfinite(x)):
            return
        for r in roots:
            if np.max(np.abs(r - x)) < match_tol:
                return
        roots.append(x)

    # 提示解本身优先（保留当前分支）
    consider(newton(comp, x_hint, theta))
    for s in _seeds(comp, x_hint, theta, radius):
        consider(newton(comp, s, theta))
    return roots


def all_positions(comp: Compiled, x: np.ndarray, theta: float) -> dict:
    """解 -> 前端使用的 {joint_id: [x, y]}。"""
    pos = comp.positions(x, theta)
    return {jid: [float(pos[jid][0]), float(pos[jid][1])] for jid in comp.joint_ids}


def config_distance(comp: Compiled, x1: np.ndarray, x2: np.ndarray) -> float:
    """自由铰点坐标的最大位移（归一化）。"""
    if x1 is None or x2 is None:
        return math.inf
    return float(np.max(np.abs(x1 - x2)) / max(comp.mean_len, 1e-12))
