"""平面连杆机构装配分支追踪求解器。

模型：杆铰网络（每个连杆为二力杆，关节为转动副）。
未知量：除驱动杆端点外的全部动铰点坐标。
方程：除驱动杆外的全部杆长约束 |p_a - p_b|^2 = L^2（按 L^2 归一化）。
单自由度条件：2 * 动铰点数 - 连杆数 == 1，此时方程组为方阵。

追踪方式：以驱动角 theta 为参数做连续延拓（预测 + Newton 校正），
全程留在原装配分支；监测 Jacobian 行列式符号与 Newton 收敛性，
奇异（汇合/折点）处二分定位边界（误差 <= 0.1°），并在边界之外
用多种子搜索后续候选位形，暂停等待用户选支；无候选则角段终止。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import root as _sp_root

ANGLE_TOL_DEG = 0.1          # 可达角段边界精度（度）
CLOSURE_RTOL = 1e-3          # 初始闭合容差（相对杆长）
RES_TOL = 1e-10              # Newton 收敛残差容差（归一化）
SOLVE_TOL = 1e-9             # 解的可接受残差
MAX_CANDIDATES = 6           # 奇异点处最多呈现的候选位形数
PROBE_OFFSETS_DEG = (0.75, 2.5, 7.5)  # 奇异点外候选探测距离（度）
MAX_SWEEP_STEPS = 20000      # 单角段步数保护上限


class MechanismError(Exception):
    """机构输入校验或求解失败，携带稳定错误码与中文说明。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _err(code: str, msg: str):
    raise MechanismError(code, msg)


class Mechanism:
    """杆铰网络模型：解析、校验输入，计算驱动初始角。"""

    def __init__(self, data: dict):
        if not isinstance(data, dict):
            _err("bad_request", "请求体必须是机构 JSON 对象")
        raw_joints = data.get("joints")
        raw_links = data.get("links")
        driver = data.get("driver")
        if not isinstance(raw_joints, list) or not raw_joints:
            _err("invalid_mechanism", "缺少关节列表 joints")
        if not isinstance(raw_links, list) or not raw_links:
            _err("invalid_mechanism", "缺少连杆列表 links")
        if not isinstance(driver, dict):
            _err("invalid_mechanism", "缺少驱动杆定义 driver")

        # ---- 关节 ----
        self.joints: dict = {}
        self.joint_order: list = []
        for j in raw_joints:
            if not isinstance(j, dict) or "id" not in j:
                _err("invalid_mechanism", f"关节定义无效: {j!r}")
            jid = j["id"]
            try:
                x = float(j["x"]); y = float(j["y"])
            except (KeyError, TypeError, ValueError):
                _err("invalid_mechanism", f"关节 {jid} 坐标无效")
            if not (math.isfinite(x) and math.isfinite(y)):
                _err("invalid_mechanism", f"关节 {jid} 坐标非有限数")
            if jid in self.joints:
                _err("invalid_mechanism", f"关节 id 重复: {jid}")
            self.joints[jid] = {"id": jid, "x": x, "y": y, "fixed": bool(j.get("fixed", False))}
            self.joint_order.append(jid)

        # ---- 连杆 ----
        self.links: dict = {}
        self.link_order: list = []
        seen_pairs = set()
        for l in raw_links:
            if not isinstance(l, dict) or "id" not in l:
                _err("invalid_mechanism", f"连杆定义无效: {l!r}")
            lid = l["id"]
            a, b = l.get("a"), l.get("b")
            try:
                length = float(l.get("length"))
            except (TypeError, ValueError):
                _err("non_positive_length", f"连杆 {lid} 长度无效")
            if lid in self.links:
                _err("invalid_mechanism", f"连杆 id 重复: {lid}")
            if a not in self.joints or b not in self.joints:
                _err("invalid_reference", f"连杆 {lid} 引用了不存在的关节")
            if a == b:
                _err("invalid_mechanism", f"连杆 {lid} 两端关节相同")
            if not math.isfinite(length) or length <= 0:
                _err("non_positive_length", f"连杆 {lid} 长度必须为正数")
            pair = frozenset((a, b))
            if pair in seen_pairs:
                _err("invalid_mechanism", f"关节对 {a}-{b} 之间存在重复连杆")
            seen_pairs.add(pair)
            self.links[lid] = {"id": lid, "a": a, "b": b, "length": length}
            self.link_order.append(lid)

        self.moving_ids = [j for j in self.joint_order if not self.joints[j]["fixed"]]
        self.fixed_ids = [j for j in self.joint_order if self.joints[j]["fixed"]]
        if not self.fixed_ids:
            _err("invalid_mechanism", "至少需要一个固定铰点")
        if not self.moving_ids:
            _err("invalid_mechanism", "至少需要一个动铰点")

        # ---- 自由度与闭环数 ----
        m = len(self.moving_ids)
        nl = len(self.links)
        if 2 * m - nl != 1:
            _err("not_single_dof",
                 f"机构非单自由度：2×动铰点数({m}) − 连杆数({nl}) = {2 * m - nl} ≠ 1")
        loops = nl - m  # 固定铰点视为同一机架节点后的独立闭环数
        if loops not in (1, 2):
            _err("invalid_topology", f"机构须含 1 或 2 个闭环，当前为 {loops}")
        self._check_connectivity()

        # ---- 驱动杆 ----
        dl = driver.get("link_id")
        if dl not in self.links:
            _err("invalid_reference", f"驱动杆引用了不存在的连杆: {dl!r}")
        link = self.links[dl]
        ja, jb = self.joints[link["a"]], self.joints[link["b"]]
        if ja["fixed"] == jb["fixed"]:
            _err("invalid_driver", "驱动杆必须连接一个固定铰点和一个动铰点")
        self.driver_link = link
        self.pivot_id = link["a"] if ja["fixed"] else link["b"]
        self.crank_id = link["b"] if ja["fixed"] else link["a"]

        try:
            ts = float(driver["theta_start"]); te = float(driver["theta_end"])
        except (KeyError, TypeError, ValueError):
            _err("invalid_driver", "驱动角度范围无效")
        if not (math.isfinite(ts) and math.isfinite(te)) or not ts < te:
            _err("invalid_driver", "角度范围无效：要求 theta_start < theta_end")
        if te - ts > 720.0:
            _err("invalid_driver", "角度范围跨度不能超过 720°")
        self.theta_start, self.theta_end = ts, te

        # ---- 初始闭合检查 ----
        for l in self.links.values():
            pa, pb = self.joints[l["a"]], self.joints[l["b"]]
            d = math.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"])
            if abs(d - l["length"]) > CLOSURE_RTOL * l["length"] + 1e-9:
                _err("not_closed",
                     f"初始构型不闭合：连杆 {l['id']} 实际长度 {d:.6g} 与设定 {l['length']:.6g} 不符")

        # ---- 驱动初始角（由曲柄端点初始坐标反算）----
        piv, ce = self.joints[self.pivot_id], self.joints[self.crank_id]
        base = math.degrees(math.atan2(ce["y"] - piv["y"], ce["x"] - piv["x"]))
        fits = [a for a in (base, base + 360.0, base - 360.0) if ts - 1e-9 <= a <= te + 1e-9]
        if not fits:
            _err("invalid_driver",
                 f"驱动杆初始角度 {base:.2f}° 不在范围 [{ts}, {te}] 内")
        self.theta_init = min(fits, key=lambda a: abs(a - base))

    def _check_connectivity(self):
        adj = {j: [] for j in self.joints}
        for l in self.links.values():
            adj[l["a"]].append(l["b"])
            adj[l["b"]].append(l["a"])
        for j, nb in adj.items():
            if not nb:
                _err("invalid_topology", f"关节 {j} 没有连接任何连杆")
        seen, stack = set(), [self.joint_order[0]]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj[n])
        if len(seen) != len(self.joints):
            _err("invalid_topology", "机构不连通：存在孤立的关节或连杆")


class Solver:
    """给定驱动角 theta，求解全部杆长约束（Newton + SciPy 兜底）。"""

    def __init__(self, mech: Mechanism):
        self.mech = mech
        self.crank_len = mech.driver_link["length"]
        self.free_ids = [j for j in mech.moving_ids if j != mech.crank_id]
        self.index = {j: 2 * k for k, j in enumerate(self.free_ids)}
        self.eq_links = [lid for lid in mech.link_order if lid != mech.driver_link["id"]]
        self.n = 2 * len(self.free_ids)
        self.scale = max(l["length"] for l in mech.links.values())
        piv = mech.joints[mech.pivot_id]
        self.pivot = np.array([piv["x"], piv["y"]], dtype=float)

    # ---- 构型与方程 ----
    def crank_pos(self, theta_deg: float) -> np.ndarray:
        t = math.radians(theta_deg)
        return self.pivot + self.crank_len * np.array([math.cos(t), math.sin(t)])

    def positions(self, v, theta_deg: float) -> dict:
        """由自由变量 v 与驱动角还原全部关节坐标。"""
        pos = {j: np.array([self.mech.joints[j]["x"], self.mech.joints[j]["y"]])
               for j in self.mech.fixed_ids}
        pos[self.mech.crank_id] = self.crank_pos(theta_deg)
        for j in self.free_ids:
            i = self.index[j]
            pos[j] = np.array([v[i], v[i + 1]], dtype=float)
        return pos

    def residuals(self, v, theta_deg: float) -> np.ndarray:
        pos = self.positions(np.asarray(v, dtype=float), theta_deg)
        r = np.empty(len(self.eq_links))
        for k, lid in enumerate(self.eq_links):
            l = self.mech.links[lid]
            d = pos[l["a"]] - pos[l["b"]]
            r[k] = (d @ d - l["length"] ** 2) / l["length"] ** 2
        return r

    def jacobian(self, v, theta_deg: float) -> np.ndarray:
        pos = self.positions(np.asarray(v, dtype=float), theta_deg)
        J = np.zeros((len(self.eq_links), self.n))
        for k, lid in enumerate(self.eq_links):
            l = self.mech.links[lid]
            d = (pos[l["a"]] - pos[l["b"]]) * (2.0 / l["length"] ** 2)
            for jid, sign in ((l["a"], 1.0), (l["b"], -1.0)):
                if jid in self.index:
                    i = self.index[jid]
                    J[k, i] += sign * d[0]
                    J[k, i + 1] += sign * d[1]
        return J

    def det_sign(self, v, theta_deg: float) -> float:
        """Jacobian 行列式（符号用于识别装配分支，0 为奇异）。"""
        try:
            return float(np.linalg.det(self.jacobian(v, theta_deg)))
        except np.linalg.LinAlgError:
            return 0.0

    # ---- 求解 ----
    def newton(self, v0, theta_deg: float, max_iter: int = 60):
        v = np.array(v0, dtype=float)
        for _ in range(max_iter):
            r = self.residuals(v, theta_deg)
            rn = float(np.max(np.abs(r))) if r.size else 0.0
            if rn < RES_TOL:
                return v, True
            J = self.jacobian(v, theta_deg)
            try:
                dv = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
            accepted = False
            alpha = 1.0
            for _ in range(8):  # 简单回溯线搜索，保证残差下降
                cand = v + alpha * dv
                if float(np.max(np.abs(self.residuals(cand, theta_deg)))) < rn:
                    v = cand
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        r = self.residuals(v, theta_deg)
        return v, float(np.max(np.abs(r))) < SOLVE_TOL

    def solve(self, v0, theta_deg: float):
        """返回解向量或 None。先 Newton，失败则用 SciPy hybr 兜底。"""
        v, ok = self.newton(v0, theta_deg)
        if ok:
            return v
        try:
            sol = _sp_root(self.residuals, np.asarray(v0, dtype=float),
                           args=(theta_deg,), jac=self.jacobian, method="hybr")
            x = np.asarray(sol.x, dtype=float)
            if float(np.max(np.abs(self.residuals(x, theta_deg)))) < SOLVE_TOL:
                return x
        except Exception:
            pass
        return None

    def initial_v(self) -> np.ndarray:
        m = self.mech
        return np.array([m.joints[j][c] for j in self.free_ids for c in ("x", "y")], dtype=float)


# ---------------------------------------------------------------------------
# 会话：角段延拓、奇异定位、候选搜索
# ---------------------------------------------------------------------------

def new_state() -> dict:
    return {"status": "running", "segments": [], "frontiers": [],
            "paused": None, "messages": []}


def start_session(mech: Mechanism) -> dict:
    """由用户给定的初始坐标出发，建立两个方向的扫描前沿。"""
    solver = Solver(mech)
    v = solver.solve(solver.initial_v(), mech.theta_init)
    if v is None:
        _err("not_closed", "初始构型无法满足全部杆长约束（不闭合或初始奇异）")
    state = new_state()
    state["messages"].append(
        f"分析开始：驱动初始角 θ={mech.theta_init:.2f}°，"
        f"范围 [{mech.theta_start}, {mech.theta_end}]°")
    if mech.theta_init < mech.theta_end - 1e-12:
        state["frontiers"].append({"theta": mech.theta_init, "v": v.tolist(), "dir": +1})
    if mech.theta_init > mech.theta_start + 1e-12:
        state["frontiers"].append({"theta": mech.theta_init, "v": v.tolist(), "dir": -1})
    return state


def run_session(mech: Mechanism, state: dict):
    """依次推进各扫描前沿，直到全部完成或在奇异点暂停。"""
    solver = Solver(mech)
    while state["frontiers"] and state["status"] != "paused":
        frontier = state["frontiers"].pop(0)
        _sweep(solver, mech, state, frontier)
    if state["status"] != "paused":
        state["status"] = "done"
        state["messages"].append("分析完成：已覆盖全部可达角段")


def apply_choice(mech: Mechanism, state: dict, index: int):
    """用户在奇异汇合处选定候选位形后，续算。"""
    p = state.get("paused")
    if state.get("status") != "paused" or not p:
        _err("invalid_state", "当前会话不在暂停状态，无法选支")
    if not (0 <= index < len(p["candidates"])):
        _err("invalid_reference", f"候选编号 {index} 无效")
    solver = Solver(mech)
    v_chosen = np.asarray(p["candidates"][index], dtype=float)
    v_star = np.asarray(p["v_star"], dtype=float)
    frontier = {"theta": p["theta_b"], "v": v_chosen.tolist(), "dir": p["dir"]}
    # 候选与汇合点接近（分支在此汇合）时，把汇合点并入新段，保证轨迹连续
    if float(np.max(np.abs(v_chosen - v_star))) <= 0.5 * solver.scale:
        frontier["prepend"] = [p["theta_star"], p["v_star"]]
    state["frontiers"].insert(0, frontier)
    state["paused"] = None
    state["status"] = "running"
    state["messages"].append(
        f"已选择候选 {index + 1}，从 θ={p['theta_b']:.2f}° 继续追踪")


def _sweep(solver: Solver, mech: Mechanism, state: dict, frontier: dict):
    dirn = frontier["dir"]
    theta = float(frontier["theta"])
    v = np.asarray(frontier["v"], dtype=float)
    samples = []
    if frontier.get("prepend"):
        samples.append([float(frontier["prepend"][0]), list(frontier["prepend"][1])])
    samples.append([theta, v.tolist()])
    seg = {"dir": dirn, "samples": samples, "end_reason": None, "boundary_theta": None}
    state["segments"].append(seg)

    theta_end = mech.theta_end if dirn > 0 else mech.theta_start
    s_cur = solver.det_sign(v, theta)
    step = 2.0
    prev_tv = None  # 上一个采样点 (theta, v)，用于外推种子

    for _ in range(MAX_SWEEP_STEPS):
        t_next = theta + dirn * step
        reached_end = (t_next - theta_end) * dirn >= -1e-12
        if reached_end:
            t_next = theta_end
        v_next = solver.solve(v, t_next)

        singular = False
        s_next = 0.0
        jump = 0.0
        if v_next is None:
            singular = True
        else:
            s_next = solver.det_sign(v_next, t_next)
            jump = float(np.max(np.abs(v_next - v))) / solver.scale
            if jump > 0.5:  # 步内跳变过大：先缩步，仍过大则按奇异处理
                if step > 0.1:
                    step = max(step * 0.5, 0.1)
                    continue
                singular = True
            elif (s_cur != 0.0 and s_next != 0.0
                    and math.copysign(1.0, s_next) != math.copysign(1.0, s_cur)):
                singular = True  # det 变号：跨过奇异汇合点

        if singular:
            theta_star, v_star = _bisect_boundary(solver, theta, v, s_cur, t_next)
            seg["samples"].append([theta_star, v_star.tolist()])
            seg["boundary_theta"] = theta_star
            tb, found = _find_candidates(solver, mech, theta_star, v_star, dirn, prev_tv)
            if found:
                seg["end_reason"] = "paused"
                state["status"] = "paused"
                state["paused"] = {
                    "theta_star": theta_star, "theta_b": tb, "dir": dirn,
                    "v_star": v_star.tolist(),
                    "candidates": [f.tolist() for f in found],
                }
                state["messages"].append(
                    f"θ={theta_star:.2f}° 处检测到奇异汇合，暂停等待选支"
                    f"（{len(found)} 个候选位形，边界误差≤0.1°）")
            else:
                seg["end_reason"] = "boundary"
                state["messages"].append(
                    f"θ={theta_star:.2f}° 之后无满足杆长约束的位形，"
                    f"角段终止（边界误差≤0.1°）")
            return

        prev_tv = (theta, v)
        theta, v = t_next, v_next
        s_cur = s_next
        seg["samples"].append([theta, v.tolist()])
        if jump > 0.25:
            step = max(step * 0.6, 0.1)
        elif jump < 0.05:
            step = min(step * 1.4, 2.0)
        if reached_end:
            seg["end_reason"] = "range_end"
            seg["boundary_theta"] = theta_end
            state["messages"].append(f"到达请求角度边界 θ={theta_end:.2f}°")
            return

    seg["end_reason"] = "aborted"
    state["messages"].append("内部迭代步数超限，角段中止")


def _bisect_boundary(solver: Solver, theta_good, v_good, sign_good, theta_bad):
    """在可解侧与不可解侧之间二分，把边界定位到 0.1° 以内。"""
    for _ in range(40):
        if abs(theta_bad - theta_good) <= ANGLE_TOL_DEG:
            break
        tm = 0.5 * (theta_good + theta_bad)
        vm = solver.solve(v_good, tm)
        if vm is not None:
            sm = solver.det_sign(vm, tm)
            if sign_good == 0.0 or sm == 0.0 or math.copysign(1.0, sm) == math.copysign(1.0, sign_good):
                theta_good, v_good = tm, vm
                continue
        theta_bad = tm
    return theta_good, v_good


def _reflect(p, p1, p2):
    """点 p 关于直线 p1-p2 的镜像；两点圆交点的另一解即为此镜像。"""
    d = p2 - p1
    n2 = float(d @ d)
    if n2 <= 1e-12:
        return None
    return 2.0 * (p1 + d * (((p - p1) @ d) / n2)) - p


def _candidate_seeds(solver: Solver, theta_star, v_star, theta_b, prev_tv):
    """构造多种子：外推、原构型、单关节反射、组合反射、确定性抖动。"""
    seeds = []
    if prev_tv is not None:
        t0, v0 = prev_tv
        dt = theta_star - t0
        if abs(dt) > 1e-12:
            w = (theta_b - theta_star) / dt
            seeds.append(v_star + (v_star - v0) * w)
    seeds.append(v_star.copy())

    pos = solver.positions(v_star, theta_star)
    adj: dict = {}
    for lid in solver.eq_links:
        l = solver.mech.links[lid]
        adj.setdefault(l["a"], []).append(l["b"])
        adj.setdefault(l["b"], []).append(l["a"])

    def set_seed_joint(seed, j, p):
        i = solver.index[j]
        seed[i], seed[i + 1] = p[0], p[1]

    # 单关节反射：仅翻转一个关节（覆盖仅子链换支的情形）
    for j in solver.free_ids:
        nb = adj.get(j, [])
        if len(nb) >= 2:
            refl = _reflect(pos[j], pos[nb[0]], pos[nb[1]])
            if refl is not None:
                seed = v_star.copy()
                set_seed_joint(seed, j, refl)
                seeds.append(seed)

    # 组合反射：按自由关节顺序依次镜像（邻居用已更新坐标），
    # 覆盖主链与其上二杆组需要同时换支的情形
    comp = v_star.copy()
    pos_seed = {j: pos[j].copy() for j in pos}
    changed = False
    for j in solver.free_ids:
        nb = adj.get(j, [])
        if len(nb) >= 2:
            refl = _reflect(pos_seed[j], pos_seed[nb[0]], pos_seed[nb[1]])
            if refl is not None:
                pos_seed[j] = refl
                set_seed_joint(comp, j, refl)
                changed = True
    if changed:
        seeds.append(comp)

    rng = np.random.default_rng(20241005)  # 固定种子，结果可复现
    h = 0.25 * solver.scale
    for _ in range(6):
        seeds.append(v_star + rng.uniform(-h, h, size=v_star.shape))
    return seeds


def _find_candidates(solver: Solver, mech: Mechanism, theta_star, v_star, dirn, prev_tv):
    """在奇异点之外的若干探测角处搜索后续候选位形（去重）。"""
    lo, hi = mech.theta_start, mech.theta_end
    for off in PROBE_OFFSETS_DEG:
        tb = theta_star + dirn * off
        if tb < lo - 1e-9 or tb > hi + 1e-9:
            continue
        found: list = []
        for seed in _candidate_seeds(solver, theta_star, v_star, tb, prev_tv):
            sol = solver.solve(seed, tb)
            if sol is None:
                continue
            if all(float(np.max(np.abs(sol - f))) > 1e-3 * solver.scale for f in found):
                found.append(sol)
            if len(found) >= MAX_CANDIDATES:
                break
        if found:
            return tb, found
    return None, []


# ---------------------------------------------------------------------------
# 序列化：把内部自由变量展开为全部动铰点坐标，供前端直接绘制
# ---------------------------------------------------------------------------

def serialize_state(mech: Mechanism, state: dict) -> dict:
    solver = Solver(mech)
    moving = list(mech.moving_ids)

    def expand(theta, v):
        pos = solver.positions(np.asarray(v, dtype=float), theta)
        out = []
        for j in moving:
            out.append(round(float(pos[j][0]), 6))
            out.append(round(float(pos[j][1]), 6))
        return out

    segs = []
    for s in state["segments"]:
        segs.append({
            "dir": s["dir"],
            "end_reason": s["end_reason"],
            "boundary_theta": s["boundary_theta"],
            "samples": [[round(float(t), 6), expand(t, v)] for t, v in s["samples"]],
        })
    paused = None
    if state.get("paused"):
        p = state["paused"]
        paused = {
            "theta_star": p["theta_star"],
            "theta_b": p["theta_b"],
            "candidates": [{"theta": p["theta_b"], "positions": expand(p["theta_b"], c)}
                           for c in p["candidates"]],
        }
    return {
        "status": state["status"],
        "messages": state["messages"],
        "moving_joint_ids": moving,
        "theta_init": mech.theta_init,
        "theta_start": mech.theta_start,
        "theta_end": mech.theta_end,
        "segments": segs,
        "paused": paused,
    }
