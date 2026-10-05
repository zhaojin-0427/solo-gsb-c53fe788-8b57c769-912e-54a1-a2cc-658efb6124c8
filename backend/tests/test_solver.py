"""求解器与 API 的集成测试。"""
import math

import pytest

from app.solver import (
    Mechanism,
    MechanismError,
    Solver,
    _find_candidates,
    apply_choice,
    run_session,
    serialize_state,
    start_session,
)


def circle_intersect(p0, r0, p1, r1, upper=True):
    """两圆交点（upper=True 取沿 p0->p1 左侧的点）。"""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d = math.hypot(dx, dy)
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    xm, ym = p0[0] + a * dx / d, p0[1] + a * dy / d
    s = 1.0 if upper else -1.0
    return (xm - s * h * dy / d, ym + s * h * dx / d)


def crank_rocker(theta_deg=60.0, crank=100.0, coupler=260.0, rocker=240.0, ground=340.0):
    """Grashof 曲柄摇杆机构（默认参数可整周回转）。"""
    A = (0.0, 0.0)
    D = (ground, 0.0)
    t = math.radians(theta_deg)
    B = (crank * math.cos(t), crank * math.sin(t))
    C = circle_intersect(B, coupler, D, rocker, upper=True)
    return {
        "joints": [
            {"id": 1, "x": A[0], "y": A[1], "fixed": True},
            {"id": 2, "x": D[0], "y": D[1], "fixed": True},
            {"id": 3, "x": B[0], "y": B[1], "fixed": False},
            {"id": 4, "x": C[0], "y": C[1], "fixed": False},
        ],
        "links": [
            {"id": 11, "a": 1, "b": 3, "length": crank},
            {"id": 12, "a": 3, "b": 4, "length": coupler},
            {"id": 13, "a": 2, "b": 4, "length": rocker},
        ],
        "driver": {"link_id": 11, "theta_start": 0.0, "theta_end": 360.0},
    }


def analyze(data):
    mech = Mechanism(data)
    state = start_session(mech)
    run_session(mech, state)
    return mech, state


def test_full_rotation_covers_range():
    mech, state = analyze(crank_rocker())
    assert state["status"] == "done"
    assert state["segments"], "应有角段"
    for seg in state["segments"]:
        assert seg["end_reason"] == "range_end"
    lo = min(min(s[0] for s in seg["samples"]) for seg in state["segments"])
    hi = max(max(s[0] for s in seg["samples"]) for seg in state["segments"])
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(360.0, abs=1e-6)


def test_solutions_satisfy_constraints():
    mech, state = analyze(crank_rocker())
    solver = Solver(mech)
    for seg in state["segments"]:
        for theta, v in seg["samples"]:
            r = solver.residuals(v, theta)
            assert max(abs(r)) < 1e-8


def test_double_rocker_boundary_accuracy():
    # 非 Grashof：150+340 > 200+200，曲柄无法整周转动。
    # 理论极限：|BD| = 200+200 = 400 时，
    # cos(theta) = (150^2 + 340^2 - 400^2) / (2*150*340) -> theta = ±102.3966°
    data = crank_rocker(theta_deg=60.0, crank=150.0, coupler=200.0, rocker=200.0)
    data["driver"]["theta_start"] = -150.0
    data["driver"]["theta_end"] = 150.0
    mech, state = analyze(data)
    assert state["status"] == "done"
    reasons = {seg["end_reason"] for seg in state["segments"]}
    assert "boundary" in reasons
    bounds = sorted(seg["boundary_theta"] for seg in state["segments"]
                    if seg["end_reason"] == "boundary")
    expect = math.degrees(math.acos((150**2 + 340**2 - 400**2) / (2 * 150 * 340)))
    assert len(bounds) == 2
    assert bounds[0] == pytest.approx(-expect, abs=0.15)
    assert bounds[1] == pytest.approx(+expect, abs=0.15)
    # 角段之外不得有采样（不跨不可达角段）
    for seg in state["segments"]:
        ts = [s[0] for s in seg["samples"]]
        assert min(ts) >= -expect - 0.15
        assert max(ts) <= expect + 0.15


def test_candidates_found_for_fourbar():
    # 四连杆在任意可达角存在开/交叉两个分支：候选搜索应找到 ≥2 个解
    mech = Mechanism(crank_rocker())
    solver = Solver(mech)
    v = solver.solve(solver.initial_v(), mech.theta_init)
    tb, found = _find_candidates(solver, mech, mech.theta_init, v, +1, None)
    assert tb is not None
    assert len(found) >= 2


def test_dyad_mechanism_pauses_and_resumes():
    # 两闭环（四连杆 + 二杆组）：二杆组相切产生折点，主轴另一分支在其外仍有解
    base = crank_rocker(theta_deg=200.0)
    # 在关节4(摇杆端)与固定点5之间加二杆组关节6
    C = (base["joints"][3]["x"], base["joints"][3]["y"])
    G = (170.0, 0.0)
    r1, r2 = 130.0, 70.0
    E = circle_intersect(C, r1, G, r2, upper=True)
    base["joints"].append({"id": 5, "x": G[0], "y": G[1], "fixed": True})
    base["joints"].append({"id": 6, "x": E[0], "y": E[1], "fixed": False})
    base["links"].append({"id": 14, "a": 4, "b": 6, "length": r1})
    base["links"].append({"id": 15, "a": 5, "b": 6, "length": r2})
    mech, state = analyze(base)
    # 必然在奇异汇合处暂停且给出候选；选支后续算，最终完成
    pauses = 0
    for _ in range(12):
        if state["status"] != "paused":
            break
        pauses += 1
        assert state["paused"]["candidates"], "暂停时必须提供候选位形"
        apply_choice(mech, state, 0)
        run_session(mech, state)
    assert pauses >= 1, "二杆组折点处应触发暂停选支"
    assert state["status"] == "done"
    assert state["segments"]
    reasons = {seg["end_reason"] for seg in state["segments"]}
    assert "paused" in reasons and "boundary" in reasons


def test_reject_non_closed():
    data = crank_rocker()
    data["links"][1]["length"] = 150.0  # 与实际距离不符
    with pytest.raises(MechanismError) as e:
        Mechanism(data)
    assert e.value.code == "not_closed"


def test_reject_non_positive_length():
    data = crank_rocker()
    data["links"][0]["length"] = -5.0
    with pytest.raises(MechanismError) as e:
        Mechanism(data)
    assert e.value.code == "non_positive_length"


def test_reject_invalid_reference():
    data = crank_rocker()
    data["links"][0]["b"] = 99
    with pytest.raises(MechanismError) as e:
        Mechanism(data)
    assert e.value.code == "invalid_reference"
    data2 = crank_rocker()
    data2["driver"]["link_id"] = 77
    with pytest.raises(MechanismError) as e2:
        Mechanism(data2)
    assert e2.value.code == "invalid_reference"


def test_reject_not_single_dof():
    data = crank_rocker()
    data["links"] = data["links"][:2]  # 去掉摇杆：2*2-2=2 ≠ 1
    with pytest.raises(MechanismError) as e:
        Mechanism(data)
    assert e.value.code == "not_single_dof"


def test_reject_driver_not_on_ground():
    data = crank_rocker()
    data["driver"]["link_id"] = 12  # 连杆12两端均为动铰点
    with pytest.raises(MechanismError) as e:
        Mechanism(data)
    assert e.value.code == "invalid_driver"


def test_serialize_shape():
    mech, state = analyze(crank_rocker())
    out = serialize_state(mech, state)
    assert out["moving_joint_ids"] == [3, 4]
    assert out["segments"]
    sample = out["segments"][0]["samples"][0]
    theta, coords = sample
    assert len(coords) == 4  # 两个动铰点 × (x, y)
