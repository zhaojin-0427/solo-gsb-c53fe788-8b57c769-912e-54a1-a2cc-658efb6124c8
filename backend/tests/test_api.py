"""API 端到端测试（FastAPI TestClient）。"""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_solver import crank_rocker


@pytest.fixture()
def client():
    with TestClient(app) as c:  # 触发 startup，初始化数据库表
        yield c


def test_analyze_and_session_flow(client):
    payload = crank_rocker()
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "done"
    assert data["session_id"] > 0
    assert data["segments"]
    assert data["moving_joint_ids"] == [3, 4]
    # 会话状态可回读
    r2 = client.get(f"/api/sessions/{data['session_id']}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "done"


def test_analyze_rejects_invalid(client):
    bad = crank_rocker()
    bad["links"][0]["length"] = 0  # 杆长非正
    r = client.post("/api/analyze", json=bad)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "non_positive_length"

    bad2 = crank_rocker()
    bad2["links"][1]["length"] = 100.0  # 不闭合
    r2 = client.post("/api/analyze", json=bad2)
    assert r2.status_code == 422
    assert r2.json()["detail"]["code"] == "not_closed"


def test_choose_requires_paused_session(client):
    r = client.post("/api/analyze", json=crank_rocker())
    sid = r.json()["session_id"]
    r2 = client.post(f"/api/sessions/{sid}/choose", json={"candidate_index": 0})
    assert r2.status_code == 422  # 未暂停，不能选支
    r3 = client.post("/api/sessions/99999/choose", json={"candidate_index": 0})
    assert r3.status_code == 404
