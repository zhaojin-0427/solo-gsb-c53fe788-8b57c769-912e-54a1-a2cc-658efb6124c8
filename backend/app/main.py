"""FastAPI 入口：机构校验、分析会话、选支续算、静态前端。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .solver import (
    Mechanism,
    MechanismError,
    apply_choice,
    run_session,
    serialize_state,
    start_session,
)

FRONTEND_DIR = os.environ.get(
    "FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend")),
)


@asynccontextmanager
async def lifespan(_app):
    db.init()
    yield


app = FastAPI(title="平面连杆装配分支追踪分析台", lifespan=lifespan)


@app.exception_handler(MechanismError)
def _mechanism_error_handler(_request: Request, exc: MechanismError):
    return JSONResponse(status_code=422,
                        content={"detail": {"code": exc.code, "message": exc.message}})


class ChooseBody(BaseModel):
    candidate_index: int


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
def analyze(payload: dict):
    """校验机构并启动分析；校验失败（不闭合/非单自由度/杆长非正/引用无效）返回 422。"""
    mech = Mechanism(payload)  # 校验失败抛 MechanismError -> 422
    state = start_session(mech)
    run_session(mech, state)
    mid = db.create_mechanism(payload)
    sid = db.create_session(mid, state)
    out = serialize_state(mech, state)
    out["session_id"] = sid
    return out


@app.post("/api/sessions/{sid}/choose")
def choose(sid: int, body: ChooseBody):
    """奇异汇合暂停后，用户选定候选位形续算。"""
    row = db.get_session(sid)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    mech_data, state = row
    mech = Mechanism(mech_data)
    apply_choice(mech, state, body.candidate_index)
    run_session(mech, state)
    db.update_session(sid, state)
    out = serialize_state(mech, state)
    out["session_id"] = sid
    return out


@app.get("/api/sessions/{sid}")
def session_state(sid: int):
    row = db.get_session(sid)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    mech_data, state = row
    mech = Mechanism(mech_data)
    out = serialize_state(mech, state)
    out["session_id"] = sid
    return out


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
