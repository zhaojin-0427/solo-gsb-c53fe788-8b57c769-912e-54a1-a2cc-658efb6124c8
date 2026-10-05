'use strict';
/* 平面连杆装配分支追踪分析台 —— 前端（原生 JS/Canvas） */

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const SNAP = 5;          // 网格吸附（px）
const JOINT_R = 7;       // 关节绘制半径
const PICK_R = 12;       // 拾取容差

const $ = id => document.getElementById(id);

const S = {
  mode: 'select',
  joints: [],            // {id, x, y, fixed}
  links: [],             // {id, a, b, length}
  driverLinkId: null,
  nextJ: 1, nextL: 1,
  selected: null,        // {kind:'joint'|'link', id}
  pendingLinkFrom: null,
  dragJoint: null,
  mouse: { x: 0, y: 0 },
  // 分析结果
  sessionId: null,
  segments: [],
  movingIds: [],
  paused: null,
  messages: [],
  analysisStatus: null,
  // 播放
  playing: false,
  playSeg: 0,
  playTheta: null,
  notice: '',
  noticeUntil: 0,
};

const TRAJ_COLORS = ['#d1495b', '#2e86ab', '#7fb069', '#b565d8', '#e07b39'];
const CAND_COLORS = ['#e0a458', '#5bc0be', '#c084fc', '#f472b6', '#a3e635', '#fbbf24'];

/* ---------------- 模型操作 ---------------- */

function jointById(id) { return S.joints.find(j => j.id === id); }
function linkById(id) { return S.links.find(l => l.id === id); }

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function addJoint(x, y, fixed) {
  const j = { id: S.nextJ++, x, y, fixed };
  S.joints.push(j);
  markDirty();
  return j;
}

function addLink(a, b) {
  if (a === b) return null;
  if (S.links.some(l => (l.a === a && l.b === b) || (l.a === b && l.b === a))) return null;
  const ja = jointById(a), jb = jointById(b);
  const l = { id: S.nextL++, a, b, length: Math.round(dist(ja, jb) * 1000) / 1000 };
  S.links.push(l);
  markDirty();
  return l;
}

function deleteJoint(id) {
  S.joints = S.joints.filter(j => j.id !== id);
  const removed = S.links.filter(l => l.a === id || l.b === id);
  S.links = S.links.filter(l => l.a !== id && l.b !== id);
  if (removed.some(l => l.id === S.driverLinkId)) S.driverLinkId = null;
  markDirty();
}

function deleteLink(id) {
  S.links = S.links.filter(l => l.id !== id);
  if (S.driverLinkId === id) S.driverLinkId = null;
  markDirty();
}

function isDriverEligible(l) {
  const ja = jointById(l.a), jb = jointById(l.b);
  return ja && jb && ja.fixed !== jb.fixed;
}

/* 修改机构后：旧轨迹立即失效 */
function markDirty() {
  S.sessionId = null;
  S.segments = [];
  S.movingIds = [];
  S.paused = null;
  S.messages = [];
  S.analysisStatus = null;
  S.playing = false;
  S.playTheta = null;
  $('btnPlay').disabled = true;
  $('btnPlay').textContent = '播放';
  setStatus('机构已修改，旧轨迹已失效，请重新分析。');
  renderPanel();
  renderResults();
}

/* ---------------- 坐标变换 ---------------- */
const cx = x => x;
const cy = y => H - y;
function toMath(e) {
  const r = cv.getBoundingClientRect();
  return { x: e.clientX - r.left, y: H - (e.clientY - r.top) };
}
const snap = v => Math.round(v / SNAP) * SNAP;

/* ---------------- 拾取 ---------------- */
function findJointNear(p) {
  let best = null, bd = PICK_R;
  for (const j of S.joints) {
    const d = Math.hypot(j.x - p.x, j.y - p.y);
    if (d < bd) { bd = d; best = j; }
  }
  return best;
}
function findLinkNear(p) {
  let best = null, bd = 8;
  for (const l of S.links) {
    const a = jointById(l.a), b = jointById(l.b);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const n2 = dx * dx + dy * dy;
    if (n2 < 1e-9) continue;
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / n2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
    if (d < bd) { bd = d; best = l; }
  }
  return best;
}

/* ---------------- 绘制 ---------------- */

function drawGrid() {
  ctx.strokeStyle = '#ececec';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= W; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = 0; y <= H; y += 40) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke();
}

function drawLinkLine(pa, pb, color, width, dash) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(cx(pa.x), cy(pa.y));
  ctx.lineTo(cx(pb.x), cy(pb.y));
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawJointShape(p, fixed, color, highlight) {
  const X = cx(p.x), Y = cy(p.y);
  if (fixed) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(X, Y - JOINT_R - 2);
    ctx.lineTo(X - JOINT_R - 2, Y + JOINT_R);
    ctx.lineTo(X + JOINT_R + 2, Y + JOINT_R);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#666';
    ctx.beginPath();
    ctx.moveTo(X - JOINT_R - 6, Y + JOINT_R + 3);
    ctx.lineTo(X + JOINT_R + 6, Y + JOINT_R + 3);
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(X, Y, JOINT_R - 2, 0, Math.PI * 2);
    ctx.fill();
  }
  if (highlight) {
    ctx.strokeStyle = '#2d6cdf';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(X, Y, JOINT_R + 3, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function positionsFromModel() {
  const pos = {};
  for (const j of S.joints) pos[j.id] = { x: j.x, y: j.y };
  return pos;
}

/* 播放时按角段内插值（绝不在角段之间插值） */
function interpolatedPositions() {
  if (!S.segments.length || S.playTheta == null) return null;
  const seg = S.segments[S.playSeg];
  if (!seg) return null;
  const samples = seg.samples;
  const t = S.playTheta;
  let k = 0;
  while (k < samples.length - 2 &&
         (samples[k + 1][0] - t) * seg.dir < 0) k++;
  const [t0, c0] = samples[k];
  const [t1, c1] = samples[k + 1];
  const u = t1 === t0 ? 0 : Math.max(0, Math.min(1, (t - t0) / (t1 - t0)));
  const pos = positionsFromModel();
  S.movingIds.forEach((jid, i) => {
    pos[jid] = {
      x: c0[2 * i] + (c1[2 * i] - c0[2 * i]) * u,
      y: c0[2 * i + 1] + (c1[2 * i + 1] - c0[2 * i + 1]) * u,
    };
  });
  return pos;
}

function drawMechanism(pos, opts = {}) {
  const ghost = opts.ghost || false;
  for (const l of S.links) {
    const a = pos[l.a], b = pos[l.b];
    if (!a || !b) continue;
    const isDrv = l.id === S.driverLinkId;
    drawLinkLine(a, b,
      ghost ? 'rgba(120,120,120,0.45)' : (isDrv ? '#e07b39' : '#4a515b'),
      isDrv && !ghost ? 4 : 3,
      ghost ? [5, 4] : []);
  }
  for (const j of S.joints) {
    const p = pos[j.id];
    if (!p) continue;
    drawJointShape(p, j.fixed,
      ghost ? 'rgba(120,120,120,0.5)' : (j.fixed ? '#333' : '#2d6cdf'),
      !ghost && S.selected && S.selected.kind === 'joint' && S.selected.id === j.id);
  }
  // 驱动杆旋转箭头
  if (!ghost && S.driverLinkId != null) {
    const l = linkById(S.driverLinkId);
    if (l) {
      const piv = jointById(l.a).fixed ? pos[l.a] : pos[l.b];
      const end = jointById(l.a).fixed ? pos[l.b] : pos[l.a];
      const ang = Math.atan2(end.y - piv.y, end.x - piv.x);
      const X = cx(piv.x), Y = cy(piv.y);
      ctx.strokeStyle = '#e07b39';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(X, Y, 26, -(ang + 0.9), -(ang - 0.6), true);
      ctx.stroke();
      const ea = -(ang - 0.6);
      ctx.fillStyle = '#e07b39';
      ctx.beginPath();
      ctx.arc(X + 26 * Math.cos(ea), Y + 26 * Math.sin(ea), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawTrajectories() {
  if (!S.segments.length) return;
  S.movingIds.forEach((jid, ji) => {
    ctx.strokeStyle = TRAJ_COLORS[ji % TRAJ_COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    for (const seg of S.segments) {
      ctx.beginPath();
      seg.samples.forEach((s, i) => {
        const X = cx(s[1][2 * ji]), Y = cy(s[1][2 * ji + 1]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  });
  // 角段边界标记（可达角段端点）
  const drv = S.driverLinkId != null ? linkById(S.driverLinkId) : null;
  if (drv) {
    const pivJ = jointById(drv.a).fixed ? jointById(drv.a) : jointById(drv.b);
    for (const seg of S.segments) {
      if (seg.end_reason !== 'boundary' && seg.end_reason !== 'paused') continue;
      const th = seg.boundary_theta * Math.PI / 180;
      const p = { x: pivJ.x + drv.length * Math.cos(th), y: pivJ.y + drv.length * Math.sin(th) };
      const X = cx(p.x), Y = cy(p.y);
      ctx.strokeStyle = '#c0392b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X - 6, Y - 6); ctx.lineTo(X + 6, Y + 6);
      ctx.moveTo(X + 6, Y - 6); ctx.lineTo(X - 6, Y + 6);
      ctx.stroke();
      ctx.fillStyle = '#c0392b';
      ctx.font = '11px sans-serif';
      ctx.fillText(`θ=${seg.boundary_theta.toFixed(1)}°`, X + 9, Y - 6);
    }
  }
}

function drawCandidates() {
  if (!S.paused) return;
  S.paused.candidates.forEach((c, i) => {
    const color = CAND_COLORS[i % CAND_COLORS.length];
    const pos = positionsFromModel();
    S.movingIds.forEach((jid, k) => {
      pos[jid] = { x: c.positions[2 * k], y: c.positions[2 * k + 1] };
    });
    for (const l of S.links) drawLinkLine(pos[l.a], pos[l.b], color, 2, [6, 4]);
    S.movingIds.forEach((jid, k) => {
      const p = pos[jid];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx(p.x), cy(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
    });
    const p0 = pos[S.movingIds[0]];
    if (p0) {
      ctx.fillStyle = color;
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(`候选${i + 1}`, cx(p0.x) + 10, cy(p0.y) - 10);
    }
  });
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  drawGrid();
  drawTrajectories();
  // 静态/播放机构
  const ipos = interpolatedPositions();
  if (ipos) {
    drawMechanism(positionsFromModel(), { ghost: true });
    drawMechanism(ipos);
  } else {
    drawMechanism(positionsFromModel());
  }
  drawCandidates();
  // 待连连杆
  if (S.pendingLinkFrom != null) {
    const a = jointById(S.pendingLinkFrom);
    if (a) drawLinkLine(a, S.mouse, '#2d6cdf', 1.5, [4, 4]);
  }
  // HUD
  ctx.fillStyle = '#333';
  ctx.font = '13px monospace';
  let hud = '';
  if (S.segments.length && S.playTheta != null) {
    hud = `θ = ${S.playTheta.toFixed(1)}°   角段 ${S.playSeg + 1}/${S.segments.length}`;
  }
  if (S.paused) hud += (hud ? '   ' : '') + `已暂停：θ*=${S.paused.theta_star.toFixed(2)}°，请选择候选分支`;
  ctx.fillText(hud, 10, 18);
  if (S.notice && performance.now() < S.noticeUntil) {
    ctx.fillStyle = '#c0392b';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(S.notice, 10, 40);
  }
}

/* ---------------- 面板 ---------------- */

function renderPanel() {
  // 关节表
  const jt = $('jointTable').querySelector('tbody');
  jt.innerHTML = '';
  for (const j of S.joints) {
    const tr = document.createElement('tr');
    if (S.selected && S.selected.kind === 'joint' && S.selected.id === j.id) tr.className = 'selected';
    tr.innerHTML = `<td>${j.id}</td><td>${j.fixed ? '固定' : '动'}</td>`;
    for (const key of ['x', 'y']) {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.type = 'number'; inp.value = Math.round(j[key] * 100) / 100;
      inp.onchange = () => {
        const v = parseFloat(inp.value);
        if (Number.isFinite(v)) { j[key] = v; markDirty(); }
      };
      td.appendChild(inp);
      tr.appendChild(td);
    }
    jt.appendChild(tr);
  }
  // 连杆表
  const lt = $('linkTable').querySelector('tbody');
  lt.innerHTML = '';
  for (const l of S.links) {
    const tr = document.createElement('tr');
    if (S.selected && S.selected.kind === 'link' && S.selected.id === l.id) tr.className = 'selected';
    const tdId = document.createElement('td'); tdId.textContent = l.id;
    const tdA = document.createElement('td'); tdA.textContent = l.a;
    const tdB = document.createElement('td'); tdB.textContent = l.b;
    const tdLen = document.createElement('td');
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = '1'; inp.value = l.length;
    inp.onchange = () => {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v)) { l.length = v; markDirty(); }
    };
    tdLen.appendChild(inp);
    const tdDrv = document.createElement('td');
    if (isDriverEligible(l)) {
      const radio = document.createElement('input');
      radio.type = 'radio'; radio.name = 'drv'; radio.checked = l.id === S.driverLinkId;
      radio.onchange = () => { S.driverLinkId = l.id; markDirty(); };
      tdDrv.appendChild(radio);
    } else {
      tdDrv.textContent = '—';
    }
    tr.append(tdId, tdA, tdB, tdLen, tdDrv);
    lt.appendChild(tr);
  }
  // 驱动杆标签
  $('driverLabel').textContent = S.driverLinkId != null ? `连杆 ${S.driverLinkId}` : '未选择';
}

function renderResults() {
  const el = $('resultInfo');
  if (!S.analysisStatus) { el.textContent = '尚未分析。'; $('candidates').innerHTML = ''; return; }
  const lines = [];
  lines.push(S.analysisStatus === 'done' ? '状态：分析完成'
    : S.analysisStatus === 'paused' ? '状态：暂停（等待选支）' : `状态：${S.analysisStatus}`);
  S.segments.forEach((seg, i) => {
    const ts = seg.samples.map(s => s[0]);
    const reason = { range_end: '到达请求角度边界', boundary: '无后续位形，角段终止',
                     paused: '奇异点暂停', aborted: '内部中止' }[seg.end_reason] || seg.end_reason || '—';
    lines.push(`角段${i + 1}：θ ${Math.min(...ts).toFixed(1)}° ~ ${Math.max(...ts).toFixed(1)}°`
      + `（${seg.samples.length} 采样，${reason}）`);
  });
  for (const m of S.messages) lines.push(m);
  el.innerHTML = lines.map((s, i) =>
    i > 0 && s.startsWith('角段') ? `<span class="seg">${s}</span>` : s
  ).join('\n');
  // 候选按钮
  const ce = $('candidates');
  ce.innerHTML = '';
  if (S.paused) {
    S.paused.candidates.forEach((c, i) => {
      const b = document.createElement('button');
      b.className = 'cand-btn';
      b.textContent = `选择候选 ${i + 1}（θ=${c.theta.toFixed(2)}°）`;
      b.onclick = () => chooseCandidate(i);
      ce.appendChild(b);
    });
  }
}

function setStatus(msg, isError = false, isOk = false) {
  const el = $('statusbar');
  el.textContent = msg;
  el.className = isError ? 'error' : isOk ? 'ok' : '';
}

/* ---------------- 分析 API ---------------- */

function buildPayload() {
  return {
    joints: S.joints.map(j => ({ id: j.id, x: j.x, y: j.y, fixed: j.fixed })),
    links: S.links.map(l => ({ id: l.id, a: l.a, b: l.b, length: l.length })),
    driver: {
      link_id: S.driverLinkId,
      theta_start: parseFloat($('thetaStart').value),
      theta_end: parseFloat($('thetaEnd').value),
    },
  };
}

async function analyze() {
  if (S.driverLinkId == null) { setStatus('请先在连杆表中选择驱动杆。', true); return; }
  setStatus('分析中…');
  let data;
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });
    data = await res.json();
    if (!res.ok) {
      setStatus('分析被拒绝：' + ((data.detail && data.detail.message) || res.status), true);
      return;
    }
  } catch (err) {
    setStatus('网络错误：' + err.message, true);
    return;
  }
  applyResult(data);
}

function applyResult(data) {
  S.sessionId = data.session_id;
  S.segments = data.segments;
  S.movingIds = data.moving_joint_ids;
  S.paused = data.paused;
  S.messages = data.messages;
  S.analysisStatus = data.status;
  S.playSeg = 0;
  S.playTheta = S.segments.length ? S.segments[0].samples[0][0] : null;
  $('btnPlay').disabled = !S.segments.length;
  const last = S.messages[S.messages.length - 1] || '';
  setStatus(S.paused
    ? `奇异汇合：θ*=${S.paused.theta_star.toFixed(2)}°，${S.paused.candidates.length} 个候选位形，请选支。`
    : `分析完成。${last}`, false, !S.paused);
  renderResults();
}

async function chooseCandidate(i) {
  if (S.sessionId == null) return;
  let data;
  try {
    const res = await fetch(`/api/sessions/${S.sessionId}/choose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_index: i }),
    });
    data = await res.json();
    if (!res.ok) {
      setStatus('选支失败：' + ((data.detail && data.detail.message) || data.detail || res.status), true);
      return;
    }
  } catch (err) {
    setStatus('网络错误：' + err.message, true);
    return;
  }
  applyResult(data);
}

/* ---------------- 播放 ---------------- */

function playerStep(dt) {
  if (!S.playing || !S.segments.length) return;
  let seg = S.segments[S.playSeg];
  if (!seg) { S.playing = false; return; }
  const speed = parseFloat($('speed').value); // 度/秒
  let t = S.playTheta + seg.dir * speed * dt;
  const tLast = seg.samples[seg.samples.length - 1][0];
  if ((t - tLast) * seg.dir >= 0) {
    if (S.playSeg + 1 < S.segments.length) {
      S.playSeg++;
      const ns = S.segments[S.playSeg];
      const gap = Math.abs(ns.samples[0][0] - tLast);
      if (gap > 0.2) {
        S.notice = `跳过不可达角段（Δθ≈${gap.toFixed(1)}°）`;
        S.noticeUntil = performance.now() + 1500;
      }
      S.playTheta = ns.samples[0][0];
    } else {
      S.playing = false;
      S.playTheta = tLast;
      $('btnPlay').textContent = '播放';
    }
    return;
  }
  S.playTheta = t;
}

let lastTs = null;
function frame(ts) {
  if (lastTs == null) lastTs = ts;
  const dt = Math.min(0.1, (ts - lastTs) / 1000);
  lastTs = ts;
  playerStep(dt);
  draw();
  requestAnimationFrame(frame);
}

/* ---------------- 交互 ---------------- */

cv.addEventListener('mousedown', e => {
  const p = toMath(e);
  if (S.mode === 'addFixed' || S.mode === 'addMoving') {
    addJoint(snap(p.x), snap(p.y), S.mode === 'addFixed');
    return;
  }
  if (S.mode === 'addLink') {
    const j = findJointNear(p);
    if (j) {
      if (S.pendingLinkFrom == null) S.pendingLinkFrom = j.id;
      else { addLink(S.pendingLinkFrom, j.id); S.pendingLinkFrom = null; }
    } else S.pendingLinkFrom = null;
    return;
  }
  if (S.mode === 'delete') {
    const j = findJointNear(p);
    if (j) { deleteJoint(j.id); return; }
    const l = findLinkNear(p);
    if (l) deleteLink(l.id);
    return;
  }
  // select
  const j = findJointNear(p);
  if (j) {
    S.selected = { kind: 'joint', id: j.id };
    S.dragJoint = j;
    renderPanel();
    return;
  }
  const l = findLinkNear(p);
  S.selected = l ? { kind: 'link', id: l.id } : null;
  renderPanel();
});

cv.addEventListener('mousemove', e => {
  const p = toMath(e);
  S.mouse = p;
  if (S.dragJoint) {
    S.dragJoint.x = snap(p.x);
    S.dragJoint.y = snap(p.y);
    markDirty();
  }
});

window.addEventListener('mouseup', () => { S.dragJoint = null; });

document.querySelectorAll('#toolbar [data-mode]').forEach(btn => {
  btn.onclick = () => {
    S.mode = btn.dataset.mode;
    S.pendingLinkFrom = null;
    document.querySelectorAll('#toolbar [data-mode]').forEach(b => b.classList.toggle('active', b === btn));
  };
});

$('btnSetDriver').onclick = () => {
  if (S.selected && S.selected.kind === 'link') {
    const l = linkById(S.selected.id);
    if (l && isDriverEligible(l)) {
      S.driverLinkId = l.id;
      markDirty();
      setStatus(`已设连杆 ${l.id} 为驱动杆。`);
    } else setStatus('驱动杆必须连接一个固定铰点和一个动铰点。', true);
  } else setStatus('请先在“选择”模式下点击一根连杆。', true);
};

$('btnAnalyze').onclick = analyze;

$('btnPlay').onclick = () => {
  if (!S.segments.length) return;
  S.playing = !S.playing;
  if (S.playing && S.playTheta == null) {
    S.playSeg = 0;
    S.playTheta = S.segments[0].samples[0][0];
  }
  if (S.playing && !S.paused && S.playSeg >= S.segments.length - 1) {
    // 播到末尾后再次播放则从头开始
    const seg = S.segments[S.playSeg];
    const tLast = seg.samples[seg.samples.length - 1][0];
    if (S.playTheta === tLast) { S.playSeg = 0; S.playTheta = S.segments[0].samples[0][0]; }
  }
  $('btnPlay').textContent = S.playing ? '暂停' : '播放';
};

$('thetaStart').onchange = $('thetaEnd').onchange = () => markDirty();

$('btnClear').onclick = () => {
  S.joints = []; S.links = []; S.driverLinkId = null;
  S.nextJ = 1; S.nextL = 1; S.selected = null;
  markDirty();
};

$('btnExample').onclick = () => { loadExample(); };

/* ---------------- 示例机构 ---------------- */

function circleIntersect(p0, r0, p1, r1, upper) {
  const dx = p1.x - p0.x, dy = p1.y - p0.y;
  const d = Math.hypot(dx, dy);
  const a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(r0 * r0 - a * a, 0));
  const xm = p0.x + a * dx / d, ym = p0.y + a * dy / d;
  const s = upper ? 1 : -1;
  return { x: xm - s * h * dy / d, y: ym + s * h * dx / d };
}

function loadExample() {
  S.joints = []; S.links = []; S.nextJ = 1; S.nextL = 1; S.driverLinkId = null;
  const A = addJoint(200, 300, true);
  const D = addJoint(540, 300, true);
  const th = 60 * Math.PI / 180;
  const B = addJoint(200 + 100 * Math.cos(th), 300 + 100 * Math.sin(th), false);
  const Cpos = circleIntersect(B, 260, D, 240, true);
  const C = addJoint(Cpos.x, Cpos.y, false);
  const crank = addLink(A.id, B.id);
  addLink(B.id, C.id);
  addLink(D.id, C.id);
  crank.length = 100;
  S.links[1].length = 260;
  S.links[2].length = 240;
  S.driverLinkId = crank.id;
  $('thetaStart').value = 0;
  $('thetaEnd').value = 360;
  markDirty();
  setStatus('已载入 Grashof 曲柄摇杆示例，点击“分析”开始。');
}

/* ---------------- 启动 ---------------- */
loadExample();
renderPanel();
requestAnimationFrame(frame);
