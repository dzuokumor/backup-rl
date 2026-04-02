import sys
import io
import time
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import PowerGridEnv, get_action_name
from environment.rendering import GridRenderer, WINDOW_WIDTH, WINDOW_HEIGHT

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(LOG_DIR / "gui_web.log", "w")

def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {msg}"
    print(line)
    _log_file.write(line + "\n")
    _log_file.flush()


def load_best_model():
    from stable_baselines3 import DQN, PPO, A2C

    best_path = None
    best_reward = -float("inf")

    for d in [ROOT / "models" / "pg", ROOT / "models" / "dqn"]:
        if not d.exists():
            continue
        for f in list(d.glob("1m_*_best.zip")) + list(d.glob("*_final.zip")):
            meta = f.with_suffix(".meta.json")
            if not meta.exists():
                meta = f.with_name(f.stem + ".meta.json")
            if meta.exists():
                with open(meta) as fh:
                    data = json.load(fh)
                reward = data.get("final_100_mean", data.get("mean_reward", -float("inf")))
                if reward > best_reward:
                    best_reward = reward
                    best_path = f

    if best_path is None:
        for d in [ROOT / "models" / "pg", ROOT / "models" / "dqn"]:
            if d.exists():
                files = sorted(d.glob("*.zip"))
                if files:
                    best_path = files[0]
                    break

    if best_path is None:
        return None, "none"

    path_str = str(best_path).lower()
    for name, cls in [("ppo", PPO), ("dqn", DQN), ("a2c", A2C)]:
        if name in path_str:
            return cls.load(best_path), name

    for name, cls in [("ppo", PPO), ("dqn", DQN), ("a2c", A2C)]:
        try:
            return cls.load(best_path), name
        except Exception:
            continue

    return None, "none"


# ── simulation state ─────────────────────────────────────────────────

class Simulation:
    def __init__(self):
        self.lock = threading.Lock()
        self.env = PowerGridEnv(
            fault_probability=0.001,
            max_disconnected_lines=10,
            fault_recovery_steps=20,
            survival_bonus=0.2,
        )
        self.obs, self.info = self.env.reset()
        self.model = None
        self.model_name = "none"
        self.episode = 1
        self.step_count = 0
        self.total_reward = 0.0
        self.running = False
        self.speed_ms = 300
        self.done = False
        self.done_status = ""
        self.latest_frame = None
        self.renderer = None
        import queue
        self._cmd_queue = queue.Queue()

    def init_renderer(self):
        self.renderer = GridRenderer(self.env)
        self._capture_frame()
        log_msg(f"renderer initialized, frame size={len(self.latest_frame) if self.latest_frame else 0} bytes")

    def _capture_frame(self):
        if self.renderer is None:
            return
        try:
            frame = self.renderer.render_to_array(self.env)
            if frame is None or frame.size == 0:
                log_msg("render_to_array returned empty frame")
                return
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            self.latest_frame = buf.getvalue()
        except Exception as e:
            log_msg(f"capture_frame error: {e}")

    def step(self):
        with self.lock:
            if self.done:
                return
            if self.model is not None:
                action, _ = self.model.predict(self.obs, deterministic=True)
                action = int(action)
            else:
                action = self.env.action_space.sample()

            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            self.step_count += 1
            self.total_reward += reward
            self._capture_frame()

            if terminated or truncated:
                self.done = True
                self.done_status = "blackout" if terminated else "survived"
                self.running = False

    def reset(self):
        with self.lock:
            self.obs, self.info = self.env.reset()
            self.episode += 1
            self.step_count = 0
            self.total_reward = 0.0
            self.done = False
            self.done_status = ""
            self.running = False
            self._capture_frame()

    def get_state(self):
        with self.lock:
            info = self.info
            return {
                "episode": int(self.episode),
                "step": int(self.step_count),
                "reward": round(float(self.total_reward), 2),
                "running": self.running,
                "done": self.done,
                "done_status": self.done_status,
                "speed_ms": int(self.speed_ms),
                "model_name": self.model_name,
                "load_served_pct": round(float(info.get("load_served_pct", 100.0)), 1),
                "total_demand": round(float(info.get("total_demand", 0)), 3),
                "total_generation": round(float(info.get("total_generation", 0)), 3),
                "num_overloaded": int(info.get("num_overloaded", 0)),
                "num_disconnected": int(info.get("num_disconnected", 0)),
                "action_name": str(info.get("action_name", "--")),
                "action_category": str(info.get("action_category", "--")),
                "cascade_events": [str(e) for e in info.get("cascade_events", [])],
                "time_of_day": round(float(info.get("time_of_day", 0)), 1),
            }


sim = Simulation()


def run_main_loop():
    """Main thread loop: handles pygame events, steps sim, renders frames."""
    import pygame

    last_step_time = 0
    loop_count = 0
    last_log_time = time.time()
    steps_since_log = 0

    log_msg("main loop started")

    while True:
        loop_count += 1

        # pump pygame events so macOS doesn't kill the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                log_msg("pygame QUIT")
                return

        # auto-step if running
        now = time.time()
        if sim.running and not sim.done:
            if (now - last_step_time) * 1000 >= sim.speed_ms:
                with sim.lock:
                    if sim.model is not None:
                        action, _ = sim.model.predict(sim.obs, deterministic=True)
                        action = int(action)
                    else:
                        action = sim.env.action_space.sample()

                    sim.obs, reward, terminated, truncated, sim.info = sim.env.step(action)
                    sim.step_count += 1
                    sim.total_reward += reward

                    cascades = sim.info.get("cascade_events", [])
                    if cascades:
                        gens = [int(sim.env.gen_available[i]) for i in range(5)]
                        log_msg(f"step {sim.step_count} CASCADE: {cascades} | gens={gens}")

                    sim._capture_frame()

                    if terminated or truncated:
                        sim.done = True
                        sim.done_status = "blackout" if terminated else "survived"
                        sim.running = False
                        log_msg(f"ep {sim.episode}: {sim.done_status} at step {sim.step_count} reward={sim.total_reward:.2f}")

                last_step_time = now
                steps_since_log += 1

        # process pending commands from web
        while not sim._cmd_queue.empty():
            try:
                cmd = sim._cmd_queue.get_nowait()
                log_msg(f"cmd: {cmd}")
                if cmd == "step":
                    sim.step()
                elif cmd == "reset":
                    sim.reset()
            except Exception:
                break

        # keep frame fresh even when idle (for animations like stars, water)
        sim._capture_frame()

        # periodic log
        if now - last_log_time > 3.0:
            frame_size = len(sim.latest_frame) if sim.latest_frame else 0
            log_msg(f"loop #{loop_count} | steps_done={steps_since_log} | running={sim.running} | frame={frame_size}b | step={sim.step_count}")
            last_log_time = now
            steps_since_log = 0

        time.sleep(0.03)


# ── fastapi ──────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response

app = FastAPI(title="RL-for-PGM GUI")


@app.on_event("startup")
def startup():
    log_msg("web server started")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/state")
def get_state():
    return JSONResponse(sim.get_state())


@app.post("/play")
def play():
    if sim.done:
        sim._cmd_queue.put("reset")
    sim.running = True
    return {"ok": True}


@app.post("/pause")
def pause():
    sim.running = False
    return {"ok": True}


@app.post("/step")
def step():
    sim.running = False
    sim._cmd_queue.put("step")
    return {"ok": True}


@app.post("/reset")
def reset():
    sim.running = False
    sim._cmd_queue.put("reset")
    return {"ok": True}


@app.post("/speed")
def set_speed(ms: int = 100):
    sim.speed_ms = max(10, min(1000, ms))
    return {"speed_ms": sim.speed_ms}


@app.get("/frame")
def get_frame():
    if sim.latest_frame is None:
        return Response(status_code=204)
    return Response(content=sim.latest_frame, media_type="image/jpeg")


def mjpeg_gen():
    while True:
        if sim.latest_frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + sim.latest_frame
                + b"\r\n"
            )
        time.sleep(0.03)


@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── html ─────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RL-for-PGM</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #111;
    color: #ccc;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    display: flex;
    height: 100vh;
    overflow: hidden;
}
#viewer {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0a0a0a;
    min-width: 0;
}
#viewer img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
#panel {
    width: 300px;
    min-width: 300px;
    background: #161616;
    border-left: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}
.section {
    padding: 12px 16px;
    border-bottom: 1px solid #222;
}
.section-label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.section-value {
    font-size: 13px;
    color: #ddd;
}
.big-value {
    font-size: 22px;
    font-weight: bold;
    color: #fff;
}
.controls {
    padding: 14px 16px;
    border-bottom: 1px solid #222;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
button {
    background: #222;
    color: #ccc;
    border: 1px solid #333;
    padding: 6px 16px;
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    border-radius: 2px;
}
button:hover { background: #2a2a2a; color: #fff; }
button:active { background: #333; }
button.active { background: #333; color: #fff; border-color: #555; }
.speed-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-bottom: 1px solid #222;
}
.speed-row label { color: #555; font-size: 11px; }
.speed-row input[type=range] {
    flex: 1;
    accent-color: #666;
}
.speed-row span { color: #888; font-size: 11px; min-width: 40px; }
.bar-bg {
    background: #222;
    height: 4px;
    margin-top: 6px;
    border-radius: 2px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.15s;
}
.cascade-item {
    font-size: 11px;
    color: #a55;
    padding: 1px 0;
}
#status-line {
    font-size: 11px;
    color: #555;
    padding: 8px 16px;
    border-bottom: 1px solid #222;
}
.header {
    padding: 14px 16px;
    border-bottom: 1px solid #2a2a2a;
}
.header h1 {
    font-size: 15px;
    font-weight: bold;
    color: #eee;
    margin-bottom: 2px;
}
.header .sub {
    font-size: 10px;
    color: #555;
}
</style>
</head>
<body>

<div id="viewer">
    <img id="frame" alt="renderer">
</div>

<div id="panel">
    <div class="header">
        <h1>RL-for-PGM</h1>
        <div class="sub" id="model-name">model: --</div>
    </div>

    <div class="controls">
        <button id="btn-play" onclick="doPlay()">play</button>
        <button id="btn-pause" onclick="doPause()">pause</button>
        <button onclick="doStep()">step</button>
        <button onclick="doReset()">reset</button>
    </div>

    <div class="speed-row">
        <label>speed</label>
        <input type="range" id="speed" min="50" max="1000" value="300" oninput="setSpeed(this.value)">
        <span id="speed-val">300ms</span>
    </div>

    <div id="status-line">ready</div>

    <div class="section">
        <div class="section-label">episode</div>
        <div class="big-value" id="v-episode">1</div>
    </div>

    <div class="section">
        <div class="section-label">step</div>
        <div class="section-value" id="v-step">0 / 1000</div>
        <div class="bar-bg"><div class="bar-fill" id="bar-step" style="width:0%;background:#666"></div></div>
    </div>

    <div class="section">
        <div class="section-label">reward</div>
        <div class="big-value" id="v-reward">0.00</div>
    </div>

    <div class="section">
        <div class="section-label">load served</div>
        <div class="section-value" id="v-served">--</div>
        <div class="bar-bg"><div class="bar-fill" id="bar-served" style="width:100%;background:#ccc"></div></div>
    </div>

    <div class="section">
        <div class="section-label">demand / generation</div>
        <div class="section-value" id="v-demand">-- / --</div>
    </div>

    <div class="section">
        <div class="section-label">grid</div>
        <div class="section-value" id="v-overloaded">overloaded: 0</div>
        <div class="section-value" id="v-disconnected">disconnected: 0</div>
    </div>

    <div class="section">
        <div class="section-label">action</div>
        <div class="section-value" id="v-action">--</div>
        <div class="section-value" id="v-action-cat" style="color:#555">--</div>
    </div>

    <div class="section">
        <div class="section-label">cascades</div>
        <div id="v-cascades"><span style="color:#555">none</span></div>
    </div>
</div>

<script>
function post(url, body) { fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: body ? JSON.stringify(body) : undefined}); }
function doPlay() { post('/play'); }
function doPause() { post('/pause'); }
function doStep() { post('/step'); }
function doReset() { post('/reset'); }
function setSpeed(v) {
    document.getElementById('speed-val').textContent = v + 'ms';
    post('/speed?ms=' + v);
}

function poll() {
    fetch('/state').then(r => r.json()).then(s => {
        document.getElementById('model-name').textContent = 'model: ' + s.model_name.toUpperCase();
        document.getElementById('v-episode').textContent = s.episode;
        document.getElementById('v-step').textContent = s.step + ' / 1000';
        document.getElementById('bar-step').style.width = (s.step / 10) + '%';
        document.getElementById('v-reward').textContent = s.reward.toFixed(2);

        let served = s.load_served_pct;
        document.getElementById('v-served').textContent = served.toFixed(1) + '%';
        let barServed = document.getElementById('bar-served');
        barServed.style.width = served + '%';
        barServed.style.background = served > 90 ? '#ccc' : served > 70 ? '#cc3' : '#c33';

        document.getElementById('v-demand').textContent = s.total_demand.toFixed(3) + ' / ' + s.total_generation.toFixed(3) + ' pu';
        document.getElementById('v-overloaded').textContent = 'overloaded: ' + s.num_overloaded;
        document.getElementById('v-overloaded').style.color = s.num_overloaded > 0 ? '#c55' : '#888';
        document.getElementById('v-disconnected').textContent = 'disconnected: ' + s.num_disconnected;
        document.getElementById('v-disconnected').style.color = s.num_disconnected > 0 ? '#c55' : '#888';

        document.getElementById('v-action').textContent = s.action_name;
        document.getElementById('v-action-cat').textContent = s.action_category;

        let cascDiv = document.getElementById('v-cascades');
        if (s.cascade_events.length > 0) {
            cascDiv.innerHTML = s.cascade_events.slice(0, 4).map(e => '<div class="cascade-item">' + e + '</div>').join('');
        } else {
            cascDiv.innerHTML = '<span style="color:#555">none</span>';
        }

        let btnPlay = document.getElementById('btn-play');
        let btnPause = document.getElementById('btn-pause');
        btnPlay.className = s.running ? '' : 'active';
        btnPause.className = s.running ? 'active' : '';

        let status = '';
        if (s.done) status = 'ep ' + s.episode + ': ' + s.done_status + ' | ' + s.reward.toFixed(1);
        else if (s.running) status = 'ep ' + s.episode + ' running';
        else status = 'ready';
        document.getElementById('status-line').textContent = status;
    }).catch(() => {});
}
setInterval(poll, 150);

// frame polling — cache-bust with timestamp
function refreshFrame() {
    let img = document.getElementById('frame');
    let newImg = new Image();
    newImg.onload = function() {
        img.src = newImg.src;
        setTimeout(refreshFrame, 50);
    };
    newImg.onerror = function() {
        setTimeout(refreshFrame, 200);
    };
    newImg.src = '/frame?' + Date.now();
}
refreshFrame();
</script>
</body>
</html>"""


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # pygame + OpenGL must init in main thread on macOS
    log_msg("loading model...")
    sim.model, sim.model_name = load_best_model()
    log_msg(f"model loaded: {sim.model_name}")

    log_msg("initializing renderer (pygame window will open and minimize)...")
    sim.init_renderer()
    log_msg("renderer ready")

    # run uvicorn in a background thread
    log_msg("starting server on http://localhost:8000")
    print("open http://localhost:8000 in your browser")
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "warning"},
        daemon=True,
    )
    server_thread.start()

    # main thread: pygame event loop + simulation stepping + frame capture
    try:
        run_main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        import pygame
        pygame.quit()
