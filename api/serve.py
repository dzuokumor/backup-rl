import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import OBS_DIM, NUM_ACTIONS, get_action_name

app = FastAPI(title="RL-for-PGM Grid Agent API")

_model = None
_model_type = None


class ObservationRequest(BaseModel):
    observation: list[float]


class ActionResponse(BaseModel):
    action: int
    action_name: str
    action_category: str
    confidence: float


def _load_model():
    global _model, _model_type
    if _model is not None:
        return

    from stable_baselines3 import DQN, PPO, A2C

    model_dirs = [ROOT / "models" / "dqn", ROOT / "models" / "pg"]
    best_path = None
    best_reward = -float("inf")

    for d in model_dirs:
        if not d.exists():
            continue
        for f in d.glob("*_final.zip"):
            meta = f.with_suffix(".meta.json")
            if meta.exists():
                import json
                with open(meta) as fh:
                    data = json.load(fh)
                if data.get("mean_reward", -float("inf")) > best_reward:
                    best_reward = data["mean_reward"]
                    best_path = f

    if best_path is None:
        for d in model_dirs:
            if not d.exists():
                continue
            files = sorted(d.glob("*_final.zip"))
            if files:
                best_path = files[0]
                break

    if best_path is None:
        raise RuntimeError("no trained model found in models/dqn or models/pg")

    path_str = str(best_path).lower()
    if "dqn" in path_str:
        _model = DQN.load(best_path)
        _model_type = "dqn"
    elif "ppo" in path_str:
        _model = PPO.load(best_path)
        _model_type = "ppo"
    elif "a2c" in path_str:
        _model = A2C.load(best_path)
        _model_type = "a2c"
    else:
        for name, cls in [("dqn", DQN), ("ppo", PPO), ("a2c", A2C)]:
            try:
                _model = cls.load(best_path)
                _model_type = name
                break
            except Exception:
                continue

    if _model is None:
        raise RuntimeError(f"failed to load model from {best_path}")

    print(f"loaded {_model_type} model from {best_path}")


@app.on_event("startup")
def startup():
    try:
        _load_model()
    except RuntimeError as e:
        print(f"warning: {e}")
        print("api will return 503 until a model is available")


@app.post("/predict", response_model=ActionResponse)
def predict(req: ObservationRequest):
    if _model is None:
        raise HTTPException(503, "no model loaded")

    if len(req.observation) != OBS_DIM:
        raise HTTPException(400, f"observation must have {OBS_DIM} values, got {len(req.observation)}")

    obs = np.array(req.observation, dtype=np.float32)
    action, _ = _model.predict(obs, deterministic=True)
    action = int(action)

    action_name, action_category = get_action_name(action)

    if hasattr(_model, "policy") and hasattr(_model.policy, "get_distribution"):
        import torch
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(_model.device)
        dist = _model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        confidence = float(probs[action])
    else:
        confidence = 1.0

    return ActionResponse(
        action=action,
        action_name=action_name,
        action_category=action_category,
        confidence=round(confidence, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_type": _model_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=False)
