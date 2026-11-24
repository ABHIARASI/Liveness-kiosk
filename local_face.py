"""Offline face engine: runs fully on this laptop with OpenCV (no AWS, no internet).

Models (in models/, downloaded once with  python get_models.py):
  - YuNet  (face_detection_yunet_2023mar.onnx)   finds faces + 5 landmarks (eyes, nose, mouth corners)
  - SFace  (face_recognition_sface_2021dec.onnx) turns a face into a 128-number vector

Offline liveness = challenge-response: the person does 2 RANDOM actions in a random order.
  turn_left / turn_right : the nose moves sideways relative to the eyes. Only a real 3D head does this;
                           a flat photo or screen, even tilted, keeps the same nose/eye ratio.
  smile                  : the mouth gets wider relative to the eye distance.
  move_closer            : the face gets bigger (never used alone: a photo can be moved closer).
Every frame must show exactly one face, and it must stay the SAME face for the whole challenge.

Privacy: only vectors are stored (data/offline_faces.json), saved at liveness-verified enrollment. No photos.

CONTRACT:
    available() -> bool
    save_vector(person_id, image_bytes) -> bool          # called by enrollment
    remove_vector(person_id) -> None
    has_vector(person_id) -> bool
    new_challenge() -> {"id", "actions": [a1, a2], "labels": [str, str], "expires": float}
    verify(challenge_id, frames) -> dict                  # frames: [{"phase": "baseline"|0|1, "image": bytes}]
"""
import json
import os
import random
import secrets
import threading
import time

import numpy as np

import config

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
VECTORS_FILE = os.path.join(config.DATA_DIR, "offline_faces.json")

ACTIONS = {
    "turn_left": "Turn your head to YOUR left",
    "turn_right": "Turn your head to YOUR right",
    "smile": "Smile big",
    "move_closer": "Move closer to the camera",
}
THREE_D_ACTIONS = ["turn_left", "turn_right", "smile"]   # at least one of these every time

# thresholds (tuned for SFace / YuNet)
MATCH_COSINE = 0.45        # SFace: >= this is the same person
REVIEW_COSINE = 0.363      # OpenCV's published SFace threshold; between -> step-up (code)
SAME_FACE_COSINE = 0.30    # every frame must stay the same person as the baseline
TURN_DELTA = 0.12          # change in (nose - eye centre) / eye distance
SMILE_RATIO = 1.10         # mouth width / eye distance grows by >= 10 %
CLOSER_RATIO = 1.20        # face width grows by >= 20 %
FRONTAL_MAX = 0.20         # baseline must be roughly frontal
MIN_FRAMES_PER_ACTION = 2  # action must be seen in at least 2 frames (noise filter)
CHALLENGE_SECONDS = 30

_lock = threading.Lock()
_models = {}
_challenges = {}


def available():
    return os.path.exists(YUNET) and os.path.exists(SFACE)


def _cv2():
    import cv2
    return cv2


def _detector(w, h):
    cv2 = _cv2()
    if "det" not in _models:
        _models["det"] = cv2.FaceDetectorYN.create(YUNET, "", (w, h), 0.8, 0.3, 5000)
    _models["det"].setInputSize((w, h))
    return _models["det"]


def _recognizer():
    if "rec" not in _models:
        _models["rec"] = _cv2().FaceRecognizerSF.create(SFACE, "")
    return _models["rec"]


def _decode(image_bytes):
    cv2 = _cv2()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > 640:                       # keep it fast
        s = 640 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def analyse(image_bytes, with_vector=True):
    """-> {"faces": n, "geometry": {...} | None, "vector": np.ndarray | None}"""
    img = _decode(image_bytes)
    if img is None:
        return {"faces": 0, "geometry": None, "vector": None}
    h, w = img.shape[:2]
    with _lock:
        _, faces = _detector(w, h).detect(img)
        faces = [] if faces is None else faces
        if len(faces) != 1:
            return {"faces": len(faces), "geometry": None, "vector": None}
        f = faces[0]
        vector = None
        if with_vector:
            rec = _recognizer()
            vector = rec.feature(rec.alignCrop(img, f)).flatten().astype(float)
    return {"faces": 1, "geometry": _geometry(f, w), "vector": vector}


def _geometry(f, frame_w):
    # YuNet row: x, y, w, h, right_eye(x,y), left_eye(x,y), nose(x,y), mouth_right(x,y), mouth_left(x,y), score
    re, le, nose = f[4:6], f[6:8], f[8:10]
    mr, ml = f[10:12], f[12:14]
    eye_dist = float(np.linalg.norm(le - re)) or 1.0
    eye_mid_x = (re[0] + le[0]) / 2
    return {
        "yaw": float((nose[0] - eye_mid_x) / eye_dist),        # + = person turned to THEIR left
        "smile": float(np.linalg.norm(ml - mr) / eye_dist),
        "size": float(f[2] / frame_w),
    }


def _cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


# ------------------------------------------------------------ vectors (saved at enrollment only)

def _load():
    try:
        with open(VECTORS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data):
    os.makedirs(os.path.dirname(VECTORS_FILE) or ".", exist_ok=True)
    tmp = VECTORS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, VECTORS_FILE)


def save_vector(person_id, image_bytes):
    if not available():
        return False
    a = analyse(image_bytes)
    if a["vector"] is None:
        return False
    data = _load()
    data[person_id] = [round(x, 6) for x in a["vector"]]
    _save(data)
    return True


def remove_vector(person_id):
    data = _load()
    if data.pop(person_id, None) is not None:
        _save(data)


def has_vector(person_id):
    return person_id in _load()


def best_match(vector):
    best, score = None, -1.0
    for pid, v in _load().items():
        s = _cosine(vector, v)
        if s > score:
            best, score = pid, s
    return best, score


# ------------------------------------------------------------ random challenge

def new_challenge():
    first = random.choice(THREE_D_ACTIONS)
    second = random.choice([a for a in ACTIONS if a != first])
    actions = [first, second]
    random.shuffle(actions)
    cid = secrets.token_urlsafe(12)
    ch = {"id": cid, "actions": actions, "labels": [ACTIONS[a] for a in actions],
          "expires": time.time() + CHALLENGE_SECONDS}
    with _lock:
        for k in [k for k, v in _challenges.items() if v["expires"] < time.time()]:
            _challenges.pop(k)
        _challenges[cid] = ch
    return ch


def _claim(challenge_id):
    with _lock:
        ch = _challenges.pop(challenge_id, None)     # single use
    if ch is None:
        return None, "Unknown or already used challenge (replay blocked)."
    if ch["expires"] < time.time():
        return None, "Challenge expired. Please start again."
    return ch, "ok"


def _action_done(action, base, frames):
    hits = 0
    for g in frames:
        if action == "turn_left" and g["yaw"] - base["yaw"] >= TURN_DELTA:
            hits += 1
        elif action == "turn_right" and base["yaw"] - g["yaw"] >= TURN_DELTA:
            hits += 1
        elif action == "smile" and g["smile"] >= base["smile"] * SMILE_RATIO:
            hits += 1
        elif action == "move_closer" and g["size"] >= base["size"] * CLOSER_RATIO:
            hits += 1
    return hits >= MIN_FRAMES_PER_ACTION, hits


def verify(challenge_id, frames):
    """Liveness + identity from the challenge frames. Never raises.
    -> {"ok", "live": True|False|None, "reason", "person_id", "cosine", "actions": [...], "frames": n}"""
    out = {"live": None, "person_id": None, "cosine": None, "actions": [], "frames": len(frames)}
    ch, why = _claim(challenge_id)
    if ch is None:
        return {**out, "live": False, "reason": why, "spoof_type": "replayed_challenge"}
    if not available():
        return {**out, "reason": "Offline face models are missing. Run python get_models.py."}

    analysed = {"baseline": [], 0: [], 1: []}
    for fr in frames:
        phase = fr.get("phase")
        if phase not in analysed:
            continue
        a = analyse(fr["image"], with_vector=True)
        if a["faces"] > 1:
            return {**out, "live": False, "reason": "More than one face in view.",
                    "spoof_type": "multiple_faces"}
        if a["faces"] == 1:
            analysed[phase].append(a)

    base_list = analysed["baseline"]
    if not base_list:
        return {**out, "reason": "No face found while looking straight. Please try again."}
    # baseline = most frontal frame
    base = min(base_list, key=lambda a: abs(a["geometry"]["yaw"]))
    if abs(base["geometry"]["yaw"]) > FRONTAL_MAX:
        return {**out, "reason": "Please look straight at the camera at the start."}

    # same person in every frame (blocks swapping faces / photos mid-challenge)
    for phase in ("baseline", 0, 1):
        for a in analysed[phase]:
            if _cosine(a["vector"], base["vector"]) < SAME_FACE_COSINE:
                return {**out, "live": False, "reason": "The face changed during the check.",
                        "spoof_type": "face_swap"}

    all_done = True
    for i, action in enumerate(ch["actions"]):
        done, hits = _action_done(action, base["geometry"], [a["geometry"] for a in analysed[i]])
        out["actions"].append({"action": action, "label": ACTIONS[action], "done": done, "frames": hits})
        all_done = all_done and done

    pid, cos = best_match(base["vector"])
    out["person_id"], out["cosine"] = pid, (round(cos, 3) if pid else None)
    if not all_done:
        missed = [a["label"] for a in out["actions"] if not a["done"]]
        return {**out, "live": False, "spoof_type": "failed_challenge",
                "reason": "Random action not performed: " + "; ".join(missed) + "."}
    return {**out, "live": True, "reason": "Both random actions performed by the same live face."}
