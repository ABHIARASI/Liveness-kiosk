"""Kiosk server: serves the kiosk web page and connects it to the decision pipeline.

Run:   python server.py        then open  http://localhost:8000

Flow:  browser camera -> AWS Rekognition Face Liveness (liveness + spoof detection)
       -> /api/decide: face authentication (Rekognition) + campus schedule + rules
       -> ALLOW / STEP-UP VERIFICATION / DENY, logged, explained by Bedrock.
Offline: if AWS is unreachable, the page switches to the safe offline process (offline.py).
"""
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import base64
import secrets
import threading

import evaluate
import face_service
import local_face
import offline
import official_liveness
import pipeline
import security
import store
import voice

app = FastAPI(title="HID Liveness Kiosk")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiosk_web", "dist")


class DecideIn(BaseModel):
    session_id: str
    location: str | None = None
    test_label: str | None = None      # set by the kiosk's test mode, for the performance report


class OverrideIn(BaseModel):
    person_id: str | None = None
    verdict: str
    pin: str = ""
    note: str = ""


class EnrollIn(BaseModel):
    session_id: str
    token: str | None = None          # one-time token issued by the admin console
    person_id: str | None = None
    password: str | None = None
    role: str = "student"
    status: str = "active"


class OfflineIn(BaseModel):
    person_id: str
    code: str


class FrameIn(BaseModel):
    phase: str | int
    image: str                         # data URL or base64 JPEG


class OfflineFaceIn(BaseModel):
    challenge_id: str
    frames: list[FrameIn]
    location: str | None = None


class StepUpIn(BaseModel):
    ticket: str
    code: str
    location: str | None = None


_stepups = {}            # ticket -> (face_check, expires): weak offline match waiting for the code
_stepup_lock = threading.Lock()


class SimulateIn(BaseModel):
    offline: bool


class SpeakIn(BaseModel):
    text: str


def _offline_error():
    offline.mark_offline()
    return JSONResponse(status_code=503, content={"detail": "Cloud unreachable: switched to offline mode.",
                                                  "offline": True})


@app.get("/api/status")
def status():
    online = offline.is_online()
    synced = offline.sync() if online and offline.pending() else 0
    threat = security.threat_status()
    return {"online": online, "simulated_offline": offline.is_simulated_offline(),
            "location": config.KIOSK_LOCATION, "pending_sync": len(offline.pending()),
            "just_synced": synced, "threat": threat, "locations": config.LOCATIONS,
            "offline_face_ready": local_face.available(),
            "metrics": evaluate.live_kiosk_report(store.recent_events(2000))}


@app.post("/api/session")
def create_session():
    if not offline.is_online():
        return _offline_error()
    try:
        session_id = official_liveness.create_session()
    except pipeline.NETWORK_ERRORS:
        return _offline_error()
    security.register_session(session_id)
    try:
        credentials, mode = security.kiosk_credentials()
    except Exception:
        raise HTTPException(500, "No AWS credentials on this machine. Run aws sts get-caller-identity.")
    return {"sessionId": session_id, "region": config.REGION, "credentials": credentials,
            "credential_mode": mode}


@app.post("/api/decide")
def decide(body: DecideIn):
    ok, why = security.claim_session(body.session_id)
    if not ok:
        raise HTTPException(400, why)
    t0 = time.perf_counter()
    try:
        live_result, reference = official_liveness.get_result(body.session_id)
        fetch_ms = int((time.perf_counter() - t0) * 1000)
        if reference is None:
            result = {"verdict": "RETRY", "verification": None, "reasons": live_result["reasons"],
                      "person_id": None, "similarity": None, "liveness": live_result,
                      "schedule": None, "decision_ms": fetch_ms, "timing": {"liveness_result": fetch_ms},
                      "explanation": "The liveness check did not finish. Please try again.",
                      "mode": "online"}
            return result
        extra = {"test_label": body.test_label} if body.test_label else None
        result = pipeline.check_official(live_result, reference, use_ai=True,
                                         location=body.location, extra=extra)
    except pipeline.OfflineError:
        return _offline_error()
    except pipeline.NETWORK_ERRORS:
        return _offline_error()
    result["timing"] = {"liveness_result": fetch_ms, **result["timing"]}
    result["decision_ms"] = fetch_ms + result["decision_ms"]   # post-capture decision time
    result["timing"]["total"] = result["decision_ms"]
    result.pop("challenge", None)
    result["speech"] = voice.speech_text(result)
    return result


@app.post("/api/enroll")
def enroll(body: EnrollIn):
    """Liveness-verified enrollment: only a LIVE face (AWS Face Liveness) can become a stored vector."""
    if body.token:
        ticket = security.consume_enroll_token(body.token)
        if ticket is None:
            raise HTTPException(403, "Enrollment link expired or already used. Start again from the admin console.")
        body.person_id, body.role, body.status = ticket["person_id"], ticket["role"], ticket["status"]
    elif body.password != config.ADMIN_PASSWORD:
        raise HTTPException(403, "Enrollment must be started from the admin console.")
    person_id = face_service.clean_person_id(body.person_id)
    if not person_id:
        raise HTTPException(400, "Use only letters, numbers, spaces, _ - . or :")
    ok, why = security.claim_session(body.session_id)
    if not ok:
        raise HTTPException(400, why)
    try:
        live, reference = official_liveness.get_result(body.session_id)
    except pipeline.NETWORK_ERRORS:
        return _offline_error()
    conf = live["signals"].get("aws_confidence", 0.0)
    if reference is None or conf < config.ENROLL_LIVE_CONFIDENCE:
        store.log_event({"verdict": "ENROLL_REJECTED", "person_id": person_id, "mode": "enroll",
                         "reasons": [f"Liveness {conf}% < {config.ENROLL_LIVE_CONFIDENCE}% required."]})
        return {"ok": False, "message": f"Enrollment rejected: liveness {conf}% is below the "
                                        f"{config.ENROLL_LIVE_CONFIDENCE}% required. A photo, screen "
                                        "or mask cannot be enrolled."}
    existing = face_service.identify(reference)
    if existing["status"] == "MATCH" and existing["person_id"] != person_id:
        store.log_event({"verdict": "ENROLL_REJECTED", "person_id": person_id, "mode": "enroll",
                         "reasons": [f"Face already enrolled as {existing['person_id']}."]})
        security.raise_alert("duplicate_enrollment",
                             f"Attempt to enroll {existing['person_id']}'s face as {person_id}.",
                             {"person_id": person_id, "location": config.KIOSK_LOCATION})
        return {"ok": False, "message": f"Enrollment rejected: this face is already enrolled as "
                                        f"{existing['person_id']} ({existing['similarity']}%)."}
    result = face_service.enroll(reference, person_id)
    if not result["ok"]:
        return result
    try:   # offline face vector, saved ONLY here at liveness-verified enrollment (no photo kept)
        offline_saved = local_face.save_vector(person_id, reference)
    except Exception:
        offline_saved = False
    store.upsert_person(person_id, role=body.role, status=body.status,
                        enrolled_via="aws_face_liveness", enroll_liveness=conf,
                        offline_vector=offline_saved)
    store.log_event({"verdict": "ENROLLED", "person_id": person_id, "mode": "enroll",
                     "reasons": [f"Liveness-verified enrollment ({conf}%)."]})
    extra = " Offline face vector saved." if offline_saved else " (Offline face vector NOT saved.)"
    return {"ok": True, "offline_vector": offline_saved,
            "message": f"{person_id} enrolled with a liveness-verified face ({conf}%).{extra}"}


@app.get("/api/enroll_token/{token}")
def enroll_token(token: str):
    item = security.peek_enroll_token(token)
    if item is None:
        raise HTTPException(404, "Enrollment link expired or already used. Start again from the admin console.")
    return {"person_id": item["person_id"], "role": item["role"], "status": item["status"]}


@app.post("/api/override")
def override(body: OverrideIn):
    if body.pin != config.GUARD_PIN:
        raise HTTPException(403, "Wrong guard PIN.")
    return pipeline.record_override(body.person_id, body.verdict, by="guard", note=body.note)


@app.post("/api/offline/decide")
def offline_decide(body: OfflineIn):
    from datetime import datetime
    result = offline.offline_decide(body.person_id, body.code, config.KIOSK_LOCATION,
                                    datetime.now(), faces_in_view=1)
    result["speech"] = voice.speech_text(result)
    return result


@app.post("/api/offline/challenge")
def offline_challenge():
    if not local_face.available():
        raise HTTPException(503, "Offline face models missing. Run python get_models.py.")
    ch = local_face.new_challenge()
    return {"challenge_id": ch["id"], "actions": ch["actions"], "labels": ch["labels"]}


def _decode_image(data):
    if "," in data[:100]:
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception:
        return b""


def _offline_face_result(face_check, code, location, t0):
    from datetime import datetime
    result = offline.offline_face_decide(face_check, code, location or config.KIOSK_LOCATION,
                                         datetime.now())
    if result["verdict"] == "REQUEST_VERIFICATION":
        ticket = secrets.token_urlsafe(12)
        with _stepup_lock:
            _stepups[ticket] = (face_check, time.time() + 120)
        result["stepup_ticket"] = ticket
    result["decision_ms"] = int((time.perf_counter() - t0) * 1000)
    result["timing"] = {"offline_face_and_liveness": result["decision_ms"]}
    result["speech"] = voice.speech_text(result)
    return result


@app.post("/api/offline/face")
def offline_face(body: OfflineFaceIn):
    """Offline access: local OpenCV face match + random-action liveness. No AWS."""
    t0 = time.perf_counter()
    frames = [{"phase": f.phase if f.phase == "baseline" else int(f.phase),
               "image": _decode_image(f.image)} for f in body.frames[:60]]
    face_check = local_face.verify(body.challenge_id, frames)
    return _offline_face_result(face_check, None, body.location, t0)


@app.post("/api/offline/stepup")
def offline_stepup(body: StepUpIn):
    t0 = time.perf_counter()
    with _stepup_lock:
        item = _stepups.pop(body.ticket, None)      # single use
    if item is None or item[1] < time.time():
        raise HTTPException(400, "Verification expired. Please start again.")
    return _offline_face_result(item[0], body.code or "", body.location, t0)


@app.post("/api/offline/simulate")
def simulate(body: SimulateIn):
    offline.set_simulated_offline(body.offline)
    return status()


@app.post("/api/offline/sync")
def sync():
    return {"synced": offline.sync(), "pending": len(offline.pending())}


@app.post("/api/speak")
def speak(body: SpeakIn):
    audio = voice.synthesize(body.text[:300])
    if audio is None:
        raise HTTPException(503, "Voice unavailable.")
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/api/log")
def log(limit: int = 50):
    return store.recent_events(limit)


if os.path.isdir(WEB_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))


if __name__ == "__main__":
    print(f"\n  Kiosk running:  http://localhost:{config.SERVER_PORT}\n")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
