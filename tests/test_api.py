"""HTTP-level tests: create a batch, poll it, download the composed file."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

FAKE_FFMPEG = Path(__file__).with_name("fake_ffmpeg.py")
FAKE_CONFIG = Path(__file__).with_name("fake_config.json")


@pytest.fixture()
def client(tmp_path: Path):
    settings = Settings()
    settings.data_dir = tmp_path / "data"
    settings.ffmpeg_binary = str(FAKE_FFMPEG)
    settings.slots = 2
    settings.slow_after_seconds = 0.35
    settings.monitor_interval = 0.01
    settings.cancel_grace_seconds = 0.3
    settings.clip_seconds = 4
    FAKE_CONFIG.write_text(json.dumps({"sleep": 0.02}), encoding="utf-8")
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    FAKE_CONFIG.unlink(missing_ok=True)


def wait_ready(client: TestClient, batch_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/batches/{batch_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("batch did not settle")


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "预览片转码批次" in response.text
    assert "片段 A" in response.text


def test_create_validate_and_download_flow(client):
    bad = client.post("/api/batches", json={"slow_clip": "Z"})
    assert bad.status_code == 400

    created = client.post("/api/batches", json={"slow_clip": "b"})
    assert created.status_code == 201
    batch = created.json()
    assert batch["slow_clip"] == "B"
    assert [s["name"] for s in batch["segments"]] == ["A", "B", "C"]
    assert all(s["attempts"][0]["status"] == "queued" for s in batch["segments"])

    final = wait_ready(client, batch["id"])
    assert final["status"] == "ready"
    assert final["download_ready"] is True
    assert final["download_url"] == f"/batches/{batch['id']}/download"

    listed = client.get("/api/batches").json()["batches"]
    assert any(b["id"] == batch["id"] for b in listed)

    download = client.get(f"/batches/{batch['id']}/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("video/mp4")
    assert "attachment" in download.headers["content-disposition"]
    assert len(download.content) > 0

    config = client.get("/api/config").json()
    assert config["slots_total"] == 2
    assert config["clip_names"] == ["A", "B", "C"]


def test_download_before_ready_is_409(client):
    FAKE_CONFIG.write_text(
        json.dumps({"sleep": 1.5}), encoding="utf-8"
    )
    created = client.post("/api/batches", json={"slow_clip": "A"}).json()
    response = client.get(f"/batches/{created['id']}/download")
    assert response.status_code == 409
    assert "detail" in response.json()
    # The batch can still finish later.
    final = wait_ready(client, created["id"], timeout=15)
    assert final["status"] == "ready"


def test_unknown_batch_is_404(client):
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/batches/nope/download").status_code == 404


def test_failed_batch_has_no_download(client):
    FAKE_CONFIG.write_text(
        json.dumps({"segments": {"C": {"normal": {"sleep": 0.01, "exit": 4}}}}),
        encoding="utf-8",
    )
    created = client.post("/api/batches", json={"slow_clip": "A"}).json()
    final = wait_ready(client, created["id"])
    assert final["status"] == "failed"
    assert final["download_ready"] is False
    response = client.get(f"/batches/{created['id']}/download")
    assert response.status_code == 409


def test_api_stays_responsive_during_slow_probe(client):
    FAKE_CONFIG.write_text(
        json.dumps({"sleep": 0.02, "probe_sleep": 1.5}), encoding="utf-8"
    )
    created = client.post("/api/batches", json={"slow_clip": "A"}).json()

    deadline = time.monotonic() + 5
    while True:
        body = client.get(f"/api/batches/{created['id']}").json()
        if any(a["status"] == "probing" for a in body["attempts"]):
            break
        assert time.monotonic() < deadline, "no attempt reached probing"
        time.sleep(0.02)

    # The list endpoint answers immediately while a probe is stalled.
    t0 = time.monotonic()
    response = client.get("/api/batches")
    assert response.status_code == 200
    assert time.monotonic() - t0 < 1.0

    # And a second batch is dispatched during the probe window.
    second = client.post("/api/batches", json={"slow_clip": "B"}).json()
    still_probing = dispatched = False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        first_body = client.get(f"/api/batches/{created['id']}").json()
        second_body = client.get(f"/api/batches/{second['id']}").json()
        still_probing = any(a["status"] == "probing" for a in first_body["attempts"])
        dispatched = any(a["status"] == "running" for a in second_body["attempts"])
        if still_probing and dispatched:
            break
        time.sleep(0.02)
    assert still_probing and dispatched

    assert wait_ready(client, created["id"], timeout=15)["status"] == "ready"
    assert wait_ready(client, second["id"], timeout=15)["status"] == "ready"


def test_failed_probe_has_no_download(client):
    # The transcode exits 0 but the integrity probe rejects the artifact.
    FAKE_CONFIG.write_text(
        json.dumps({"sleep": 0.02, "probe_exit": 1}), encoding="utf-8"
    )
    created = client.post("/api/batches", json={"slow_clip": "A"}).json()
    final = wait_ready(client, created["id"])
    assert final["status"] == "failed"
    assert final["download_ready"] is False
    response = client.get(f"/batches/{created['id']}/download")
    assert response.status_code == 409
