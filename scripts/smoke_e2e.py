#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = os.getenv("YAF_API_BASE", "http://127.0.0.1:8000").rstrip("/")
REFERENCE = os.getenv("YAF_REFERENCE", "Luke 21:5-28")
TRANSLATION = os.getenv("YAF_TRANSLATION", "WEB")


def _request(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {path}: {exc.reason}") from exc


def main() -> None:
    health = _request("/health")
    if health.get("status") != "ok":
        raise RuntimeError("Health check did not return status=ok")

    passage = _request(f"/api/bible/passage?{urlencode({'reference': REFERENCE, 'translation': TRANSLATION})}")
    passage_text = passage.get("text")
    if not passage_text:
        raise RuntimeError("Passage lookup returned empty text")

    study = _request(
        "/api/study-plan",
        method="POST",
        payload={"reference": REFERENCE, "translation": TRANSLATION, "passage_text": passage_text},
    )
    if not study.get("discussion_questions"):
        raise RuntimeError("Study plan did not include discussion questions")

    image = _request(
        "/api/passage-image",
        method="POST",
        payload={
            "reference": REFERENCE,
            "translation": TRANSLATION,
            "style": "modern_editorial_illustration",
        },
    )
    if not image.get("image_b64_or_url"):
        raise RuntimeError("Passage image endpoint returned no image payload")

    chat = _request(
        "/api/persona-chat",
        method="POST",
        payload={
            "messages": [{"role": "user", "content": "What is one key takeaway from this passage?"}],
            "reference_context": REFERENCE,
            "translation": TRANSLATION,
        },
    )
    if not chat.get("reply"):
        raise RuntimeError("Persona chat returned empty reply")

    hymn = _request(
        "/api/hymn/generate",
        method="POST",
        payload={
            "reference": REFERENCE,
            "translation": TRANSLATION,
            "style_hint": "modern worship hymn, acoustic",
            "passage_text": passage_text,
        },
    )
    job_id = hymn.get("job_id")
    if not job_id:
        raise RuntimeError("Hymn generation did not return a job_id")

    status = hymn.get("job_status")
    for _ in range(8):
        if status in {"completed", "failed"}:
            break
        time.sleep(1.0)
        job = _request(f"/api/hymn/jobs/{job_id}")
        status = job.get("status")

    print("Smoke flow complete")
    print(f"reference={REFERENCE}")
    print(f"hymn_job_status={status}")


if __name__ == "__main__":
    main()
