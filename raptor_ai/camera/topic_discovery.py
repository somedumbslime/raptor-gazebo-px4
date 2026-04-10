from __future__ import annotations

import subprocess


def list_gz_topics(timeout_s: float = 2.0) -> list[str]:
    try:
        p = subprocess.run(
            ["gz", "topic", "-l"],
            capture_output=True,
            text=True,
            timeout=max(0.2, float(timeout_s)),
            check=False,
        )
    except Exception:
        return []
    if p.returncode != 0:
        return []
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def topic_message_type(topic: str, timeout_s: float = 2.0) -> str:
    t = str(topic).strip()
    if not t:
        return ""
    try:
        p = subprocess.run(
            ["gz", "topic", "-i", "-t", t],
            capture_output=True,
            text=True,
            timeout=max(0.2, float(timeout_s)),
            check=False,
        )
    except Exception:
        return ""
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    for line in out.splitlines():
        if "gz.msgs." in line:
            return line.strip()
    return ""


def discover_image_topics(
    *,
    hints: tuple[str, ...] = ("camera", "image"),
    max_topics: int = 300,
    timeout_s: float = 2.0,
) -> list[str]:
    topics = list_gz_topics(timeout_s=timeout_s)
    if not topics:
        return []

    scored: list[tuple[int, str]] = []
    for topic in topics[: max(1, int(max_topics))]:
        lower = topic.lower()
        score = 0
        for h in hints:
            if h in lower:
                score += 1
        if score == 0:
            continue
        scored.append((score, topic))

    scored.sort(key=lambda x: (-x[0], x[1]))
    ordered = [t for _, t in scored]

    out: list[str] = []
    for t in ordered:
        mtype = topic_message_type(t, timeout_s=timeout_s)
        if "gz.msgs.Image" in mtype:
            out.append(t)
    return out
