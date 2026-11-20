"""Shared helpers for ingestion: polite HTTP fetching and atomic file writes."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "pt-public-data-rag/0.1 (portfolio research pipeline)"

# INE's usage rules allow up to 15-20 req/s; we stay far below that on purpose.
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_last_request_at = 0.0


class DeadlineExceeded(TimeoutError):
    pass


def _read_with_deadline(resp, deadline: float) -> bytes:
    """Read the body in chunks, aborting if the total time exceeds the deadline.

    The `timeout` passed to urlopen only bounds each individual socket read, so a
    server that trickles bytes slowly can keep a request alive indefinitely.
    """
    start, chunks = time.monotonic(), []
    while chunk := resp.read(64 * 1024):
        chunks.append(chunk)
        if time.monotonic() - start > deadline:
            raise DeadlineExceeded(f"download exceeded {deadline:.0f}s total")
    return b"".join(chunks)


def fetch(url: str, *, timeout: float = 60.0, deadline: float = 300.0, max_retries: int = 4) -> bytes:
    """GET a URL with rate limiting and exponential backoff on transient errors.

    `timeout` bounds each socket operation; `deadline` bounds the whole download.
    """
    global _last_request_at
    for attempt in range(max_retries + 1):
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _read_with_deadline(resp, deadline)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt == max_retries:
                raise
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise
            reason = str(exc)
        backoff = 2**attempt * 5
        log.warning("Retry %d/%d for %s in %ds (%s)", attempt + 1, max_retries, url, backoff, reason)
        time.sleep(backoff)
    raise AssertionError("unreachable")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
