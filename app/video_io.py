from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import requests

from .utils import ensure_dir, log, safe_slug


MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(512 * 1024 * 1024)))
REQUEST_TIMEOUT = (10, 120)


def fetch_video(video_url: str, task_id: str, work_dir: Path) -> Path:
    ensure_dir(work_dir)
    parsed = urlparse(video_url)

    if parsed.scheme in ("http", "https"):
        return _download_http(video_url, task_id, work_dir)

    if parsed.scheme == "file":
        return _copy_local(_path_from_file_url(video_url), task_id, work_dir)

    possible_path = Path(video_url)
    if possible_path.exists():
        return _copy_local(possible_path, task_id, work_dir)

    raise FileNotFoundError(f"unsupported or inaccessible video_url for task {task_id}: {video_url}")


def _download_http(video_url: str, task_id: str, work_dir: Path) -> Path:
    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    target = work_dir / f"{safe_slug(task_id)}{suffix}"

    log(f"[{task_id}] downloading video")
    with requests.get(video_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        bytes_written = 0
        with target.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > MAX_VIDEO_BYTES:
                    raise ValueError(f"video exceeds MAX_VIDEO_BYTES={MAX_VIDEO_BYTES}")
                fh.write(chunk)

    return target


def _copy_local(source: Path, task_id: str, work_dir: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"local video does not exist: {source}")

    suffix = source.suffix or ".mp4"
    target = work_dir / f"{safe_slug(task_id)}{suffix}"
    if source == target:
        return target

    log(f"[{task_id}] copying local video: {source}")
    shutil.copyfile(source, target)
    return target


def _path_from_file_url(file_url: str) -> Path:
    parsed = urlparse(file_url)
    if parsed.netloc and parsed.netloc not in ("localhost", ""):
        return Path(f"//{parsed.netloc}{url2pathname(unquote(parsed.path))}")
    return Path(url2pathname(unquote(parsed.path)))
