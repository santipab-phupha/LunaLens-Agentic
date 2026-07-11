from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import ensure_dir, env_int, log


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    timestamp_sec: float
    frame_index: int
    width: int
    height: int
    brightness: float
    dominant_color: str


@dataclass(frozen=True)
class VideoSample:
    video_path: Path
    frames: list[FrameInfo]
    duration_sec: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    motion_score: float | None

    @property
    def frame_paths(self) -> list[Path]:
        return [frame.path for frame in self.frames]

    def metadata(self) -> dict[str, object]:
        return {
            "video_file": self.video_path.name,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "source_width": self.width,
            "source_height": self.height,
            "sampled_frame_count": len(self.frames),
            "motion_score": self.motion_score,
            "frames": [
                {
                    "timestamp_sec": round(frame.timestamp_sec, 2),
                    "frame_index": frame.frame_index,
                    "width": frame.width,
                    "height": frame.height,
                    "brightness": round(frame.brightness, 3),
                    "dominant_color": frame.dominant_color,
                }
                for frame in self.frames
            ],
        }


def sample_video_frames(video_path: Path, output_dir: Path, task_id: str) -> VideoSample:
    ensure_dir(output_dir)
    target_count = env_int("MAX_FRAMES", 8, minimum=1, maximum=12)
    frame_width = env_int("FRAME_WIDTH", 512, minimum=224, maximum=1280)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        frame_count_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = frame_count_raw if frame_count_raw > 0 else None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        duration = (frame_count / fps) if frame_count and fps else None

        if frame_count:
            sample_count = min(target_count, frame_count)
            indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()
            frames = _sample_by_indices(capture, output_dir, indices, fps, frame_width)
        else:
            frames = _sample_sequential(capture, output_dir, target_count, frame_width)

        if not frames:
            raise ValueError("no frames could be extracted")

        motion_score = _estimate_motion(frames)
        log(f"[{task_id}] sampled {len(frames)} frame(s)")
        return VideoSample(
            video_path=video_path,
            frames=frames,
            duration_sec=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            motion_score=motion_score,
        )
    finally:
        capture.release()


def _sample_by_indices(
    capture: cv2.VideoCapture,
    output_dir: Path,
    indices: list[int],
    fps: float | None,
    target_width: int,
) -> list[FrameInfo]:
    frames: list[FrameInfo] = []
    for order, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        timestamp = (frame_index / fps) if fps else float(order)
        frames.append(_write_frame(frame, output_dir, order, frame_index, timestamp, target_width))
    return frames


def _sample_sequential(
    capture: cv2.VideoCapture,
    output_dir: Path,
    target_count: int,
    target_width: int,
) -> list[FrameInfo]:
    buffered: list[tuple[int, np.ndarray]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_index % 30 == 0:
            buffered.append((frame_index, frame.copy()))
        frame_index += 1

    frames: list[FrameInfo] = []
    if not buffered:
        return frames

    picks = np.linspace(0, len(buffered) - 1, num=min(target_count, len(buffered)), dtype=int).tolist()
    for order, buffered_index in enumerate(picks):
        original_index, frame = buffered[buffered_index]
        frames.append(_write_frame(frame, output_dir, order, original_index, float(order), target_width))
    return frames


def _write_frame(
    frame: np.ndarray,
    output_dir: Path,
    order: int,
    frame_index: int,
    timestamp: float,
    target_width: int,
) -> FrameInfo:
    resized = _resize_frame(frame, target_width)
    height, width = resized.shape[:2]
    target = output_dir / f"frame_{order:02d}.jpg"
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(target, format="JPEG", quality=85, optimize=True)
    return FrameInfo(
        path=target,
        timestamp_sec=timestamp,
        frame_index=frame_index,
        width=width,
        height=height,
        brightness=float(np.mean(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)) / 255.0),
        dominant_color=_dominant_color_name(resized),
    )


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width == target_width:
        return frame
    ratio = target_width / float(width)
    return cv2.resize(frame, (target_width, max(1, int(height * ratio))), interpolation=cv2.INTER_AREA)


def _dominant_color_name(frame_bgr: np.ndarray) -> str:
    mean_bgr = np.mean(frame_bgr.reshape(-1, 3), axis=0)
    blue, green, red = [float(channel) / 255.0 for channel in mean_bgr]
    max_channel = max(red, green, blue)
    min_channel = min(red, green, blue)
    if max_channel < 0.18:
        return "dark"
    if min_channel > 0.82:
        return "light"
    if max_channel - min_channel < 0.08:
        return "gray"

    hue, saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
    if saturation < 0.2:
        return "muted"
    degrees = hue * 360.0
    if degrees < 20 or degrees >= 340:
        return "red"
    if degrees < 45:
        return "orange"
    if degrees < 70:
        return "yellow"
    if degrees < 165:
        return "green"
    if degrees < 200:
        return "cyan"
    if degrees < 255:
        return "blue"
    if degrees < 295:
        return "purple"
    return "magenta"


def _estimate_motion(frames: list[FrameInfo]) -> float | None:
    if len(frames) < 2:
        return None

    previous: np.ndarray | None = None
    scores: list[float] = []
    for frame in frames:
        image = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        if previous is not None:
            scores.append(float(np.mean(cv2.absdiff(previous, image)) / 255.0))
        previous = image

    if not scores:
        return None
    return round(float(np.mean(scores)), 4)
