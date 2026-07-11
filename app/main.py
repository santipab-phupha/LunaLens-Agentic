from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from .caption_agent import CaptionAgent
from .frame_sampler import sample_video_frames
from .schema import Task, load_tasks
from .utils import ensure_caption_map, ensure_dir, fallback_caption, load_local_env, log, safe_slug, write_json_atomic
from .video_io import fetch_video


DEFAULT_INPUT_PATH = Path("/input/tasks.json")
DEFAULT_OUTPUT_PATH = Path("/output/results.json")


def process_task(task: Task, agent: CaptionAgent, root_work_dir: Path) -> dict[str, object]:
    log(f"[{task.task_id}] starting")
    task_dir = root_work_dir / safe_slug(task.task_id)
    ensure_dir(task_dir)

    try:
        video_path = fetch_video(task.video_url, task.task_id, task_dir)
        sample = sample_video_frames(video_path, task_dir / "frames", task.task_id)
        captions = agent.caption_video(sample, task.styles)
    except Exception as exc:
        log(f"[{task.task_id}] failed; writing safe fallback captions: {exc}")
        captions = {style: fallback_caption(style) for style in task.styles}

    return {"task_id": task.task_id, "captions": ensure_caption_map(captions, task.styles)}


def run(input_path: Path, output_path: Path) -> int:
    ensure_dir(output_path.parent)
    try:
        tasks = load_tasks(input_path)
    except Exception as exc:
        log(f"could not load tasks from {input_path}: {exc}")
        write_json_atomic(output_path, [])
        return 0

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="track2-openai-captioner-") as tmp:
        agent = CaptionAgent.from_env()
        root_work_dir = Path(tmp)
        for task in tasks:
            results.append(process_task(task, agent, root_work_dir))

    write_json_atomic(output_path, results)
    log(f"wrote {len(results)} result(s) to {output_path}")
    return 0


def main() -> int:
    load_local_env()
    input_path = Path(os.getenv("TASKS_PATH", str(DEFAULT_INPUT_PATH)))
    output_path = Path(os.getenv("RESULTS_PATH", str(DEFAULT_OUTPUT_PATH)))
    return run(input_path, output_path)


if __name__ == "__main__":
    sys.exit(main())
