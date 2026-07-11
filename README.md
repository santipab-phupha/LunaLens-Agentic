# Track 2 OpenAI Video Captioning Agent

Dockerized Python solution for AMD Developer Hackathon ACT II, Track 2: Video Captioning Agent. The app reads `/input/tasks.json`, downloads each video, samples frames, uses the OpenAI API as the primary vision-language model, and writes valid JSON to `/output/results.json` before exiting.

## Input Format

`/input/tasks.json`

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

For local testing only, `video_url` may also be a `file://` URL or a plain filesystem path.

## Output Format

`/output/results.json`

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

Every requested style is included. Captions are one English sentence, roughly 8 to 22 words, with no markdown or labels inside the caption text.

## Environment

Create a local `.local.env` from `.local.env.example`:

```bash
cp .local.env.example .local.env
```

Set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.6-luna
OPENAI_FALLBACK_MODEL=gpt-5.4-mini
OPENAI_REASONING_EFFORT=high
OPENAI_IMAGE_DETAIL=low
MAX_FRAMES=8
FRAME_WIDTH=512
CAPTION_CANDIDATES=2
SELF_CHECK=true
```

Never commit API keys. `.local.env` is ignored by both Git and Docker.

## Local Python Run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m app.main
```

By default, the app reads `/input/tasks.json` and writes `/output/results.json`. For local testing with custom paths:

```powershell
$env:TASKS_PATH = ".\sample_input\tasks.json"
$env:RESULTS_PATH = ".\sample_output\results.json"
python -m app.main
```

The app loads `.local.env` for local development if it exists. Existing environment variables are not overwritten.

`gpt-5.6-luna` is the cost-oriented primary model. Because GPT-5.6 access may be limited during preview, the agent retries model-access failures with `OPENAI_FALLBACK_MODEL`. Set that variable to an empty value to disable model fallback. High reasoning effort prioritizes caption quality, while low image detail limits image-token cost; lower reasoning effort if runtime or cost becomes more important.

## Docker Build

```bash
docker build --platform linux/amd64 -t track2-openai-captioner:latest .
```

## Docker Run

```bash
docker run --rm \
  --env-file .local.env \
  -v "$(pwd)/sample_input:/input" \
  -v "$(pwd)/sample_output:/output" \
  track2-openai-captioner:latest
```

PowerShell:

```powershell
docker run --rm `
  --env-file .local.env `
  -v "${PWD}\sample_input:/input" `
  -v "${PWD}\sample_output:/output" `
  track2-openai-captioner:latest
```

## Pipeline

1. Load and validate `/input/tasks.json`.
2. Download each video to a temporary file.
3. Extract `MAX_FRAMES` evenly distributed frames with OpenCV.
4. Resize frames to `FRAME_WIDTH` while preserving aspect ratio.
5. Save frames as JPEG and encode them as Base64 data URLs.
6. Ask OpenAI to extract schema-validated structured video facts from the sampled frames.
7. Ask OpenAI to generate schema-validated `CAPTION_CANDIDATES` captions per requested style.
8. If `SELF_CHECK=true`, ask OpenAI to select or rewrite schema-validated final captions.
9. Validate every requested style and write `/output/results.json`.

The whole video is not sent to OpenAI; only sampled frames are sent. The system does not hardcode or cache sample answers.

## Failure Handling

If `OPENAI_API_KEY` is missing or an OpenAI request fails, the app logs the issue to stderr and writes safe fallback captions for every requested style. It always creates `/output`, always writes valid JSON, and exits with code 0 on handled failures.
