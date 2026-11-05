# Running the real voice call on your laptop

This project was built in a cloud session that had no microphone, no speaker, and no
`ANTHROPIC_API_KEY`. `faster-whisper` and `piper-tts` installed fine from PyPI, but both
models are hosted on huggingface.co, which that network blocked (`403 Forbidden` from the
proxy). So the voice layer was finished and tested with fake adapters, a fake Anthropic API,
and a tiny stand-in Piper model that exercises the real Piper library but doesn't speak.
The steps below run the real thing.

## 1. What you need

- Python 3.12 and git
- Chrome, Edge or Firefox (the call page captures the mic with the Web Audio API)
- An Anthropic API key
- Disk for the models: roughly 0.5 GB for Whisper small plus about 60 MB per Piper voice
- Headphones (recommended; the page also mutes the mic while the agent talks)

## 2. Install

```bash
git clone https://github.com/gursharan-brar/mailgent helpdesk-voice
cd helpdesk-voice
python3.12 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e ".[dev,voice]"
export ANTHROPIC_API_KEY=sk-ant-...   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
pytest                                # all offline; should pass before you go further
```

## 3. Download the models

Whisper small in faster-whisper's CTranslate2 format (`Systran/faster-whisper-small`),
saved into `models/whisper/`:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8', download_root='models/whisper')"
```

This leaves `config.json`, `model.bin`, `tokenizer.json` and `vocabulary.txt` under
`models/whisper/`. (Without `download_root` they go to the Hugging Face cache in
`~/.cache/huggingface/`, which also works.)

Two Piper voices, one for the agent and one for the simulated caller in the no-mic test:

```bash
mkdir -p models/piper
python -m piper.download_voices en_US-lessac-medium en_US-amy-medium --download-dir models/piper
```

This gives `models/piper/en_US-lessac-medium.onnx`, `en_US-lessac-medium.onnx.json`,
`en_US-amy-medium.onnx` and `en_US-amy-medium.onnx.json`. If that command can't reach the
internet, download the same four files in a browser:

```
https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

`models/` is git-ignored.

## 4. Point the app at the real adapters

```bash
export HELPDESK_STT=faster-whisper
export HELPDESK_WHISPER_DIR=models/whisper
export HELPDESK_TTS=piper
export HELPDESK_PIPER_VOICE=models/piper/en_US-lessac-medium.onnx
export HELPDESK_PIPER_CALLER_VOICE=models/piper/en_US-amy-medium.onnx   # only used by voice_test
```

Optional: `HELPDESK_WHISPER_DEVICE=cuda HELPDESK_WHISPER_COMPUTE=float16` on an NVIDIA GPU,
`HELPDESK_WHISPER_BEAM=5` for more accurate but slower transcription (default 1),
`HELPDESK_PIPER_LENGTH_SCALE=0.9` to make the voice a bit faster.

## 5. Check the speech models with no mic

```bash
HELPDESK_REAL_VOICE_TESTS=1 pytest tests/test_voice_units.py -k real_models   # Piper says a line, Whisper hears it
python -m helpdesk.voice_test --scenario hd-001 --grade
python -m helpdesk.latency_report
```

`voice_test` has Piper speak the simulated caller's lines, Whisper transcribe them, and the
agent answer (streamed, spoken sentence by sentence). It saves every WAV file and
`call.json` (what the caller said, what Whisper heard, what the agent said, per-turn
latency, the ticket and grade) to `results/calls/<timestamp>-hd-001/`. Play the WAVs to hear
the call.

## 6. The real voice call

```bash
python -m helpdesk.voice.server      # loads both speech models at startup, then serves
```

Open http://127.0.0.1:8000, click **Start call**, and allow the microphone. The agent
greets you; speak, then pause. About a second of silence ends your turn. Watch the level
meter: if your turns end too early or don't end, move the silence threshold slider (the red
mark) or change "End turn after … ms". The mic only works on `localhost` or over HTTPS.

Per-turn latency (STT, LLM first token, TTS first audio, first audio, total) shows on the
page and is appended to `results/latency.jsonl`; `python -m helpdesk.latency_report`
summarizes it.

## 7. The Phase 2 eval run

```bash
python -m helpdesk.run --runs 3 --models haiku   # 10 scenarios x 3 runs, prints the table
# fill in prices.toml, then re-print the table with costs:
python -m helpdesk.report
```

## Troubleshooting

- **`403` or proxy errors while downloading**: your network blocks huggingface.co. Download
  the files on another network and copy them into `models/` as shown above.
  `HELPDESK_WHISPER_MODEL` also accepts a path to a folder that holds the four Whisper files.
- **The first reply is slow**: the server loads the models at startup; `voice_test` loads
  them on the first turn.
- **The agent hears itself**: use headphones. The page mutes the mic while reply audio
  plays, but laptop speakers can still bleed into the next turn.
- **`ANTHROPIC_API_KEY is not set`**: export it in the same terminal that runs the server.
- **To run this in the cloud session instead**: add `ANTHROPIC_API_KEY` as an environment
  secret and allow huggingface.co (plus the download hosts it redirects to) in the
  environment's network settings. There's still no mic there, but `voice_test` and the
  harness would run.
