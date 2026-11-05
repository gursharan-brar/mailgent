# helpdesk-voice

A Tier 1 IT help desk agent for a small office, built on the Anthropic SDK, plus the tools to
test it: realistic call scenarios, a simulated caller, a grader, an eval harness, and a voice
layer (speech-to-text, text-to-speech, browser call page).

| Role | Model |
|---|---|
| Help desk agent | `claude-haiku-5-5` |
| Simulated caller | `claude-haiku-5-5` |
| Grader (LLM judge) | `claude-sonnet-5-5` |

The API key is read from `ANTHROPIC_API_KEY`.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
pytest            # offline: no key, network, or audio hardware needed
```

## Phase 1: text agent and scenarios

```bash
python -m helpdesk.chat --scenario hd-001      # you type the caller's side
python -m helpdesk.chat --scenario hd-007 --no-brief
```

The chat prints a brief for you (persona, hidden root cause, red herrings) since you are
playing the caller; the agent never sees it. Press Enter at the first prompt to use the
scenario's opening line. The call ends when the agent files its ticket, or type `/quit`.
Transcripts go to `results/chats/`.

### Scenarios

`scenarios/hd-001.json` … `hd-030.json`. File names are sequential ids so they can't hint at
the cause.

- `hd-001` to `hd-010` are full scenarios. Fields: `id`, `tier` (1-3), `caller_persona`
  (`name`, `tech_skill`, `mood`), `opening_line`, `hidden_root_cause`, `correct_fix_steps`
  (ordered), `red_herrings`, `success_criteria`, `must_escalate`, plus two extras:
  `category` (used to check coverage) and `device` (what the caller is using, so the
  simulated caller can answer consistently).
- `hd-011` to `hd-030` are stubs: `id`, `tier`, `problem` (one line). The harness skips them.

Coverage of the full set: Windows lockouts (001, 002), printers (003, 004), Wi-Fi drops (005),
email sync (006), VPN (007), slow computer (008), MFA (009, 010). Four of the ten must be
escalated (002, 004, 009, 010).

### The agent

`helpdesk/agent.py`: `next_agent_message(conversation)` takes the call so far (a list of
`Turn`, ending with the caller) and returns the agent's next message. The system prompt is
`prompts/agent.md`. Rules it follows:

- one question or one instruction per reply, plain spoken language;
- never asks for passwords or codes, never suggests registry edits, disabling antivirus or
  the firewall, approving prompts the caller didn't start, remote-access tools, or admin
  credentials;
- escalates when the fix needs admin rights, on signs of a security incident, or at reply 12;
- ends every call with one `submit_ticket` tool call, validated against the `Ticket` model
  (`helpdesk/models.py`); an invalid ticket goes back to the model as a tool error to fix.

After each caller message the agent gets a one-line note from the "help desk software" with
its reply count (a mid-conversation system message). History is replayed append-only,
including the model's thinking blocks, as Haiku 5.5 requires.

Tunable through environment variables: `HELPDESK_AGENT_EFFORT` (low|medium|high|xhigh|max;
unset = the model's default), `HELPDESK_PROMPT_CACHE=0` to turn off prompt caching.

## Phase 2: simulated caller, grader, harness

```bash
python -m helpdesk.run --runs 3 --models haiku          # all 10 full scenarios x 3 runs
python -m helpdesk.run --runs 1 --scenarios hd-001,hd-007 --concurrency 2
python -m helpdesk.run --runs 3 --models haiku,sonnet   # compare agent models
python -m helpdesk.report                               # table for the newest results file
python -m helpdesk.report results/20261105-120000.jsonl
```

Each run: the simulated caller opens with the scenario's opening line, then the agent and the
caller alternate until the agent files a ticket. Hard caps: 15 agent replies
(`--max-turns`) and 200,000 agent+caller tokens per run (`--max-tokens-per-run`, checked
before every model call). Then the grader scores the call. Runs execute in parallel
(`--concurrency`, default 5).

Output: `results/<timestamp>.jsonl`, one line per run with the full transcript, ticket, grade,
token counts per role, cost, and wall time. The report is printed at the end and also saved
next to it as `results/<timestamp>.md`.

**Simulated caller** (`helpdesk/caller.py`, prompt `prompts/caller.md`, `claude-haiku-5-5`):
plays the scenario's persona at its tech-skill level and mood, answers only what it's asked,
never states the hidden root cause (it knows it only to keep its answers consistent), goes
along with red herrings if the agent chases them, and confirms the fix once the key step is
done. Each caller reply is a single request containing the transcript so far.

**Grader** (`helpdesk/grader.py`):

1. Deterministic checks: a ticket was submitted, its status matches `must_escalate`
   (`escalated` vs `resolved`), and the pattern scanner in `helpdesk/safety.py` finds no risky
   step in anything the agent said.
2. LLM judge on `claude-sonnet-5-5` with the rubric in `prompts/grader.md`, using structured
   output so it returns JSON: `root_cause_correct`, `fix_correct`, `risky_advice` (list),
   `efficiency` (turns used), `score` (0-100), `reason` (one line).

A run **passes** when the deterministic checks pass, the judge marks root cause and fix
correct, lists no risky advice, and scores at least 70 (`HELPDESK_PASS_SCORE`).

The judge request turns on server-side refusal fallback (beta `server-side-fallback-2026-07-01`,
`fallbacks: "default"`): if a safety classifier declines the grading request, the API retries
it on a fallback model in the same call, and `grade.judge_model` records which model answered.
If your organization rejects the parameter, the grader retries without it automatically; set
`HELPDESK_GRADER_FALLBACKS=off` to never send it.

**Cost**: `prices.toml` has blank placeholders for each model. Fill in USD per million tokens
from Anthropic's pricing page. Until then cost shows as `n/a`. The report recomputes cost from
the saved token counts, so filling prices in later and re-running the report works.

Other knobs: `HELPDESK_CALLER_EFFORT`, `HELPDESK_GRADER_EFFORT` (unset = model default).

## Phase 3: voice layer

```bash
python -m helpdesk.voice.server                       # browser call at http://127.0.0.1:8000
python -m helpdesk.voice_test --scenario hd-001       # no-mic end-to-end call
python -m helpdesk.voice_test --scenario hd-010 --grade
python -m helpdesk.latency_report                     # summarize results/latency.jsonl
```

Real speech models: see **[SETUP_LOCAL.md](SETUP_LOCAL.md)**. Without them everything runs
on the fake adapters (the default).

**Adapters** (`helpdesk/voice/adapters.py`): `STTAdapter.transcribe(wav) -> text` and
`TTSAdapter.synthesize(text) -> wav` (16-bit PCM mono WAV bytes). Real ones are
`FasterWhisperSTT` (faster-whisper, `small` model, CPU int8 by default) and `PiperTTS`.
Fakes: `FakeTTS` returns silent audio sized to the text, with the text tucked into an extra
WAV chunk, and `FakeSTT` returns that text, or canned text for any other audio. That
round trip lets full conversations run through the voice pipeline with no models. Chosen
by environment variable:

| Variable | Values |
|---|---|
| `HELPDESK_STT` | `fake` (default), `faster-whisper` |
| `HELPDESK_TTS` | `fake` (default), `piper` |
| `HELPDESK_PIPER_VOICE` | path to the agent's Piper `.onnx` voice |
| `HELPDESK_PIPER_CALLER_VOICE` | optional second voice for the simulated caller in `voice_test` |
| `HELPDESK_WHISPER_MODEL`, `_DEVICE`, `_COMPUTE`, `_BEAM`, `_DIR` | `small`, `cpu`, `int8`, `1`, download folder |

**Server** (`helpdesk/voice/server.py`): FastAPI with one WebSocket endpoint, `/ws`, and the
call page `static/call.html`. The page captures the mic with an AudioWorklet, ends a turn
after a configurable silence (RMS threshold plus silence duration, with 300 ms of pre-roll),
sends the turn as a 16 kHz WAV, and plays reply audio as it arrives. You can also type a
caller line. On the server, the agent's reply streams from the API, is split into sentences
as it arrives, and each sentence is synthesized while the model is still writing the rest,
so speech starts after the first sentence. The protocol is documented at the top of
`server.py`.

**Latency** per turn, logged to `results/latency.jsonl`:

- `stt_ms`: speech-to-text time;
- `llm_first_token_ms`: request sent to first reply token;
- `tts_first_audio_ms`: synthesis time of the first sentence;
- `first_audio_ms`: end of the caller's audio to the first reply audio being ready;
- `total_ms`: end of the caller's audio to the last reply audio.

**No-mic test** (`helpdesk/voice_test.py`): the simulated caller's lines go through TTS,
then STT, then the agent and sentence-by-sentence TTS, all through the same pipeline as the
server. The agent only hears what STT produced. It saves `call.json` and every WAV under
`results/calls/<timestamp>-<id>/` and logs latency.

## Layout

```
helpdesk/          agent, caller, grader, harness (run), report, models, safety, pricing, CLIs
helpdesk/voice/    adapters, audio helpers, sentence splitter, voice pipeline, server
prompts/           agent.md, caller.md, grader.md
scenarios/         30 scenario files
static/call.html   browser call page
prices.toml        blank price placeholders for you to fill in
SETUP_LOCAL.md     running the real speech models and mic call on your machine
tests/             offline tests; tests/fake_api.py fakes the Anthropic HTTP API so the real SDK is exercised
results/           run outputs (git-ignored)
```
