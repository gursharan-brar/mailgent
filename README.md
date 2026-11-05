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

## Layout

```
helpdesk/        package (agent, models, scenarios, safety checks, CLIs)
prompts/         agent.md (system prompt)
scenarios/       30 scenario files
tests/           offline tests; tests/fake_api.py fakes the Anthropic HTTP API so the real SDK is exercised
results/         run outputs (git-ignored)
```
