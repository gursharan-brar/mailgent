import pytest

from helpdesk.safety import scan_agent_turns, scan_text

RISKY = [
    ("Open regedit and go to HKEY_LOCAL_MACHINE.", "registry edit"),
    ("Let's edit the registry to fix that.", "registry edit"),
    ("Can you turn off Windows Defender for a minute?", "disable security software"),
    ("Please disable your antivirus and try again.", "disable security software"),
    ("Go ahead and switch off the firewall.", "disable security software"),
    ("What's your password?", "asks for a password or code"),
    ("Read me the six-digit code from the app.", "asks for a password or code"),
    ("I'll need your password to check.", "asks for a password or code"),
    ("Go ahead and install TeamViewer so I can look.", "remote access or cleaner tool"),
    ("Log in with the admin password on the sticky note.", "admin credentials"),
    ("Right-click it and run it as administrator.", "admin credentials"),
    ("Run Set-ExecutionPolicy Unrestricted first.", "security-weakening command"),
    ("You don't need to worry, just disable the firewall.", "disable security software"),
]

SAFE = [
    "I'll never ask for your password.",
    "Please don't share your password with anyone, including me.",
    "Don't turn off your antivirus, it's not the problem.",
    "That needs admin rights, so I'm escalating it to our admin.",
    "Type your password again slowly, and use the eye icon to check it.",
    "Can you read me the error message exactly?",
    "Instead of disabling the firewall, let's check which VPN profile is selected.",
    "Deny any sign-in requests you didn't start, and don't approve any of them.",
    "Is the little Caps Lock light on?",
    "Never give anyone your verification code.",
]


@pytest.mark.parametrize("text,rule", RISKY)
def test_flags_risky_instructions(text, rule):
    hits = scan_text(text)
    assert rule in {h.rule for h in hits}, hits


@pytest.mark.parametrize("text", SAFE)
def test_ignores_safe_or_negated_text(text):
    assert scan_text(text) == []


def test_turn_numbers_are_one_based():
    hits = scan_agent_turns(["Hi, what's going on?", "Open regedit please."])
    assert [h.turn for h in hits] == [2]
    assert "regedit" in hits[0].excerpt
