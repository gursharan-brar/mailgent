"""Split streamed text into sentences so each can be spoken as soon as it is complete."""

from __future__ import annotations

import re

_ABBREVIATIONS = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "vs", "etc", "st", "no", "approx", "min", "sec"}
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)|\n+")


class SentenceSplitter:
    """Feed text deltas; get back complete sentences. Call flush() at the end of the stream.

    A sentence ends at . ! ? or … followed by whitespace, or at a newline. Common abbreviations
    ("e.g.", "Dr.") don't end a sentence, and "2.5" doesn't either (no space after the dot).
    A run with no boundary longer than max_chars is cut at the last comma or space.
    """

    def __init__(self, max_chars: int = 240):
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        while (cut := self._find_cut()) is not None:
            sentence, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:].lstrip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

    def _find_cut(self) -> int | None:
        for match in _BOUNDARY.finditer(self._buffer):
            if match.group().startswith("\n"):
                return match.end()
            if match.group() == ".":
                word = re.search(r"(\S+)$", self._buffer[: match.start()])
                if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
                    continue
            return match.end()
        if len(self._buffer) > self.max_chars:
            window = self._buffer[: self.max_chars]
            cut = max(window.rfind(", "), window.rfind("; "))
            if cut > 0:
                return cut + 1
            space = window.rfind(" ")
            return space if space > 0 else self.max_chars
        return None
