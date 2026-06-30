"""Fast, deterministic prompt-injection detection that runs before any LLM call.

This exists as a separate layer from the LLM-based scope check in slots.py
because an injection attempt's whole point is to make an LLM ignore its
instructions — so the first line of defense against it must not depend on an
LLM being faithfully obedient. Off-topic / legal-advice judgment is nuanced
and is handled by the LLM interpret call instead; this module only catches
clear-cut manipulation attempts.
"""
import re

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above|earlier)\s*instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above|earlier)",
    r"forget (all|your|the)?\s*(previous|prior)?\s*instructions",
    r"you are now\b",
    r"new instructions\s*:",
    r"system prompt",
    r"reveal (your|the) (system|instructions|prompt)",
    r"what (is|are) your (instructions|system prompt|rules)",
    r"pretend (you('?re| are)|to be) (a|an)\b",
    r"jailbreak",
    r"\bDAN\b",
    r"developer mode",
    r"override your (rules|instructions|guidelines)",
    r"do anything now",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def is_injection_attempt(text: str) -> bool:
    return any(pattern.search(text) for pattern in _COMPILED)
