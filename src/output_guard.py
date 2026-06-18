from __future__ import annotations

import re

PII_PATTERNS: list[tuple[str, str, str]] = [
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "EMAIL", "[EMAIL REDACTED]"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE", "[PHONE REDACTED]"),
    (r"\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b", "SSN", "[SSN REDACTED]"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "CC", "[CC REDACTED]"),
    (r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b", "CC_DASHED", "[CC REDACTED]"),
    (r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "IP", "[IP REDACTED]"),
]


def redact_pii(text: str) -> tuple[str, list[dict[str, str]]]:
    if not text:
        return text, []

    redacted = text
    findings: list[dict[str, str]] = []

    for pattern, label, replacement in PII_PATTERNS:
        matches = re.finditer(pattern, redacted)
        for m in matches:
            findings.append({"type": label, "match": m.group()})
        redacted = re.sub(pattern, replacement, redacted)

    return redacted, findings


def guard_agent_output(
    summary: str | None,
    reviewer_questions: list[str] | None = None,
) -> tuple[str | None, list[str] | None, list[dict[str, str]]]:
    all_findings: list[dict[str, str]] = []

    if summary:
        summary, findings = redact_pii(summary)
        all_findings.extend(findings)

    guarded_questions = None
    if reviewer_questions:
        guarded_questions = []
        for q in reviewer_questions:
            q_redacted, findings = redact_pii(q)
            guarded_questions.append(q_redacted)
            all_findings.extend(findings)

    return summary, guarded_questions, all_findings
