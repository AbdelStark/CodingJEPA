"""Compiled regexes for the secret-pattern redactor (spec/05 §Redaction).

Minimal v0.1 set; the full curated table is the deliverable of #99. The patterns
below cover the four families spec/05 names: AWS access keys, GitHub PATs, JWTs,
and bare ≥32-char hex strings (other than the known-hash whitelist).
"""

from __future__ import annotations

import re

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("github_oauth", re.compile(r"\bgho_[A-Za-z0-9]{36}\b")),
    ("github_app", re.compile(r"\b(?:ghu_|ghs_)[A-Za-z0-9]{36}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    ("hex_secret", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
)

EMAIL_PATTERN: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

KNOWN_HASH_FIELDS: frozenset[str] = frozenset(
    {
        "git_sha",
        "commit_sha",
        "manifest_hash",
        "checkpoint_hash",
        "index_id",
        "tokenizer_hash",
        "model_hash",
        "config_hash",
        "data_manifest_hash",
        "source_hash",
    }
)

__all__ = ["EMAIL_PATTERN", "KNOWN_HASH_FIELDS", "SECRET_PATTERNS"]
