"""Compiled regexes for the secret-pattern redactor (spec/05 §Redaction) and the
data-pipeline secret scanner (spec/06 §T4). Closed set; adding a pattern is an
RFC amendment. ``tests/test_secret_patterns.py`` enforces the closure with
≥3 positive + ≥3 negative fixtures per pattern.
"""

from __future__ import annotations

import re

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("github_oauth", re.compile(r"\bgho_[A-Za-z0-9]{36}\b")),
    ("github_app", re.compile(r"\b(?:ghu_|ghs_)[A-Za-z0-9]{36}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    (
        "ssh_private_key",
        re.compile(
            r"-----BEGIN (?:OPENSSH|RSA|DSA|EC|PGP|ENCRYPTED) PRIVATE KEY-----"
            r"[\s\S]*?-----END (?:OPENSSH|RSA|DSA|EC|PGP|ENCRYPTED) PRIVATE KEY-----"
        ),
    ),
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
