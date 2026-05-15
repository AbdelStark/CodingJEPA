"""≥3 positive + ≥3 negative fixtures per pattern. Spec/05 §Redaction, spec/06 §T4."""

from __future__ import annotations

import pytest

from codingjepa.safety.secret_patterns import SECRET_PATTERNS

# ---- Fixtures: (pattern_name, sample_text, should_match) --------------------

POSITIVES: list[tuple[str, str]] = [
    # AWS access key IDs (AKIA / ASIA prefix + 16 chars).
    ("aws_access_key_id", "key=AKIAIOSFODNN7EXAMPLE"),
    ("aws_access_key_id", "creds: ASIA0123456789ABCDEF"),
    ("aws_access_key_id", "AKIA1234567890ABCDEF in env"),
    # GitHub PATs.
    ("github_pat", "Bearer ghp_" + "a" * 36),
    ("github_pat", "leaked: ghp_" + "Z" * 36),
    ("github_pat", "ghp_" + "x9" * 18 + " somewhere"),
    # GitHub OAuth.
    ("github_oauth", "gho_" + "a" * 36),
    ("github_oauth", "gho_" + "Z" * 36),
    ("github_oauth", "gho_" + "0" * 36 + " inline"),
    # GitHub App (ghu_ / ghs_).
    ("github_app", "ghu_" + "a" * 36),
    ("github_app", "ghs_" + "Z" * 36),
    ("github_app", "saw ghs_" + "b" * 36 + " in trace"),
    # JWTs.
    ("jwt", "eyJhbGciOi.eyJzdWIiOi.signature_part_here"),
    ("jwt", "Bearer eyJabc.eyJdef.sigxyz"),
    ("jwt", "tok=eyJX.eyJY.ZZZ"),
    # OpenSSH/RSA private key blocks.
    (
        "ssh_private_key",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nblob\n-----END OPENSSH PRIVATE KEY-----",
    ),
    ("ssh_private_key", "-----BEGIN RSA PRIVATE KEY-----\nbase64==\n-----END RSA PRIVATE KEY-----"),
    ("ssh_private_key", "-----BEGIN EC PRIVATE KEY-----\nb64\n-----END EC PRIVATE KEY-----"),
    # Long bare hex (≥ 32 chars).
    ("hex_secret", "deadbeef" * 4),  # 32 chars
    ("hex_secret", "0" * 40),
    ("hex_secret", "f" * 64),
]

NEGATIVES: list[tuple[str, str]] = [
    # AWS-shaped strings that should NOT match.
    ("aws_access_key_id", "AKIA short"),
    ("aws_access_key_id", "no-key here"),
    ("aws_access_key_id", "AKIA1234"),  # too short.
    # GitHub PAT shapes that should NOT match.
    ("github_pat", "ghp_short"),
    ("github_pat", "gph_" + "a" * 36),  # wrong prefix
    ("github_pat", "ghp_" + "a" * 10),  # too short
    # GitHub OAuth shapes that should NOT match.
    ("github_oauth", "gho_short"),
    ("github_oauth", "gho-" + "a" * 36),  # wrong delimiter
    ("github_oauth", "gho_" + "a" * 10),
    # GitHub App shapes that should NOT match.
    ("github_app", "ghu_short"),
    ("github_app", "ghv_" + "a" * 36),  # wrong prefix
    ("github_app", "ghs_short"),
    # JWT shapes that should NOT match.
    ("jwt", "eyJ.no.dotsplit"),
    ("jwt", "no jwt here"),
    ("jwt", "abc.def.ghi"),  # missing eyJ prefix
    # SSH-key shapes that should NOT match.
    ("ssh_private_key", "-----BEGIN CERTIFICATE-----\nfoo\n-----END CERTIFICATE-----"),
    ("ssh_private_key", "-----BEGIN RSA PUBLIC KEY-----\nfoo\n-----END RSA PUBLIC KEY-----"),
    ("ssh_private_key", "no PEM block here"),
    # Hex shapes that should NOT match.
    ("hex_secret", "deadbee"),  # 7 chars
    ("hex_secret", "x" * 64),  # not hex
    ("hex_secret", "abcdef12345"),  # 11 chars, below threshold
]


@pytest.fixture(scope="module")
def patterns_by_name() -> dict[str, object]:
    return {name: pat for name, pat in SECRET_PATTERNS}


@pytest.mark.parametrize("pattern_name, sample", POSITIVES)
def test_positive(patterns_by_name: dict[str, object], pattern_name: str, sample: str) -> None:
    pat = patterns_by_name[pattern_name]
    assert pat.search(sample), f"{pattern_name}: expected to match in {sample!r}"  # type: ignore[union-attr]


@pytest.mark.parametrize("pattern_name, sample", NEGATIVES)
def test_negative(patterns_by_name: dict[str, object], pattern_name: str, sample: str) -> None:
    pat = patterns_by_name[pattern_name]
    assert not pat.search(sample), f"{pattern_name}: did NOT expect a match in {sample!r}"  # type: ignore[union-attr]


def test_three_positive_and_three_negative_per_pattern() -> None:
    """Acceptance gate: every registered pattern carries ≥3 positives + ≥3 negatives."""

    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    for name, _ in POSITIVES:
        pos[name] = pos.get(name, 0) + 1
    for name, _ in NEGATIVES:
        neg[name] = neg.get(name, 0) + 1
    for name, _ in SECRET_PATTERNS:
        assert pos.get(name, 0) >= 3, f"{name}: only {pos.get(name, 0)} positive fixtures"
        assert neg.get(name, 0) >= 3, f"{name}: only {neg.get(name, 0)} negative fixtures"


def test_pattern_names_are_unique() -> None:
    names = [n for n, _ in SECRET_PATTERNS]
    assert len(names) == len(set(names))
