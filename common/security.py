"""Shared-secret authentication between the coordinator and campus nodes."""
from __future__ import annotations

import hmac
import os

TOKEN_HEADER = "X-FedTrust-Token"
DEFAULT_TOKEN = "fedtrust-dev-token"


def get_token() -> str:
    return os.environ.get("FEDTRUST_TOKEN", DEFAULT_TOKEN)


def token_ok(presented: str | None) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(presented.encode(), get_token().encode())
