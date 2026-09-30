# rbac
"""
Minimal API-key authentication and role-based access control.

Roles map to a fixed permission set. Keys are provided via
Settings.api_keys() (env var API_KEYS_JSON), never hardcoded.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.config import get_settings

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"chat", "admin:view_stats", "admin:manage_keys"},
    "user": {"chat"},
    "readonly": set(),
}


class AuthError(Exception):
    """Raised when an API key is missing or unrecognized."""


class PermissionError_(Exception):
    """Raised when a principal lacks a required permission."""


@dataclass(frozen=True)
class Principal:
    api_key: str
    role: str

    @property
    def id(self) -> str:
        # Stable, non-reversible identifier used for rate limiting / logging.
        # Uses sha256 to ensure consistency across multiple workers and restarts.
        digest = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:6]
        return f"{self.role}:{digest}"

    def has_permission(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, set())


def authenticate(api_key: str | None) -> Principal:
    if not api_key:
        raise AuthError("Missing API key")
    keys = get_settings().api_keys()
    role = keys.get(api_key)
    if role is None:
        raise AuthError("Invalid API key")
    if role not in ROLE_PERMISSIONS:
        raise AuthError(f"Unknown role '{role}' for API key")
    return Principal(api_key=api_key, role=role)


def authorize(principal: Principal, permission: str) -> None:
    if not principal.has_permission(permission):
        raise PermissionError_(f"Role '{principal.role}' lacks permission '{permission}'")
