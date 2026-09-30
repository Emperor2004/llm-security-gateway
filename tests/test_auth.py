# test auth & RBAC
import pytest
from app.auth.rbac import (
    AuthError,
    PermissionError_,
    Principal,
    authenticate,
    authorize,
)


def test_authenticate_valid_keys():
    principal_admin = authenticate("demo-admin-key")
    assert principal_admin.role == "admin"
    assert principal_admin.has_permission("chat")
    assert principal_admin.has_permission("admin:view_stats")
    assert principal_admin.has_permission("admin:manage_keys")

    principal_user = authenticate("demo-user-key")
    assert principal_user.role == "user"
    assert principal_user.has_permission("chat")
    assert not principal_user.has_permission("admin:view_stats")

    principal_readonly = authenticate("demo-readonly-key")
    assert principal_readonly.role == "readonly"
    assert not principal_readonly.has_permission("chat")


def test_authenticate_missing_key_raises_autherror():
    with pytest.raises(AuthError, match="Missing API key"):
        authenticate(None)

    with pytest.raises(AuthError, match="Missing API key"):
        authenticate("")


def test_authenticate_invalid_key_raises_autherror():
    with pytest.raises(AuthError, match="Invalid API key"):
        authenticate("non-existent-key")


def test_authorize_scope_enforcement():
    admin = Principal(api_key="demo-admin-key", role="admin")
    user = Principal(api_key="demo-user-key", role="user")
    readonly = Principal(api_key="demo-readonly-key", role="readonly")

    # Admin has all permissions
    authorize(admin, "chat")
    authorize(admin, "admin:view_stats")

    # User only has chat
    authorize(user, "chat")
    with pytest.raises(PermissionError_, match="lacks permission 'admin:view_stats'"):
        authorize(user, "admin:view_stats")

    # Readonly has no permissions
    with pytest.raises(PermissionError_, match="lacks permission 'chat'"):
        authorize(readonly, "chat")


def test_principal_id_is_non_reversible():
    principal = Principal(api_key="secret-api-key-12345", role="user")
    pid = principal.id
    assert pid.startswith("user:")
    assert "secret" not in pid
    assert len(pid) > 5
