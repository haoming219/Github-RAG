import sys, pathlib, asyncio, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock
from agent.session import SessionManager


@pytest.fixture
def manager():
    return SessionManager(ttl_seconds=2)


def test_get_creates_session(manager):
    agent = manager.get_or_create("sid-1", factory=lambda: MagicMock())
    assert agent is not None


def test_get_returns_same_instance(manager):
    factory = MagicMock(side_effect=lambda: MagicMock())
    a1 = manager.get_or_create("sid-2", factory=factory)
    a2 = manager.get_or_create("sid-2", factory=factory)
    assert a1 is a2
    assert factory.call_count == 1


def test_expired_session_recreated(manager):
    factory = MagicMock(side_effect=lambda: MagicMock())
    a1 = manager.get_or_create("sid-3", factory=factory)
    manager._sessions["sid-3"]["last_active"] = time.time() - 10
    a2 = manager.get_or_create("sid-3", factory=factory)
    assert a1 is not a2
    assert factory.call_count == 2


def test_touch_updates_last_active(manager):
    manager.get_or_create("sid-4", factory=lambda: MagicMock())
    old_ts = manager._sessions["sid-4"]["last_active"]
    time.sleep(0.01)
    manager.touch("sid-4")
    assert manager._sessions["sid-4"]["last_active"] > old_ts
