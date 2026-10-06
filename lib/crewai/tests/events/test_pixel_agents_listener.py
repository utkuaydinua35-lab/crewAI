"""Tests for the Pixel Agents listener: crew events reach a local office server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from crewai import Agent, Task
from crewai.events import crewai_event_bus
from crewai.events.listeners.pixel_agents import PixelAgentsListener, discover_servers
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import CrewKickoffCompletedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


# The fake office server listens on loopback.
pytestmark = pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$"])

TOKEN = "test-token"


@pytest.fixture
def office(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake Pixel Agents server registered under a temporary PIXEL_AGENTS_HOME."""
    received: list[tuple[str, str | None, dict[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            received.append(
                (
                    self.path,
                    self.headers.get("Authorization"),
                    json.loads(self.rfile.read(length)),
                )
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (tmp_path / "servers").mkdir()
    (tmp_path / "servers" / "a.json").write_text(
        json.dumps({"port": server.server_port, "pid": os.getpid(), "token": TOKEN})
    )
    monkeypatch.setenv("PIXEL_AGENTS_HOME", str(tmp_path))
    yield received
    server.shutdown()


def _wait_for(received: list[Any], count: int) -> None:
    deadline = time.monotonic() + 5
    while len(received) < count and time.monotonic() < deadline:
        time.sleep(0.02)


def test_discover_servers_skips_dead_and_malformed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = tmp_path / "servers"
    servers.mkdir()
    (servers / "live.json").write_text(
        json.dumps({"port": 1, "pid": os.getpid(), "token": "t"})
    )
    (servers / "dead.json").write_text(
        json.dumps({"port": 2, "pid": 2**22 + 12345, "token": "t"})
    )
    (servers / "junk.json").write_text("{not json")
    monkeypatch.setenv("PIXEL_AGENTS_HOME", str(tmp_path))
    assert discover_servers() == [(1, "t")]


def test_discover_servers_without_any_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIXEL_AGENTS_HOME", str(tmp_path))
    assert discover_servers() == []


def test_crew_run_is_forwarded_as_hook_events(office: list[Any]) -> None:
    agent = Agent(role="Researcher", goal="Find facts", backstory="Curious")
    task = Task(description="Research pixel art", expected_output="Notes", agent=agent)

    with crewai_event_bus.scoped_handlers():
        PixelAgentsListener()
        crewai_event_bus.emit(
            agent,
            AgentExecutionStartedEvent(
                agent=agent, task=task, tools=[], task_prompt="Research pixel art"
            ),
        )
        crewai_event_bus.flush()
        tool_fields = {
            "tool_name": "search",
            "tool_args": {},
            "agent_id": str(agent.id),
            "agent_role": agent.role,
        }
        crewai_event_bus.emit(agent, ToolUsageStartedEvent(**tool_fields))
        crewai_event_bus.flush()
        now = ToolUsageStartedEvent(**tool_fields).timestamp
        crewai_event_bus.emit(
            agent,
            ToolUsageFinishedEvent(
                **tool_fields, started_at=now, finished_at=now, output="x"
            ),
        )
        crewai_event_bus.flush()
        crewai_event_bus.emit(
            agent, AgentExecutionCompletedEvent(agent=agent, task=task, output="done")
        )
        crewai_event_bus.flush()
        crewai_event_bus.emit(
            None, CrewKickoffCompletedEvent(crew_name="crew", output="done")
        )
        crewai_event_bus.flush()

    _wait_for(office, 8)
    assert {path for path, _, _ in office} == {"/api/hooks/crewai"}
    assert {auth for _, auth, _ in office} == {f"Bearer {TOKEN}"}
    events = [body for _, _, body in office]
    assert [e["hook_event_name"] for e in events] == [
        "SessionStart",
        "PreToolUse",
        "PreToolUse",
        "PostToolUse",
        "PostToolUse",
        "Stop",
        "SessionEnd",
    ]
    assert {e["session_id"] for e in events} == {f"crewai-{agent.id}"}
    assert events[0]["agent_role"] == "Researcher"
    assert events[1]["tool_input"] == {"description": "Research pixel art"}
    assert events[2]["tool_name"] == "search"
