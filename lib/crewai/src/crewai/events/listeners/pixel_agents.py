"""Pixel Agents integration: show a running crew as characters in a pixel-art office.

Pixel Agents (VS Code extension or ``npx pixel-agents``) runs a local server
that renders AI agents as animated characters. This listener forwards CrewAI
events to it, so every agent of a crew becomes its own character that types
while it works on a task, shows the tool it is using, and leaves when the crew
finishes.

Opt in from your own code before kicking off a crew::

    from crewai.events.listeners.pixel_agents import PixelAgentsListener

    PixelAgentsListener()

The listener discovers running servers from ``~/.pixel-agents/servers/*.json``
(falling back to ``~/.pixel-agents/server.json``) and POSTs each event to
``http://127.0.0.1:<port>/api/hooks/crewai`` with the server's bearer token.
Delivery is best-effort and runs on a background thread: if no server is
running, events are dropped and the crew is unaffected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
from typing import Any
import urllib.request

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


PROVIDER_ID = "crewai"
HOOK_PATH = f"/api/hooks/{PROVIDER_ID}"
REQUEST_TIMEOUT_S = 2.0
TASK_DESCRIPTION_MAX_LENGTH = 200

# The server is always on loopback; never route it through HTTP(S)_PROXY.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _pixel_agents_dir() -> Path:
    return Path(os.environ.get("PIXEL_AGENTS_HOME", Path.home() / ".pixel-agents"))


def _is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_server(path: Path) -> tuple[int, str] | None:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None
    port, token = entry.get("port"), entry.get("token")
    if not isinstance(port, int) or not isinstance(token, str):
        return None
    if "pid" in entry and not _is_alive(entry["pid"]):
        return None
    return port, token


def discover_servers() -> list[tuple[int, str]]:
    """Return ``(port, token)`` for every live Pixel Agents server."""
    base = _pixel_agents_dir()
    servers: list[tuple[int, str]] = []
    try:
        registry = sorted((base / "servers").glob("*.json"))
    except OSError:
        registry = []
    for path in registry:
        server = _read_server(path)
        if server and server not in servers:
            servers.append(server)
    if not servers:
        legacy = _read_server(base / "server.json")
        if legacy:
            servers.append(legacy)
    return servers


def _session_id(agent_id: Any) -> str | None:
    return f"crewai-{agent_id}" if agent_id else None


class PixelAgentsListener(BaseEventListener):
    """Forwards crew, agent and tool events to running Pixel Agents servers."""

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._open_sessions: set[str] = set()
        threading.Thread(
            target=self._deliver_forever, name="pixel-agents", daemon=True
        ).start()
        super().__init__()

    # ── delivery ──

    def _send(self, payload: dict[str, Any]) -> None:
        self._queue.put(payload)

    def _deliver_forever(self) -> None:
        while True:
            payload = self._queue.get()
            body = json.dumps(payload, default=str).encode("utf-8")
            for port, token in discover_servers():
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}{HOOK_PATH}",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
                try:
                    with _opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
                        response.read()
                except Exception:  # noqa: S112 - best-effort, never break the crew
                    continue

    # ── session bookkeeping ──

    def _ensure_session(self, session_id: str, agent_role: str | None) -> None:
        with self._lock:
            if session_id in self._open_sessions:
                return
            self._open_sessions.add(session_id)
        self._send(
            {
                "hook_event_name": "SessionStart",
                "session_id": session_id,
                "cwd": os.getcwd(),
                "agent_role": agent_role,
            }
        )

    def _end_all_sessions(self, reason: str) -> None:
        with self._lock:
            sessions, self._open_sessions = self._open_sessions, set()
        for session_id in sessions:
            self._send(
                {
                    "hook_event_name": "SessionEnd",
                    "session_id": session_id,
                    "reason": reason,
                }
            )

    # ── event bus wiring ──

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(_: Any, event: AgentExecutionStartedEvent) -> None:
            session_id = _session_id(getattr(event.agent, "id", None))
            if not session_id:
                return
            self._ensure_session(session_id, getattr(event.agent, "role", None))
            description = str(getattr(event.task, "description", "") or "")
            self._send(
                {
                    "hook_event_name": "PreToolUse",
                    "session_id": session_id,
                    "tool_name": "Task",
                    "tool_input": {
                        "description": description[:TASK_DESCRIPTION_MAX_LENGTH]
                    },
                }
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def on_agent_finished(
            _: Any, event: AgentExecutionCompletedEvent | AgentExecutionErrorEvent
        ) -> None:
            session_id = _session_id(getattr(event.agent, "id", None))
            if not session_id:
                return
            self._send({"hook_event_name": "PostToolUse", "session_id": session_id})
            self._send({"hook_event_name": "Stop", "session_id": session_id})

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(_: Any, event: ToolUsageStartedEvent) -> None:
            session_id = _session_id(event.agent_id)
            if not session_id:
                return
            self._ensure_session(session_id, event.agent_role)
            self._send(
                {
                    "hook_event_name": "PreToolUse",
                    "session_id": session_id,
                    "tool_name": event.tool_name,
                }
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_finished(
            _: Any, event: ToolUsageFinishedEvent | ToolUsageErrorEvent
        ) -> None:
            session_id = _session_id(event.agent_id)
            if session_id:
                self._send({"hook_event_name": "PostToolUse", "session_id": session_id})

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(_: Any, __: CrewKickoffCompletedEvent) -> None:
            self._end_all_sessions("exit")

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(_: Any, __: CrewKickoffFailedEvent) -> None:
            self._end_all_sessions("error")
