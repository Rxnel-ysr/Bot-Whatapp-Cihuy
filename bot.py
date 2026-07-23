#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

from neonize.client import NewClient
from neonize.utils import build_jid
from neonize.events import MessageEv, ConnectedEv, event

from helper import Helper
from arch_agent import (
    OllamaModel,
    SYSTEM,
    DEFAULT_MODEL,
    registry,
    ResponseParser,
    MAX_TOOL_DEPTH,
    DESTRUCTIVE_PATTERNS,
    _safety_check,
)

# ── Configuration ────────────────────────────────────────────────────────────
ALLOWED_GROUP = [
    '120363425335904709'
]
ADMIN_NUMBER    = os.environ.get("ADMIN_NUMBER", "")
ADMIN_JID    = os.environ.get("ADMIN_JID", "")
BOT_SESSION     = os.environ.get("BOT_SESSION", "test-bot")
CMD_PREFIX      = os.environ.get("BOT_PREFIX", ".ai")
CONFIRM_TIMEOUT = int(os.environ.get("BOT_CONFIRM_TIMEOUT", "60"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("whatsapp_bot")

client = NewClient(BOT_SESSION)
h = Helper(client)
parser = ResponseParser()

# ── Per-chat state ───────────────────────────────────────────────────────────

@dataclass
class ChatSession:
    model: OllamaModel
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_command: str | None = None
    pending_since: float = 0.0

sessions: dict[str, ChatSession] = {}
_sessions_lock = threading.Lock()


def _get_session(chat_id: str) -> ChatSession:
    with _sessions_lock:
        if chat_id not in sessions:
            model = OllamaModel(model=DEFAULT_MODEL)
            model.history = [{"role": "system", "content": SYSTEM}]
            sessions[chat_id] = ChatSession(model=model)
        return sessions[chat_id]


def _is_authorized(orig) -> bool:
    """Only ever act on messages you sent yourself, in the admin chat."""
    try:
        return orig.chat.jid.User == orig.sender.jid.User
    except AttributeError:
        return False

def _is_allowed_group(orig) -> bool:
    try:
        return orig.isGroup and orig.chat.jid.User in ALLOWED_GROUP
    except AttributeError:
        return False


def _reply(orig, text: str) -> None:
    try:
        if orig.isGroup and orig.sender.jid.User:
            log.info(f"Send group: {orig.chat.phone.User}")
            client.send_message(build_jid(orig.chat.jid.User, 'g.us'), f"@{orig.pushName}\n\n{text}")
        elif not orig.isGroup and orig.chat.phone.User:
            log.info(f"Send: {orig.chat.phone.User}")
            client.send_message(build_jid(orig.chat.phone.User), text)
        elif(orig.sender.phone.User.__len__() == 0):
            log.info(f"Send: {ADMIN_NUMBER} - Self message")
            client.send_message(build_jid(ADMIN_NUMBER), text)
    except Exception:
        log.exception("Failed to send reply to %s", chat_id)


# ── Confirmation-aware tool dispatch ─────────────────────────────────────────
# arch_agent's shell tool blocks on input() for destructive commands. Over a
# chat transport that just hangs, so destructive commands are intercepted
# here and turned into an explicit chat confirmation instead.

def _dispatch_tool(chat_id: str, session: ChatSession, tool: str, args: dict) -> str:
    if tool == "shell":
        cmd = args.get("cmd")
        cmd_str = cmd if isinstance(cmd, str) else " && ".join(cmd or [])

        blocked = _safety_check(cmd_str)
        if blocked:
            return blocked

        if DESTRUCTIVE_PATTERNS.search(cmd_str):
            session.pending_command = cmd_str
            session.pending_since = time.monotonic()
            _reply(
                chat_id,
                f"Destructive command needs confirmation:\n{cmd_str}\n"
                f"Reply .yes or .no within {CONFIRM_TIMEOUT}s.",
            )
            return "[PENDING_CONFIRMATION] Waiting for the user to reply .yes or .no in chat."

    return registry.run(tool, args)


# ── Core processing (always runs off the event-loop thread) ─────────────────

def _drain_tool_calls(chat_id: str, session: ChatSession, current) -> None:
    depth = 0
    while current.tool_call and depth < MAX_TOOL_DEPTH:
        depth += 1
        call = current.tool_call
        log.info("tool[%d] %s(%s)", depth, call.tool, call.args)

        result = _dispatch_tool(chat_id, session, call.tool, call.args)
        session.model.inject_tool_result(result, tool_name=call.tool)

        if result.startswith("[PENDING_CONFIRMATION]"):
            return  # resume once .yes / .no arrives

        try:
            raw = session.model.followup()
        except Exception as e:
            log.exception("Follow-up call failed")
            _reply(chat_id, f"[ERROR] Follow-up failed: {e}")
            return

        current = parser.parse(raw)
        if current.message:
            _reply(chat_id, current.message)


def _process(orig, text: str) -> None:
    session = _get_session(orig.sender.phone.User)
    with session.lock:
        try:
            raw = session.model.chat(text)
        except Exception as e:
            log.exception("Model call failed")
            _reply(orig, f"[ERROR] Model call failed: {e}")
            return

        current = parser.parse(raw)
        if current.message:
            _reply(orig, current.message)
        # _drain_tool_calls(orig.sender.phone.User, session, current)


def _resume_pending(chat_id: str, session: ChatSession, approved: bool) -> None:
    with session.lock:
        cmd = session.pending_command
        session.pending_command = None
        if cmd is None:
            return

        result = registry.run("shell", {"cmd": cmd}) if approved else f"[SKIPPED] User declined via chat: {cmd}"
        session.model.inject_tool_result(result, tool_name="shell")

        try:
            raw = session.model.followup()
        except Exception as e:
            log.exception("Follow-up call failed")
            _reply(chat_id, f"[ERROR] Follow-up failed: {e}")
            return

        current = parser.parse(raw)
        if current.message:
            _reply(chat_id, current.message)
        _drain_tool_calls(chat_id, session, current)


def _expire_pending_loop() -> None:
    while True:
        time.sleep(5)
        now = time.monotonic()
        with _sessions_lock:
            items = list(sessions.items())
        for chat_id, session in items:
            if session.pending_command and now - session.pending_since > CONFIRM_TIMEOUT:
                _reply(chat_id, "Confirmation timed out — skipping the pending command.")
                threading.Thread(target=_resume_pending, args=(chat_id, session, False), daemon=True).start()


# ── Event handlers ───────────────────────────────────────────────────────────

@client.event(ConnectedEv)
def on_connected(client: NewClient, ev: ConnectedEv):
    log.info("Connected to WhatsApp (session=%s, admin=%s)", BOT_SESSION, ADMIN_NUMBER)


@client.event(MessageEv)
def on_message(client: NewClient, ev: MessageEv):
    try:
        orig = h.getOrigin(ev)
    except Exception:
        log.exception("Failed to parse incoming event")
        return

    if not _is_authorized(orig) and not _is_allowed_group(orig):
        return
    
    text = (h.getMsgStr(ev) or "").strip()
    chat_id = orig.sender.phone.User
    session = sessions.get(chat_id)

    # A pending destructive-command confirmation takes priority over everything.
    if session and session.pending_command:
        low = text.lower()
        if low in (".yes", "yes"):
            threading.Thread(target=_resume_pending, args=(chat_id, session, True), daemon=True).start()
        elif low in (".no", "no"):
            threading.Thread(target=_resume_pending, args=(chat_id, session, False), daemon=True).start()
        else:
            _reply(chat_id, f"Still waiting on confirmation for:\n{session.pending_command}\nReply .yes or .no.")
        return

    low = text.lower()

    if low == ".reset":
        _get_session(chat_id).model.reset()
        _reply(orig, "[RESET] Conversation cleared.")
        return

    if low == ".help":
        _reply(
            orig,
            f"Commands:\n{CMD_PREFIX} <prompt>  — ask the agent\n"
            f".reset  — clear this chat's history\n"
            f".help   — this message",
        )
        return

    if not low.startswith(CMD_PREFIX):
        return

    prompt = text[len(CMD_PREFIX):].strip()
    if not prompt:
        _reply(orig, f"Usage: {CMD_PREFIX} <your message>")
        return

    threading.Thread(target=_process, args=(orig, prompt), daemon=True).start()


if __name__ == "__main__":
    threading.Thread(target=_expire_pending_loop, daemon=True).start()
    client.connect()
    event.wait()
