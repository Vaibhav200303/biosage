"""Bounded session memory for six-turn conversational follow-ups."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    max_turns: int = 6
    _turns: list[ConversationTurn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        self._turns.append(ConversationTurn(role=role, content=content.strip()))
        self._turns = self._turns[-self.max_turns :]

    def recent(self) -> list[ConversationTurn]:
        return list(self._turns)

    def as_prompt(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self._turns]

    def clear(self) -> None:
        self._turns.clear()


def memory_from_session(session_state: object, key: str = "biosage_memory", max_turns: int = 6) -> ConversationMemory:
    """Create/reuse memory in Streamlit session state without importing Streamlit."""

    memory = getattr(session_state, "get", lambda _key, _default=None: None)(key)
    if not isinstance(memory, ConversationMemory):
        memory = ConversationMemory(max_turns=max_turns)
        try:
            session_state[key] = memory  # type: ignore[index]
        except Exception:
            pass
    return memory
