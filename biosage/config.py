"""Configuration and secret access; no credentials are hard-coded."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "BioSage"
    gemini_api_key: str | None = None
    gemini_model: str = Field(default="gemini-3.5-flash-lite")
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    retrieval_min_score: float = Field(default=0.05, ge=0.01, le=1)
    max_memory_turns: int = Field(default=6, ge=1, le=20)


def _streamlit_secret(name: str) -> str | None:
    """Read a Streamlit secret if Streamlit is installed and running in its context."""

    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    key = os.getenv("GEMINI_API_KEY") or _streamlit_secret("GEMINI_API_KEY")
    return Settings(
        app_name=os.getenv("BIOSAGE_APP_NAME", "BioSage"),
        gemini_api_key=key,
        gemini_model=os.getenv("BIOSAGE_GEMINI_MODEL", "gemini-3.5-flash-lite"),
        retrieval_top_k=int(os.getenv("BIOSAGE_RETRIEVAL_TOP_K", "6")),
        retrieval_min_score=float(os.getenv("BIOSAGE_RETRIEVAL_MIN_SCORE", "0.05")),
    )
