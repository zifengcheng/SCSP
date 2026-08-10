"""Composable baseline input and hidden-state transformations."""

from .echo import build_echo_input
from .htp import HTPRewirer, build_htp_layout, ensure_htp_tokens

__all__ = ["HTPRewirer", "build_echo_input", "build_htp_layout", "ensure_htp_tokens"]
