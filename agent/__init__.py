"""Agent-based poetry generation package."""

__all__ = ["generate_poem_with_agent"]


def generate_poem_with_agent(*args, **kwargs):
    from .api import generate_poem_with_agent as _generate_poem_with_agent

    return _generate_poem_with_agent(*args, **kwargs)
