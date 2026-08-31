from __future__ import annotations

import importlib.metadata
from typing import Any

from resume_bench.providers.base import PipelineSpec, Provider, ProviderConfigError

_REGISTRY: dict[str, type[Provider]] = {}


def register_provider(name: str):
    """Decorator to register a provider class."""
    def deco(cls: type[Provider]) -> type[Provider]:
        if name in _REGISTRY:
            raise ValueError(f"provider '{name}' already registered")

        _REGISTRY[name] = cls
        return cls

    return deco


def load_plugins() -> None:
    """Load third-party providers from entry points."""
    for ep in importlib.metadata.entry_points(group="resume_bench.providers"):
        ep.load()


def create_provider(spec: PipelineSpec, settings: Any = None) -> Provider:
    """Create a provider instance from a pipeline spec."""
    if spec.provider_name not in _REGISTRY:
        load_plugins()

    if spec.provider_name not in _REGISTRY:
        raise ProviderConfigError(
            f"no provider '{spec.provider_name}'; known: {sorted(_REGISTRY)}"
        )

    return _REGISTRY[spec.provider_name](spec, settings)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return sorted(_REGISTRY.keys())


def list_pipelines() -> list[PipelineSpec]:
    """List all defined pipeline specs."""
    from resume_bench.providers.pipelines import PIPELINES

    return list(PIPELINES)
