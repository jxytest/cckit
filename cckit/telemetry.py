"""Lightweight OpenTelemetry bootstrap for cckit.

Reads standard ``OTEL_EXPORTER_OTLP_*`` environment variables and, when
present, configures a global :class:`TracerProvider` with an OTLP/HTTP
exporter.  If the ``opentelemetry`` packages are not installed the module
gracefully degrades to a no-op tracer with zero overhead.

Usage::

    from cckit.telemetry import setup_telemetry, get_tracer

    setup_telemetry()                    # call once at process startup
    tracer = get_tracer("cckit.engine")  # per-module tracer
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ENDPOINT_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)

_setup_done = False


def _otel_enabled() -> bool:
    return any(os.environ.get(v) for v in _ENDPOINT_VARS)


def setup_telemetry() -> None:
    """Configure the global TracerProvider if OTLP env vars are set.

    Safe to call multiple times — only the first invocation takes effect.
    """
    global _setup_done
    if _setup_done:
        return

    if not _otel_enabled():
        return

    _setup_done = True

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.debug("opentelemetry packages not installed — OTEL disabled")
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "cckit")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OTEL tracing enabled — service=%s", service_name)

    # Auto-instrument httpx so requests to the bridge carry traceparent headers,
    # enabling gen_ai.chat spans to be children of cckit.agent.execute.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.debug("httpx instrumentation enabled")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-httpx not installed — skipping")


def get_tracer(name: str = "cckit") -> Any:
    """Return an OpenTelemetry tracer (or a no-op stub)."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopSpan:
    """Minimal stub that satisfies the ``with tracer.start_as_current_span()`` protocol."""

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def set_attribute(self, _key: str, _value: Any) -> None:
        pass

    def add_event(self, _name: str, _attributes: Any = None) -> None:
        pass

    def record_exception(self, _exc: BaseException) -> None:
        pass

    def set_status(self, _status: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class _NoopTracer:
    """Returned when ``opentelemetry`` is not installed."""

    def start_as_current_span(self, _name: str, **_kw: Any) -> _NoopSpan:
        return _NoopSpan()
