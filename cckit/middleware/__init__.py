"""Agent execution middleware — pluggable cross-cutting concerns.

Usage::

    from cckit.middleware import Middleware, RetryMiddleware, ConcurrencyMiddleware

    runner = Runner(
        middlewares=[ConcurrencyMiddleware(), RetryMiddleware(max_retries=3)],
    )
"""

from cckit.middleware.base import Middleware, SdkQueryFunc
from cckit.middleware.concurrency import ConcurrencyMiddleware
from cckit.middleware.logging import LoggingMiddleware
from cckit.middleware.retry import RetryMiddleware

__all__ = [
    "ConcurrencyMiddleware",
    "LoggingMiddleware",
    "Middleware",
    "RetryMiddleware",
    "SdkQueryFunc",
]
