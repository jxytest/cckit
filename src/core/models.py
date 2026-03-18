"""Shared Pydantic model base with sensible defaults."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CustomModel(BaseModel):
    """Base model for all domain objects.

    Features:
    - Datetime fields serialize to ISO-8601 strings.
    - Extra fields are forbidden to catch typos early.
    - Enum values are used (not names) in serialization.
    """

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )
