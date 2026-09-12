"""Native compatibility import for canonical deployment profiles."""

from h3serve.deployment_profiles import (
    ExecutionGraph,
    INT8_16GB_BACKEND,
    INT8_24GB_BACKEND,
    RESOURCE_BACKENDS,
    ResourceBackendDefinition,
    ResourceBackendId,
    W4A8_16GB_BACKEND,
    W4A8_24GB_BACKEND,
    W4A8_8GB_BACKEND,
    WeightTier,
    get_resource_backend,
)

__all__ = [
    "ExecutionGraph",
    "INT8_16GB_BACKEND",
    "INT8_24GB_BACKEND",
    "RESOURCE_BACKENDS",
    "ResourceBackendDefinition",
    "ResourceBackendId",
    "W4A8_16GB_BACKEND",
    "W4A8_24GB_BACKEND",
    "W4A8_8GB_BACKEND",
    "WeightTier",
    "get_resource_backend",
]
