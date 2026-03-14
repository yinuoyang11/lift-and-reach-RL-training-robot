#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of runners for environment-agent interaction."""

from .on_policy_runner import OnPolicyRunner

try:
    from .distillation_runner import DistillationRunner
except ImportError:  # pragma: no cover - optional in lightweight overlays
    DistillationRunner = None

__all__ = ["OnPolicyRunner", "DistillationRunner"]
