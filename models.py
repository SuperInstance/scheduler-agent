#!/usr/bin/env python3
"""
Shared data models for the Scheduler Agent.

Contains all enums, dataclasses, and constants used by both
scheduler.py and task_queue.py to avoid circular imports.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Dict, List, Optional

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model Tiers
# ---------------------------------------------------------------------------

class ModelTier(IntEnum):
    """Ordered model capability tiers — higher value = more capable."""
    CHEAP = 0       # glm-4.7-flash — bulk spray
    GOOD = 1        # glm-4.7 / deepseek-chat — solid mid-tier
    RUNNER = 2      # glm-5-turbo — daily driver
    EXPERT = 3      # glm-5.1 — complex reasoning
    REASONER = 4    # deepseek-reasoner — deep thinking


# ---------------------------------------------------------------------------
# Model Profile
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Metadata for a model in the fleet."""
    name: str
    tier: ModelTier
    cost_per_1k_tokens: float
    speed_tokens_per_sec: float
    quality_score: float          # 0.0 – 1.0
    best_for: List[str] = field(default_factory=list)
    max_concurrent: int = 10      # parallel request limit


# ---------------------------------------------------------------------------
# Schedule Slot
# ---------------------------------------------------------------------------

@dataclass
class ScheduleSlot:
    """A time-of-day window with a model assignment."""
    start_hour: int               # UTC hour (inclusive)
    end_hour: int                 # UTC hour (exclusive)
    model: str
    reason: str
    rooms: List[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Scheduled Task
# ---------------------------------------------------------------------------

@dataclass
class ScheduledTask:
    """A task that has been assigned a model and time slot."""
    task_id: str
    room_id: str
    description: str
    required_tier: ModelTier
    estimated_tokens: int
    deadline: Optional[float] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_model: Optional[str] = None
    status: str = "pending"       # pending | scheduled | running | done | failed
    agent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Fleet Model Registry
# ---------------------------------------------------------------------------

FLEET_MODELS: Dict[str, ModelProfile] = {
    "glm-4.7-flash": ModelProfile(
        "glm-4.7-flash", ModelTier.CHEAP,
        cost_per_1k_tokens=0.0001, speed_tokens_per_sec=500,
        quality_score=0.60, best_for=["bulk", "sweep"],
    ),
    "glm-4.7": ModelProfile(
        "glm-4.7", ModelTier.GOOD,
        cost_per_1k_tokens=0.0005, speed_tokens_per_sec=200,
        quality_score=0.75, best_for=["coding", "review"],
    ),
    "glm-5-turbo": ModelProfile(
        "glm-5-turbo", ModelTier.RUNNER,
        cost_per_1k_tokens=0.002, speed_tokens_per_sec=150,
        quality_score=0.85, best_for=["coding", "planning"],
    ),
    "glm-5.1": ModelProfile(
        "glm-5.1", ModelTier.EXPERT,
        cost_per_1k_tokens=0.005, speed_tokens_per_sec=80,
        quality_score=0.95, best_for=["architecture", "strategy"],
    ),
    "deepseek-reasoner": ModelProfile(
        "deepseek-reasoner", ModelTier.REASONER,
        cost_per_1k_tokens=0.003, speed_tokens_per_sec=30,
        quality_score=0.90, best_for=["deep_thinking"],
    ),
    "deepseek-chat": ModelProfile(
        "deepseek-chat", ModelTier.GOOD,
        cost_per_1k_tokens=0.0003, speed_tokens_per_sec=300,
        quality_score=0.70, best_for=["iteration", "spreading"],
    ),
}
