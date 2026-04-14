#!/usr/bin/env python3
"""
Fleet Scheduler — Scheduling as Intelligence

Not one smart model doing everything, but the right model
at the right time in the right room.

Cost optimization: cheap models for bulk work at night,
expensive models for critical decisions during the day.

Extracted and enhanced from flux-lcar-scheduler.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import (
    FLEET_MODELS, ModelTier, ModelProfile, ScheduleSlot, ScheduledTask,
)
from task_queue import TaskQueue, QueuedTask


# ---------------------------------------------------------------------------
# Fleet Scheduler
# ---------------------------------------------------------------------------

class FleetScheduler:
    """Schedule the right model at the right time for the right task.

    Features
    --------
    - Time-based model selection per room
    - Budget tracking with daily caps
    - Priority queue integration with backpressure
    - Dynamic tier adjustment based on load
    - Fair-share scheduling across agents
    - Cost optimization strategies
    """

    def __init__(
        self,
        daily_budget: float = 1.0,
        task_queue: Optional[TaskQueue] = None,
    ) -> None:
        self.schedule: List[ScheduleSlot] = []
        self.task_queue: TaskQueue = task_queue or TaskQueue()
        self.completed: List[ScheduledTask] = []
        self.daily_budget: float = daily_budget
        self.spent: float = 0.0
        self._agent_share: Dict[str, float] = {}   # agent_id → tokens consumed
        self._tier_usage: Dict[ModelTier, int] = {t: 0 for t in ModelTier}
        self._setup_default_schedule()

    # ---- schedule setup ---------------------------------------------------

    def _setup_default_schedule(self) -> None:
        """Build the default 24 h time-of-day schedule optimised for cost."""
        self.schedule = [
            # Night shift — cheap bulk work
            ScheduleSlot(0, 6, "glm-4.7-flash", "night bulk", ["*"]),
            ScheduleSlot(0, 6, "deepseek-chat", "night iteration", ["engineering", "workshop"]),
            # Morning — good models for fresh starts
            ScheduleSlot(6, 10, "glm-5-turbo", "morning driver", ["*"]),
            ScheduleSlot(6, 10, "deepseek-reasoner", "morning deep think", ["ready-room"]),
            # Peak hours — expert for critical decisions
            ScheduleSlot(10, 14, "glm-5.1", "peak expert", ["bridge", "nav"]),
            ScheduleSlot(10, 14, "glm-5-turbo", "peak runner", ["*"]),
            # Afternoon
            ScheduleSlot(14, 18, "glm-5-turbo", "afternoon", ["*"]),
            # Evening — wind down
            ScheduleSlot(18, 22, "glm-4.7", "evening review", ["*"]),
            # Late night — cheap again
            ScheduleSlot(22, 24, "glm-4.7-flash", "late night bulk", ["*"]),
        ]

    def load_schedule(self, slots: List[Dict[str, Any]]) -> None:
        """Replace the schedule from a list of dicts (e.g. parsed YAML)."""
        self.schedule = [
            ScheduleSlot(
                start_hour=s["start_hour"],
                end_hour=s["end_hour"],
                model=s["model"],
                reason=s.get("reason", ""),
                rooms=s.get("rooms", ["*"]),
            )
            for s in slots
        ]

    # ---- time-based model selection ---------------------------------------

    def get_current_model(self, room_id: str = "*") -> Tuple[str, str]:
        """Return ``(model_name, reason)`` for *right now* in *room_id*."""
        hour = datetime.now(timezone.utc).hour

        candidates: List[ScheduleSlot] = []
        for slot in self.schedule:
            if slot.start_hour <= hour < slot.end_hour:
                if "*" in slot.rooms or room_id in slot.rooms:
                    candidates.append(slot)

        if not candidates:
            return "glm-5-turbo", "fallback"

        # Prefer room-specific over wildcard
        room_specific = [s for s in candidates if room_id in s.rooms]
        best = room_specific[0] if room_specific else candidates[0]
        return best.model, best.reason

    def get_model_for_hour(self, hour: int, room_id: str = "*") -> Tuple[str, str]:
        """Like :meth:`get_current_model` but for any UTC hour."""
        candidates: List[ScheduleSlot] = []
        for slot in self.schedule:
            if slot.start_hour <= hour < slot.end_hour:
                if "*" in slot.rooms or room_id in slot.rooms:
                    candidates.append(slot)
        if not candidates:
            return "glm-5-turbo", "fallback"
        room_specific = [s for s in candidates if room_id in s.rooms]
        if room_specific:
            return room_specific[0].model, room_specific[0].reason
        return candidates[0].model, candidates[0].reason

    # ---- task submission --------------------------------------------------

    def submit_task(
        self,
        task_id: str,
        room_id: str,
        description: str,
        required_tier: ModelTier,
        est_tokens: int,
        priority: int = 0,
        deadline: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> QueuedTask:
        """Submit a task to the priority queue and return it."""
        qt = self.task_queue.enqueue(
            task_id=task_id,
            room_id=room_id,
            description=description,
            required_tier=required_tier,
            estimated_tokens=est_tokens,
            priority=priority,
            deadline=deadline,
            agent_id=agent_id,
        )
        return qt

    # ---- scheduling -------------------------------------------------------

    def schedule_pending(self) -> List[ScheduledTask]:
        """Assign models to pending tasks respecting schedule, budget, and load."""
        scheduled: List[ScheduledTask] = []
        remaining_budget = self.daily_budget - self.spent

        pending = self.task_queue.peek_all()

        for qt in pending:
            task = qt.to_scheduled_task()

            # Compute estimated cost
            model_name, reason = self.get_current_model(task.room_id)
            model = FLEET_MODELS.get(model_name)
            if model is None:
                continue

            # Dynamic tier adjustment: if load is high, step down a tier
            effective_tier = self._adjust_tier(task.required_tier)
            tier_order = list(ModelTier)

            if tier_order.index(model.tier) < tier_order.index(effective_tier):
                continue  # model not powerful enough yet

            est_cost = (task.estimated_tokens / 1000) * model.cost_per_1k_tokens
            if est_cost > remaining_budget:
                continue  # can't afford it

            # Fair-share check: cap per-agent usage to 60 % of budget
            if task.agent_id:
                agent_used = self._agent_share.get(task.agent_id, 0.0)
                agent_limit = self.daily_budget * 0.60
                if agent_used + est_cost > agent_limit:
                    continue

            # Commit
            task.assigned_model = model_name
            task.status = "scheduled"
            remaining_budget -= est_cost
            self._tier_usage[model.tier] += 1
            self.task_queue.mark_scheduled(qt.task_id, model_name)
            scheduled.append(task)

        return scheduled

    def _adjust_tier(self, required_tier: ModelTier) -> ModelTier:
        """Dynamically lower the tier when the system is under heavy load.

        If more than 5 tasks are pending, step down one tier to save budget.
        """
        load = self.task_queue.size()
        tier_order = list(ModelTier)
        idx = tier_order.index(required_tier)
        if load > 10 and idx > 0:
            return tier_order[idx - 1]
        if load > 20 and idx > 1:
            return tier_order[idx - 2]
        return required_tier

    # ---- completion -------------------------------------------------------

    def complete_task(self, task_id: str, actual_tokens: int) -> Optional[float]:
        """Mark a task done and return the actual cost."""
        qt = self.task_queue.remove(task_id)
        if qt is None:
            return None

        task = qt.to_scheduled_task()
        model = FLEET_MODELS.get(task.assigned_model or "")
        cost = 0.0
        if model:
            cost = (actual_tokens / 1000) * model.cost_per_1k_tokens
            self.spent += cost
        task.status = "done"

        if task.agent_id:
            self._agent_share[task.agent_id] = (
                self._agent_share.get(task.agent_id, 0.0) + cost
            )

        self.completed.append(task)
        return cost

    def fail_task(self, task_id: str, reason: str = "") -> None:
        """Mark a task as failed."""
        qt = self.task_queue.remove(task_id)
        if qt is None:
            return
        task = qt.to_scheduled_task()
        task.status = "failed"
        self.completed.append(task)

    # ---- cost optimization ------------------------------------------------

    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Return a list of actionable cost-optimization recommendations."""
        suggestions: List[Dict[str, Any]] = []

        # 1. Budget pacing
        now = datetime.now(timezone.utc)
        day_fraction = (now.hour * 60 + now.minute) / 1440.0
        expected_spend = self.spent / day_fraction if day_fraction > 0.01 else 0.0
        if expected_spend > self.daily_budget:
            suggestions.append({
                "type": "over_budget",
                "severity": "high",
                "message": (
                    f"Projected daily spend ${expected_spend:.4f} exceeds "
                    f"budget ${self.daily_budget:.4f}. Consider downgrading tasks."
                ),
            })

        # 2. Tier downgrade candidates
        for tier, count in self._tier_usage.items():
            if tier >= ModelTier.EXPERT and count > 5:
                suggestions.append({
                    "type": "tier_downgrade",
                    "severity": "medium",
                    "message": (
                        f"{tier.name} used {count}× today. Consider moving "
                        f"some tasks to {ModelTier(tier - 1).name} to save costs."
                    ),
                })

        # 3. Night-shift opportunity
        hour = now.hour
        if 6 < hour < 22 and self.task_queue.size() > 3:
            suggestions.append({
                "type": "defer_to_night",
                "severity": "low",
                "message": (
                    f"{self.task_queue.size()} low-priority tasks could be "
                    f"deferred to the night shift (00:00–06:00 UTC) for "
                    f"cheaper execution."
                ),
            })

        # 4. Agent imbalance
        if self._agent_share:
            max_agent = max(self._agent_share, key=self._agent_share.get)
            max_share = self._agent_share[max_agent]
            if max_share > self.daily_budget * 0.5:
                suggestions.append({
                    "type": "agent_imbalance",
                    "severity": "medium",
                    "message": (
                        f"Agent '{max_agent}' consumes {max_share:.4f} "
                        f"({max_share / self.daily_budget * 100:.0f}% of budget). "
                        f"Consider redistributing tasks."
                    ),
                })

        if not suggestions:
            suggestions.append({
                "type": "all_clear",
                "severity": "info",
                "message": "No cost optimizations needed — spending is on track.",
            })

        return suggestions

    # ---- status & reporting -----------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a dict summarising the scheduler state."""
        now_hour = datetime.now(timezone.utc).hour
        current_model, reason = self.get_current_model()

        return {
            "current_time_utc": f"{now_hour:02d}:00",
            "current_model": current_model,
            "schedule_reason": reason,
            "pending_tasks": self.task_queue.size(),
            "completed_today": len(self.completed),
            "budget_used": round(self.spent, 6),
            "budget_remaining": round(self.daily_budget - self.spent, 6),
            "budget_limit": self.daily_budget,
            "agent_shares": {
                k: round(v, 6) for k, v in self._agent_share.items()
            },
            "tier_usage": {t.name: c for t, c in self._tier_usage.items()},
        }

    def list_model_tiers(self) -> List[Dict[str, Any]]:
        """Return every fleet model as a serialisable dict."""
        rows: List[Dict[str, Any]] = []
        for name, prof in sorted(FLEET_MODELS.items(), key=lambda kv: kv[1].tier):
            rows.append({
                "name": name,
                "tier": prof.tier.name,
                "cost_per_1k": prof.cost_per_1k_tokens,
                "speed_tps": prof.speed_tokens_per_sec,
                "quality": prof.quality_score,
                "best_for": prof.best_for,
            })
        return rows

    def show_schedule(self) -> List[Dict[str, Any]]:
        """Return the time-of-day schedule as a list of dicts."""
        return [
            {
                "start": f"{s.start_hour:02d}:00",
                "end": f"{s.end_hour:02d}:00",
                "model": s.model,
                "reason": s.reason,
                "rooms": s.rooms,
            }
            for s in sorted(self.schedule, key=lambda s: s.start_hour)
        ]

    # ---- serialization ----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.status(), indent=2)
