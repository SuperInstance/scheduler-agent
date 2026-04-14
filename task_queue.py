#!/usr/bin/env python3
"""
Priority Task Queue — deadline-aware, deduplicating, backpressure-ready.

Designed to sit between task submission and the FleetScheduler.
"""

from __future__ import annotations

import time
import hashlib
from datetime import datetime, timezone
from enum import IntEnum
from typing import Dict, List, Optional

from dataclasses import dataclass, field

from models import ModelTier, ScheduledTask


# ---------------------------------------------------------------------------
# Priority Levels
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Higher value → more important."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# ---------------------------------------------------------------------------
# Queue Entry
# ---------------------------------------------------------------------------

@dataclass
class QueuedTask:
    """A task sitting in the priority queue."""
    task_id: str
    room_id: str
    description: str
    required_tier: ModelTier
    estimated_tokens: int
    priority: int = Priority.NORMAL
    deadline: Optional[float] = None       # unix timestamp
    created_at: float = field(default_factory=time.time)
    agent_id: Optional[str] = None         # owning agent
    dedup_key: Optional[str] = None        # fingerprint for dedup
    assigned_model: Optional[str] = None   # set by scheduler
    status: str = "pending"                # pending | scheduled

    # -- helpers ------------------------------------------------------------

    def to_scheduled_task(self) -> ScheduledTask:
        """Convert into a :class:`ScheduledTask` for the scheduler."""
        return ScheduledTask(
            task_id=self.task_id,
            room_id=self.room_id,
            description=self.description,
            required_tier=self.required_tier,
            estimated_tokens=self.estimated_tokens,
            deadline=self.deadline,
            priority=self.priority,
            created_at=self.created_at,
            agent_id=self.agent_id,
            assigned_model=self.assigned_model,
        )

    def deadline_urgency(self) -> float:
        """Seconds until deadline (negative means overdue)."""
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.time()

    def compute_dedup_key(self) -> str:
        """Fingerprint based on description + room + tier."""
        raw = f"{self.description}|{self.room_id}|{self.required_tier.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------

class TaskQueue:
    """Priority-aware task queue with deduplication and backpressure.

    Parameters
    ----------
    max_size
        Maximum number of pending tasks. ``None`` means unlimited.
    dedup_enabled
        Whether to reject duplicate tasks (based on dedup_key).
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        dedup_enabled: bool = True,
    ) -> None:
        self._queue: List[QueuedTask] = []
        self._seen_keys: Dict[str, str] = {}     # dedup_key → task_id
        self._agent_affinity: Dict[str, List[str]] = {}  # agent → [task_ids]
        self.max_size: Optional[int] = max_size
        self.dedup_enabled: bool = dedup_enabled

    # ---- enqueue ----------------------------------------------------------

    def enqueue(
        self,
        task_id: str,
        room_id: str,
        description: str,
        required_tier: ModelTier,
        estimated_tokens: int,
        priority: int = Priority.NORMAL,
        deadline: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> QueuedTask:
        """Add a task to the queue. Returns the created :class:`QueuedTask`.

        Raises :class:`TaskDuplicateError` if dedup is enabled and the task
        already exists, or :class:`TaskQueueFullError` if at capacity.
        """
        qt = QueuedTask(
            task_id=task_id,
            room_id=room_id,
            description=description,
            required_tier=required_tier,
            estimated_tokens=estimated_tokens,
            priority=priority,
            deadline=deadline,
            agent_id=agent_id,
        )

        # Dedup
        if self.dedup_enabled:
            qt.dedup_key = qt.compute_dedup_key()
            if qt.dedup_key in self._seen_keys:
                existing = self._seen_keys[qt.dedup_key]
                raise TaskDuplicateError(
                    f"Duplicate of task {existing} (key={qt.dedup_key})"
                )
            self._seen_keys[qt.dedup_key] = task_id

        # Backpressure
        if self.max_size is not None and len(self._queue) >= self.max_size:
            raise TaskQueueFullError(
                f"Queue at capacity ({self.max_size}). Rejecting task {task_id}."
            )

        # Agent affinity bookkeeping
        if agent_id:
            self._agent_affinity.setdefault(agent_id, []).append(task_id)

        self._queue.append(qt)
        return qt

    # ---- dequeue / peek ---------------------------------------------------

    def dequeue(self) -> Optional[QueuedTask]:
        """Pop the highest-priority task from the queue."""
        if not self._queue:
            return None
        self._sort()
        return self._queue.pop(0)

    def peek(self) -> Optional[QueuedTask]:
        """Return the highest-priority task without removing it."""
        if not self._queue:
            return None
        self._sort()
        return self._queue[0]

    def peek_all(self) -> List[QueuedTask]:
        """Return all pending tasks sorted by priority."""
        self._sort()
        return list(self._queue)

    def peek_for_agent(self, agent_id: str, limit: int = 5) -> List[QueuedTask]:
        """Return tasks with affinity for *agent_id* (preferring that agent)."""
        self._sort()
        agent_ids = set(self._agent_affinity.get(agent_id, []))
        agent_tasks = [t for t in self._queue if t.task_id in agent_ids]
        other_tasks = [t for t in self._queue if t.task_id not in agent_ids]
        return (agent_tasks + other_tasks)[:limit]

    # ---- status mutations -------------------------------------------------

    def mark_scheduled(self, task_id: str, model: Optional[str] = None) -> bool:
        """Mark a task as scheduled and optionally store the assigned model. Returns ``True`` if found."""
        for t in self._queue:
            if t.task_id == task_id:
                t.status = "scheduled"
                if model is not None:
                    t.assigned_model = model
                return True
        return False

    def remove(self, task_id: str) -> Optional[QueuedTask]:
        """Remove a task by ID (e.g. on completion / failure)."""
        for i, t in enumerate(self._queue):
            if t.task_id == task_id:
                removed = self._queue.pop(i)
                self._cleanup_dedup(removed)
                self._cleanup_affinity(removed)
                return removed
        return None

    # ---- deadline ---------------------------------------------------------

    def overdue_tasks(self) -> List[QueuedTask]:
        """Return all tasks past their deadline."""
        now = time.time()
        return [
            t for t in self._queue
            if t.deadline is not None and t.deadline < now
        ]

    def deadline_sorted(self) -> List[QueuedTask]:
        """Return tasks sorted by deadline urgency (soonest first)."""
        return sorted(
            self._queue,
            key=lambda t: t.deadline if t.deadline else float("inf"),
        )

    # ---- introspection ----------------------------------------------------

    def size(self) -> int:
        """Number of tasks in the queue."""
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def contains(self, task_id: str) -> bool:
        return any(t.task_id == task_id for t in self._queue)

    def summary(self) -> Dict[str, object]:
        """Quick stats dict."""
        pending = [t for t in self._queue if t.status == "pending"]
        scheduled = [t for t in self._queue if t.status == "scheduled"]
        return {
            "total": len(self._queue),
            "pending": len(pending),
            "scheduled": len(scheduled),
            "overdue": len(self.overdue_tasks()),
            "max_size": self.max_size,
            "utilization": (
                len(self._queue) / self.max_size
                if self.max_size else None
            ),
        }

    # ---- internal ---------------------------------------------------------

    def _sort(self) -> None:
        """Sort in-place: priority desc, then deadline asc, then created asc."""
        self._queue.sort(key=lambda t: (
            -t.priority,
            t.deadline if t.deadline else float("inf"),
            t.created_at,
        ))

    def _cleanup_dedup(self, qt: QueuedTask) -> None:
        if qt.dedup_key and qt.dedup_key in self._seen_keys:
            del self._seen_keys[qt.dedup_key]

    def _cleanup_affinity(self, qt: QueuedTask) -> None:
        if qt.agent_id and qt.agent_id in self._agent_affinity:
            ids = self._agent_affinity[qt.agent_id]
            if qt.task_id in ids:
                ids.remove(qt.task_id)
            if not ids:
                del self._agent_affinity[qt.agent_id]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TaskQueueError(Exception):
    """Base for queue errors."""


class TaskDuplicateError(TaskQueueError):
    """Raised when a duplicate task is submitted."""


class TaskQueueFullError(TaskQueueError):
    """Raised when the queue is at capacity (backpressure)."""
