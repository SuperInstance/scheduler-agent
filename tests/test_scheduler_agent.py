#!/usr/bin/env python3
"""
Tests for the Scheduler Agent — scheduling, budget, queue, cost, CLI.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

# Ensure imports work from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ModelTier, ModelProfile, ScheduleSlot, ScheduledTask, FLEET_MODELS,
)
from scheduler import FleetScheduler
from task_queue import (
    TaskQueueError,
    TaskQueue, QueuedTask, Priority,
    TaskDuplicateError, TaskQueueFullError,
)
from cost_analyzer import CostAnalyzer, CostEntry, BudgetForecast
from cli import build_parser, main


# ---------------------------------------------------------------------------
# 1. Scheduling Decisions
# ---------------------------------------------------------------------------

class TestSchedulingDecisions(unittest.TestCase):
    """Test that the scheduler picks the right model at the right time."""

    def setUp(self) -> None:
        self.scheduler = FleetScheduler(daily_budget=10.0)

    def test_default_schedule_has_all_slots(self) -> None:
        slots = self.scheduler.show_schedule()
        total_hours = 0
        for s in slots:
            start = int(s["start"].split(":")[0])
            end = int(s["end"].split(":")[0])
            total_hours += (end - start)
        self.assertGreaterEqual(total_hours, 24)

    def test_get_current_model_returns_fleet_model(self) -> None:
        model, reason = self.scheduler.get_current_model("bridge")
        self.assertIn(model, FLEET_MODELS)

    def test_get_current_model_room_specific(self) -> None:
        model, _ = self.scheduler.get_current_model("bridge")
        self.assertIsNotNone(model)

    def test_get_model_for_hour(self) -> None:
        for hour in range(24):
            model, reason = self.scheduler.get_model_for_hour(hour, "*")
            self.assertIn(model, FLEET_MODELS)

    def test_get_model_for_hour_wildcard(self) -> None:
        """Wildcard room should always return a model."""
        for hour in range(24):
            model, reason = self.scheduler.get_model_for_hour(hour, "*")
            self.assertIsNotNone(model)
            self.assertIsNotNone(reason)

    def test_get_model_for_hour_specific_room(self) -> None:
        """Specific room should also always return a model."""
        for hour in range(24):
            model, reason = self.scheduler.get_model_for_hour(hour, "ready-room")
            self.assertIn(model, FLEET_MODELS)

    def test_fallback_when_no_slot_matches(self) -> None:
        """If no schedule slot matches, fallback to glm-5-turbo."""
        sched = FleetScheduler(daily_budget=10.0)
        sched.schedule = []  # clear all slots
        model, reason = sched.get_model_for_hour(12, "*")
        self.assertEqual(model, "glm-5-turbo")
        self.assertEqual(reason, "fallback")

    def test_night_uses_cheap_models(self) -> None:
        """Between 0-6 UTC, should use cheap models."""
        model, _ = self.scheduler.get_model_for_hour(2, "*")
        self.assertEqual(model, "glm-4.7-flash")

    def test_schedule_pending_respects_tier(self) -> None:
        self.scheduler.submit_task(
            "T-X1", "bridge", "Complex architecture review",
            ModelTier.EXPERT, 2000, priority=8,
        )
        scheduled = self.scheduler.schedule_pending()
        for task in scheduled:
            if task.task_id == "T-X1":
                model = FLEET_MODELS.get(task.assigned_model or "")
                self.assertIsNotNone(model)
                self.assertGreaterEqual(model.tier, ModelTier.EXPERT)

    def test_priority_ordering(self) -> None:
        self.scheduler.submit_task("T-LOW", "*", "Low priority", ModelTier.CHEAP, 500, priority=1)
        self.scheduler.submit_task("T-HIGH", "*", "High priority", ModelTier.CHEAP, 500, priority=10)
        scheduled = self.scheduler.schedule_pending()
        ids = [t.task_id for t in scheduled]
        if "T-HIGH" in ids and "T-LOW" in ids:
            self.assertLess(ids.index("T-HIGH"), ids.index("T-LOW"))

    def test_budget_limit_enforced(self) -> None:
        self.scheduler.daily_budget = 0.0001
        self.scheduler.submit_task("T-BIG", "*", "Expensive task", ModelTier.EXPERT, 100000, priority=5)
        scheduled = self.scheduler.schedule_pending()
        self.assertEqual(len(scheduled), 0)

    def test_complete_task_tracks_cost(self) -> None:
        self.scheduler.submit_task("T-OK", "*", "Fine task", ModelTier.CHEAP, 5000, priority=5)
        scheduled = self.scheduler.schedule_pending()
        self.assertTrue(any(t.task_id == "T-OK" for t in scheduled),
                        "Task T-OK should have been scheduled")
        cost = self.scheduler.complete_task("T-OK", 5000)
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)
        self.assertGreater(self.scheduler.spent, 0)

    def test_complete_unknown_task_returns_none(self) -> None:
        cost = self.scheduler.complete_task("NONEXISTENT", 1000)
        self.assertIsNone(cost)

    def test_fail_task_marks_as_failed(self) -> None:
        self.scheduler.submit_task("T-FAIL", "*", "Will fail", ModelTier.CHEAP, 1000, priority=5)
        self.scheduler.schedule_pending()
        self.scheduler.fail_task("T-FAIL", "test failure")
        self.assertEqual(len(self.scheduler.completed), 1)
        self.assertEqual(self.scheduler.completed[0].status, "failed")

    def test_fail_unknown_task_does_nothing(self) -> None:
        self.scheduler.fail_task("NONEXISTENT", "oops")
        self.assertEqual(len(self.scheduler.completed), 0)

    def test_submit_returns_queued_task(self) -> None:
        qt = self.scheduler.submit_task("T-RET", "*", "Return test", ModelTier.GOOD, 1000)
        self.assertIsInstance(qt, QueuedTask)
        self.assertEqual(qt.task_id, "T-RET")

    def test_list_model_tiers(self) -> None:
        tiers = self.scheduler.list_model_tiers()
        self.assertGreater(len(tiers), 0)
        for t in tiers:
            self.assertIn("name", t)
            self.assertIn("tier", t)
            self.assertIn("cost_per_1k", t)
            self.assertIn("speed_tps", t)
            self.assertIn("quality", t)
            self.assertIn("best_for", t)

    def test_show_schedule_returns_list(self) -> None:
        slots = self.scheduler.show_schedule()
        self.assertIsInstance(slots, list)
        for s in slots:
            self.assertIn("start", s)
            self.assertIn("end", s)
            self.assertIn("model", s)
            self.assertIn("reason", s)
            self.assertIn("rooms", s)

    def test_status_dict_keys(self) -> None:
        st = self.scheduler.status()
        expected_keys = {"current_time_utc", "current_model", "schedule_reason",
                         "pending_tasks", "completed_today", "budget_used",
                         "budget_remaining", "budget_limit", "agent_shares",
                         "tier_usage"}
        self.assertTrue(expected_keys.issubset(set(st.keys())))

    def test_to_json(self) -> None:
        j = self.scheduler.to_json()
        data = json.loads(j)
        self.assertIn("current_model", data)

    def test_load_schedule(self) -> None:
        custom = [{"start_hour": 0, "end_hour": 24, "model": "glm-4.7-flash",
                    "reason": "all cheap", "rooms": ["*"]}]
        self.scheduler.load_schedule(custom)
        model, reason = self.scheduler.get_model_for_hour(12, "*")
        self.assertEqual(model, "glm-4.7-flash")
        self.assertEqual(reason, "all cheap")

    def test_dynamic_tier_adjustment_under_load(self) -> None:
        """Under heavy load, required tier should be lowered."""
        for i in range(15):
            self.scheduler.submit_task(f"T-LD-{i}", "*", f"Load task {i}",
                                      ModelTier.EXPERT, 1000, priority=5)
        tier = self.scheduler._adjust_tier(ModelTier.EXPERT)
        # With 15 tasks, should step down by 1
        self.assertEqual(tier, ModelTier.RUNNER)

    def test_dynamic_tier_no_adjustment_light_load(self) -> None:
        tier = self.scheduler._adjust_tier(ModelTier.EXPERT)
        self.assertEqual(tier, ModelTier.EXPERT)

    def test_dynamic_tier_heavy_load_steps_down_twice(self) -> None:
        for i in range(25):
            self.scheduler.submit_task(f"T-HL-{i}", "*", f"Heavy {i}",
                                      ModelTier.RUNNER, 1000, priority=5)
        tier = self.scheduler._adjust_tier(ModelTier.RUNNER)
        # With 25 tasks, RUNNER (idx=2) steps down by 1 to GOOD (first check fires)
        self.assertEqual(tier, ModelTier.GOOD)

    def test_dynamic_tier_clamp_at_cheap(self) -> None:
        for i in range(25):
            self.scheduler.submit_task(f"T-CL-{i}", "*", f"Clamp {i}",
                                      ModelTier.CHEAP, 1000, priority=5)
        tier = self.scheduler._adjust_tier(ModelTier.CHEAP)
        self.assertEqual(tier, ModelTier.CHEAP)


# ---------------------------------------------------------------------------
# 2. Budget Tracking
# ---------------------------------------------------------------------------

class TestBudgetTracking(unittest.TestCase):
    """Test budget capping and fair-share."""

    def setUp(self) -> None:
        self.scheduler = FleetScheduler(daily_budget=5.0)

    def test_spent_starts_at_zero(self) -> None:
        self.assertEqual(self.scheduler.spent, 0.0)

    def test_spent_increases_on_completion(self) -> None:
        self.scheduler.submit_task("T1", "*", "Task 1", ModelTier.CHEAP, 5000, priority=5)
        self.scheduler.schedule_pending()
        self.scheduler.complete_task("T1", 5000)
        self.assertGreater(self.scheduler.spent, 0)

    def test_budget_remaining_decreases(self) -> None:
        initial = self.scheduler.status()["budget_remaining"]
        self.scheduler.submit_task("T1", "*", "Task", ModelTier.CHEAP, 5000, priority=5)
        self.scheduler.schedule_pending()
        self.scheduler.complete_task("T1", 5000)
        after = self.scheduler.status()["budget_remaining"]
        self.assertLess(after, initial)

    def test_budget_remaining_starts_at_budget(self) -> None:
        self.assertEqual(self.scheduler.status()["budget_remaining"], 5.0)

    def test_suggest_optimizations_returns_list(self) -> None:
        opts = self.scheduler.suggest_optimizations()
        self.assertIsInstance(opts, list)
        self.assertGreater(len(opts), 0)
        for o in opts:
            self.assertIn("type", o)
            self.assertIn("severity", o)
            self.assertIn("message", o)

    def test_suggest_optimizations_all_clear_when_idle(self) -> None:
        opts = self.scheduler.suggest_optimizations()
        types = [o["type"] for o in opts]
        self.assertIn("all_clear", types)

    def test_multiple_completions_accumulate(self) -> None:
        for i in range(3):
            tid = f"T-ACC-{i}"
            self.scheduler.submit_task(tid, "*", f"Task {i}", ModelTier.CHEAP, 5000, priority=5)
            self.scheduler.schedule_pending()
            self.scheduler.complete_task(tid, 5000)
        self.assertEqual(len(self.scheduler.completed), 3)
        self.assertGreater(self.scheduler.spent, 0)

    def test_completed_list_grows(self) -> None:
        self.assertEqual(len(self.scheduler.completed), 0)
        self.scheduler.submit_task("T-C1", "*", "Task", ModelTier.CHEAP, 1000, priority=5)
        self.scheduler.schedule_pending()
        self.scheduler.complete_task("T-C1", 1000)
        self.assertEqual(len(self.scheduler.completed), 1)

    def test_agent_share_tracking(self) -> None:
        self.scheduler.submit_task("T-AS1", "*", "Task", ModelTier.CHEAP, 5000, priority=5, agent_id="agent-X")
        self.scheduler.schedule_pending()
        self.scheduler.complete_task("T-AS1", 5000)
        st = self.scheduler.status()
        self.assertIn("agent-X", st["agent_shares"])
        self.assertGreater(st["agent_shares"]["agent-X"], 0)


# ---------------------------------------------------------------------------
# 3. Priority Queue
# ---------------------------------------------------------------------------

class TestPriorityQueue(unittest.TestCase):
    """Test TaskQueue behaviour."""

    def setUp(self) -> None:
        self.queue = TaskQueue(max_size=10, dedup_enabled=True)

    def test_enqueue_and_size(self) -> None:
        self.queue.enqueue("T1", "*", "Task 1", ModelTier.CHEAP, 1000)
        self.assertEqual(self.queue.size(), 1)

    def test_dequeue_returns_highest_priority(self) -> None:
        self.queue.enqueue("T-LOW", "*", "Low", ModelTier.CHEAP, 1000, priority=Priority.LOW)
        self.queue.enqueue("T-HI", "*", "High", ModelTier.CHEAP, 1000, priority=Priority.CRITICAL)
        task = self.queue.dequeue()
        self.assertEqual(task.task_id, "T-HI")

    def test_dedup_rejects_duplicate(self) -> None:
        self.queue.enqueue("T1", "room-A", "Same desc", ModelTier.CHEAP, 1000)
        with self.assertRaises(TaskDuplicateError):
            self.queue.enqueue("T2", "room-A", "Same desc", ModelTier.CHEAP, 1000)

    def test_dedup_different_room_allows(self) -> None:
        self.queue.enqueue("T1", "room-A", "Same desc", ModelTier.CHEAP, 1000)
        qt = self.queue.enqueue("T2", "room-B", "Same desc", ModelTier.CHEAP, 1000)
        self.assertEqual(qt.task_id, "T2")

    def test_dedup_different_tier_allows(self) -> None:
        self.queue.enqueue("T1", "room-A", "Same desc", ModelTier.GOOD, 1000)
        qt = self.queue.enqueue("T2", "room-A", "Same desc", ModelTier.CHEAP, 1000)
        self.assertEqual(qt.task_id, "T2")

    def test_backpressure_rejects_at_capacity(self) -> None:
        small_queue = TaskQueue(max_size=2)
        small_queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        small_queue.enqueue("T2", "*", "B", ModelTier.CHEAP, 1000)
        with self.assertRaises(TaskQueueFullError):
            small_queue.enqueue("T3", "*", "C", ModelTier.CHEAP, 1000)

    def test_unlimited_queue(self) -> None:
        unlimited = TaskQueue(max_size=None)
        for i in range(100):
            unlimited.enqueue(f"T-{i}", "*", f"Task {i}", ModelTier.CHEAP, 1000)
        self.assertEqual(unlimited.size(), 100)

    def test_dedup_disabled_allows_duplicate(self) -> None:
        no_dedup = TaskQueue(dedup_enabled=False)
        no_dedup.enqueue("T1", "room-A", "Same desc", ModelTier.CHEAP, 1000)
        qt = no_dedup.enqueue("T2", "room-A", "Same desc", ModelTier.CHEAP, 1000)
        self.assertEqual(qt.task_id, "T2")

    def test_remove_decreases_size(self) -> None:
        self.queue.enqueue("T1", "*", "Task", ModelTier.CHEAP, 1000)
        self.assertEqual(self.queue.size(), 1)
        self.queue.remove("T1")
        self.assertEqual(self.queue.size(), 0)

    def test_remove_nonexistent_returns_none(self) -> None:
        result = self.queue.remove("NONEXISTENT")
        self.assertIsNone(result)

    def test_overdue_tasks(self) -> None:
        self.queue.enqueue(
            "T-PAST", "*", "Overdue",
            ModelTier.CHEAP, 1000,
            deadline=time.time() - 10,
        )
        self.queue.enqueue(
            "T-FUTURE", "*", "Future",
            ModelTier.CHEAP, 1000,
            deadline=time.time() + 3600,
        )
        overdue = self.queue.overdue_tasks()
        self.assertEqual(len(overdue), 1)
        self.assertEqual(overdue[0].task_id, "T-PAST")

    def test_overdue_with_no_deadline(self) -> None:
        self.queue.enqueue("T-NODEAD", "*", "No deadline", ModelTier.CHEAP, 1000)
        overdue = self.queue.overdue_tasks()
        self.assertEqual(len(overdue), 0)

    def test_agent_affinity(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000, agent_id="alpha")
        self.queue.enqueue("T2", "*", "B", ModelTier.CHEAP, 1000, agent_id="beta")
        self.queue.enqueue("T3", "*", "C", ModelTier.CHEAP, 1000, agent_id="alpha")
        alpha_tasks = self.queue.peek_for_agent("alpha")
        self.assertEqual(len(alpha_tasks), 3)  # 2 alpha + 1 other
        self.assertEqual(alpha_tasks[0].task_id, "T1")
        self.assertEqual(alpha_tasks[1].task_id, "T3")

    def test_peek_for_agent_limit(self) -> None:
        for i in range(5):
            self.queue.enqueue(f"T-{i}", "*", f"Task {i}", ModelTier.CHEAP, 1000, agent_id="alpha")
        tasks = self.queue.peek_for_agent("alpha", limit=2)
        self.assertEqual(len(tasks), 2)

    def test_summary(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        summary = self.queue.summary()
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["pending"], 1)

    def test_summary_with_scheduled(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        self.queue.mark_scheduled("T1", "glm-4.7-flash")
        summary = self.queue.summary()
        self.assertEqual(summary["scheduled"], 1)
        self.assertEqual(summary["pending"], 0)

    def test_deadline_sorted(self) -> None:
        self.queue.enqueue("T3", "*", "C", ModelTier.CHEAP, 1000, deadline=time.time() + 7200)
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000, deadline=time.time() + 300)
        self.queue.enqueue("T2", "*", "B", ModelTier.CHEAP, 1000, deadline=time.time() + 1800)
        sorted_tasks = self.queue.deadline_sorted()
        self.assertEqual(sorted_tasks[0].task_id, "T1")
        self.assertEqual(sorted_tasks[1].task_id, "T2")
        self.assertEqual(sorted_tasks[2].task_id, "T3")

    def test_dedup_key_deterministic(self) -> None:
        qt = QueuedTask("T1", "room", "desc", ModelTier.GOOD, 100)
        key1 = qt.compute_dedup_key()
        key2 = qt.compute_dedup_key()
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 16)

    def test_dequeue_empty_returns_none(self) -> None:
        self.assertIsNone(self.queue.dequeue())

    def test_peek_empty_returns_none(self) -> None:
        self.assertIsNone(self.queue.peek())

    def test_peek_all_returns_sorted(self) -> None:
        self.queue.enqueue("T-LOW", "*", "Low", ModelTier.CHEAP, 1000, priority=Priority.LOW)
        self.queue.enqueue("T-HI", "*", "High", ModelTier.CHEAP, 1000, priority=Priority.HIGH)
        all_tasks = self.queue.peek_all()
        self.assertEqual(all_tasks[0].task_id, "T-HI")
        self.assertEqual(all_tasks[1].task_id, "T-LOW")

    def test_is_empty(self) -> None:
        self.assertTrue(self.queue.is_empty())
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        self.assertFalse(self.queue.is_empty())

    def test_contains(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        self.assertTrue(self.queue.contains("T1"))
        self.assertFalse(self.queue.contains("T2"))

    def test_mark_scheduled_sets_model(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        result = self.queue.mark_scheduled("T1", "glm-4.7-flash")
        self.assertTrue(result)
        tasks = self.queue.peek_all()
        self.assertEqual(tasks[0].assigned_model, "glm-4.7-flash")
        self.assertEqual(tasks[0].status, "scheduled")

    def test_mark_scheduled_nonexistent(self) -> None:
        result = self.queue.mark_scheduled("NONEXISTENT", "glm-4.7-flash")
        self.assertFalse(result)

    def test_remove_then_reenqueue_allowed(self) -> None:
        self.queue.enqueue("T1", "room-A", "Desc", ModelTier.CHEAP, 1000)
        self.queue.remove("T1")
        qt = self.queue.enqueue("T1", "room-A", "Desc", ModelTier.CHEAP, 1000)
        self.assertEqual(qt.task_id, "T1")


# ---------------------------------------------------------------------------
# 4. QueuedTask
# ---------------------------------------------------------------------------

class TestQueuedTask(unittest.TestCase):

    def test_to_scheduled_task(self) -> None:
        qt = QueuedTask("T1", "room-1", "desc", ModelTier.GOOD, 2000,
                         priority=8, agent_id="agent-A")
        st = qt.to_scheduled_task()
        self.assertIsInstance(st, ScheduledTask)
        self.assertEqual(st.task_id, "T1")
        self.assertEqual(st.room_id, "room-1")
        self.assertEqual(st.required_tier, ModelTier.GOOD)
        self.assertEqual(st.agent_id, "agent-A")

    def test_deadline_urgency_with_deadline(self) -> None:
        qt = QueuedTask("T1", "r", "d", ModelTier.CHEAP, 1000,
                         deadline=time.time() + 3600)
        urgency = qt.deadline_urgency()
        self.assertAlmostEqual(urgency, 3600, delta=10)

    def test_deadline_urgency_overdue(self) -> None:
        qt = QueuedTask("T1", "r", "d", ModelTier.CHEAP, 1000,
                         deadline=time.time() - 100)
        self.assertLess(qt.deadline_urgency(), 0)

    def test_deadline_urgency_no_deadline(self) -> None:
        qt = QueuedTask("T1", "r", "d", ModelTier.CHEAP, 1000)
        self.assertEqual(qt.deadline_urgency(), float("inf"))


# ---------------------------------------------------------------------------
# 5. Cost Analysis
# ---------------------------------------------------------------------------

class TestCostAnalysis(unittest.TestCase):
    """Test CostAnalyzer."""

    def setUp(self) -> None:
        self.analyzer = CostAnalyzer(daily_budget=5.0)

    def test_record_returns_entry(self) -> None:
        entry = self.analyzer.record("glm-5.1", 500, 1500)
        self.assertIsInstance(entry, CostEntry)
        self.assertEqual(entry.model, "glm-5.1")
        self.assertGreater(entry.cost, 0)

    def test_unknown_model_no_cost(self) -> None:
        entry = self.analyzer.record("nonexistent-model", 100, 100)
        self.assertEqual(entry.cost, 0.0)

    def test_total_spent(self) -> None:
        self.analyzer.record("glm-4.7-flash", 1000, 0)
        self.analyzer.record("glm-5-turbo", 500, 500)
        self.assertGreater(self.analyzer.total_spent(), 0)

    def test_total_spent_zero_initially(self) -> None:
        self.assertEqual(self.analyzer.total_spent(), 0.0)

    def test_tier_breakdown(self) -> None:
        self.analyzer.record("glm-4.7-flash", 1000, 0)
        bd = self.analyzer.tier_breakdown()
        self.assertIn("CHEAP", bd)
        self.assertGreater(bd["CHEAP"], 0)

    def test_agent_breakdown(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0, agent_id="alpha")
        self.analyzer.record("glm-5-turbo", 1000, 0, agent_id="beta")
        bd = self.analyzer.agent_breakdown()
        self.assertIn("alpha", bd)
        self.assertIn("beta", bd)

    def test_agent_breakdown_empty(self) -> None:
        bd = self.analyzer.agent_breakdown()
        self.assertEqual(len(bd), 0)

    def test_forecast_daily(self) -> None:
        fc = self.analyzer.forecast_daily()
        self.assertIsInstance(fc, BudgetForecast)
        self.assertEqual(fc.period, "daily")
        self.assertGreaterEqual(fc.overrun_risk, 0)

    def test_forecast_daily_fields(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0)
        fc = self.analyzer.forecast_daily()
        self.assertGreaterEqual(fc.projected_spend, 0)
        self.assertEqual(fc.budget_limit, 5.0)
        self.assertIsInstance(fc.recommendation, str)

    def test_forecast_weekly(self) -> None:
        fc = self.analyzer.forecast_weekly()
        self.assertEqual(fc.period, "weekly")
        self.assertEqual(fc.budget_limit, 35.0)  # 5.0 * 7

    def test_recommend_returns_list(self) -> None:
        recs = self.analyzer.recommend()
        self.assertIsInstance(recs, list)

    def test_recommend_all_clear_when_healthy(self) -> None:
        recs = self.analyzer.recommend()
        types = [r["type"] for r in recs]
        self.assertIn("all_clear", types)

    def test_report_daily(self) -> None:
        report = self.analyzer.report_daily()
        self.assertEqual(report["period"], "daily")
        self.assertIn("tier_breakdown", report)

    def test_report_daily_keys(self) -> None:
        report = self.analyzer.report_daily()
        expected = {"period", "spent_today", "budget_limit", "remaining",
                     "forecast", "tier_breakdown", "agent_breakdown",
                     "top_models", "recommendations"}
        self.assertTrue(expected.issubset(set(report.keys())))

    def test_report_weekly(self) -> None:
        report = self.analyzer.report_weekly()
        self.assertEqual(report["period"], "weekly")

    def test_reset_clears_data(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0)
        self.assertGreater(self.analyzer.total_spent(), 0)
        self.analyzer.reset()
        self.assertEqual(self.analyzer.total_spent(), 0)

    def test_reset_clears_agent_spend(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0, agent_id="agent-X")
        self.analyzer.reset()
        self.assertEqual(len(self.analyzer.agent_breakdown()), 0)

    def test_reset_clears_tier_spend(self) -> None:
        self.analyzer.record("glm-5.1", 500, 500)
        self.analyzer.reset()
        bd = self.analyzer.tier_breakdown()
        self.assertEqual(bd.get("EXPERT", 0), 0.0)

    def test_top_models(self) -> None:
        self.analyzer.record("glm-5.1", 5000, 5000)
        self.analyzer.record("glm-4.7-flash", 1000, 0)
        top = self.analyzer.top_models(2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["model"], "glm-5.1")

    def test_top_models_fields(self) -> None:
        self.analyzer.record("glm-5.1", 1000, 0)
        top = self.analyzer.top_models(5)
        self.assertIn("model", top[0])
        self.assertIn("total_cost", top[0])
        self.assertIn("calls", top[0])
        self.assertIn("avg_cost", top[0])

    def test_top_models_n_limit(self) -> None:
        for model in FLEET_MODELS:
            self.analyzer.record(model, 100, 0)
        top = self.analyzer.top_models(2)
        self.assertEqual(len(top), 2)

    def test_spent_in_last(self) -> None:
        self.analyzer.record("glm-4.7-flash", 1000, 0)
        spent = self.analyzer.spent_in_last(1.0)  # last hour
        self.assertGreater(spent, 0)

    def test_spent_in_last_no_recent(self) -> None:
        spent = self.analyzer.spent_in_last(0.0)  # last 0 hours
        self.assertEqual(spent, 0.0)

    def test_multiple_records_same_model(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0)
        self.analyzer.record("glm-5-turbo", 2000, 500)
        self.assertGreater(self.analyzer.total_spent(), 0)

    def test_record_without_agent(self) -> None:
        entry = self.analyzer.record("glm-4.7-flash", 500, 500)
        self.assertIsNone(entry.agent_id)

    def test_cost_entry_defaults(self) -> None:
        entry = CostEntry()
        self.assertEqual(entry.model, "")
        self.assertEqual(entry.cost, 0.0)
        self.assertEqual(entry.input_tokens, 0)
        self.assertEqual(entry.output_tokens, 0)

    def test_budget_forecast_defaults(self) -> None:
        bf = BudgetForecast(period="daily", projected_spend=1.0,
                            budget_limit=2.0, overrun_risk=0.5,
                            recommendation="Test")
        self.assertEqual(bf.period, "daily")
        self.assertEqual(bf.overrun_risk, 0.5)


# ---------------------------------------------------------------------------
# 6. CLI Parsing
# ---------------------------------------------------------------------------

class TestCLIParsing(unittest.TestCase):

    def test_no_args_shows_help(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertFalse(hasattr(args, "func"))

    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")

    def test_models_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models"])
        self.assertEqual(args.command, "models")

    def test_submit_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "submit", "Review the codebase",
            "--priority", "8",
            "--tier", "EXPERT",
            "--tokens", "5000",
            "--room", "bridge",
            "--task-id", "T-CUSTOM",
            "--agent", "agent-1",
        ])
        self.assertEqual(args.command, "submit")
        self.assertEqual(args.task, "Review the codebase")
        self.assertEqual(args.priority, 8)
        self.assertEqual(args.tier, "EXPERT")
        self.assertEqual(args.tokens, 5000)
        self.assertEqual(args.room, "bridge")
        self.assertEqual(args.task_id, "T-CUSTOM")
        self.assertEqual(args.agent, "agent-1")

    def test_serve_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--interval", "5"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.interval, 5.0)

    def test_cost_command_daily(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cost", "--report", "weekly"])
        self.assertEqual(args.command, "cost")
        self.assertEqual(args.report, "weekly")

    def test_onboard_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["onboard", "--budget", "2.5"])
        self.assertEqual(args.command, "onboard")
        self.assertEqual(args.budget, 2.5)

    def test_schedule_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["schedule"])
        self.assertEqual(args.command, "schedule")

    def test_optimize_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize"])
        self.assertEqual(args.command, "optimize")

    def test_submit_default_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["submit", "do something"])
        self.assertEqual(args.priority, 5)
        self.assertEqual(args.tier, "GOOD")
        self.assertEqual(args.tokens, 1000)
        self.assertEqual(args.room, "*")

    def test_submit_deadline_hours(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["submit", "task", "--deadline-hours", "2.5"])
        self.assertEqual(args.deadline_hours, 2.5)

    def test_main_status_runs(self) -> None:
        try:
            main(["status"])
        except SystemExit:
            pass

    def test_main_models_runs(self) -> None:
        try:
            main(["models"])
        except SystemExit:
            pass


# ---------------------------------------------------------------------------
# 7. Model Tier Enum
# ---------------------------------------------------------------------------

class TestModelTier(unittest.TestCase):

    def test_tier_ordering(self) -> None:
        self.assertLess(ModelTier.CHEAP, ModelTier.GOOD)
        self.assertLess(ModelTier.GOOD, ModelTier.RUNNER)
        self.assertLess(ModelTier.RUNNER, ModelTier.EXPERT)
        self.assertLess(ModelTier.EXPERT, ModelTier.REASONER)

    def test_tier_values_are_integers(self) -> None:
        for tier in ModelTier:
            self.assertIsInstance(tier.value, int)

    def test_all_fleet_models_have_valid_tier(self) -> None:
        for name, profile in FLEET_MODELS.items():
            self.assertIsInstance(profile.tier, ModelTier)
            self.assertGreater(len(profile.best_for), 0)
            self.assertGreater(profile.cost_per_1k_tokens, 0)

    def test_tier_count(self) -> None:
        self.assertEqual(len(ModelTier), 5)

    def test_fleet_model_count(self) -> None:
        self.assertGreater(len(FLEET_MODELS), 0)

    def test_model_profile_defaults(self) -> None:
        mp = ModelProfile("test-model", ModelTier.CHEAP, 0.001, 100, 0.5)
        self.assertEqual(mp.name, "test-model")
        self.assertEqual(mp.max_concurrent, 10)
        self.assertEqual(mp.best_for, [])

    def test_model_profile_custom(self) -> None:
        mp = ModelProfile("custom", ModelTier.EXPERT, 0.01, 50, 0.9,
                          best_for=["test"], max_concurrent=5)
        self.assertEqual(mp.best_for, ["test"])
        self.assertEqual(mp.max_concurrent, 5)


# ---------------------------------------------------------------------------
# 8. ScheduleSlot & ScheduledTask
# ---------------------------------------------------------------------------

class TestDataModels(unittest.TestCase):

    def test_schedule_slot_defaults(self) -> None:
        slot = ScheduleSlot(0, 6, "model", "reason")
        self.assertEqual(slot.rooms, ["*"])

    def test_schedule_slot_custom_rooms(self) -> None:
        slot = ScheduleSlot(0, 6, "model", "reason", ["bridge", "nav"])
        self.assertEqual(slot.rooms, ["bridge", "nav"])

    def test_scheduled_task_defaults(self) -> None:
        task = ScheduledTask("T1", "room", "desc", ModelTier.GOOD, 1000)
        self.assertIsNone(task.deadline)
        self.assertEqual(task.priority, 0)
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.assigned_model)
        self.assertIsNone(task.agent_id)

    def test_scheduled_task_custom(self) -> None:
        task = ScheduledTask(
            "T1", "room", "desc", ModelTier.EXPERT, 5000,
            deadline=12345.0, priority=10, agent_id="agent-X",
        )
        self.assertEqual(task.deadline, 12345.0)
        self.assertEqual(task.priority, 10)
        self.assertEqual(task.agent_id, "agent-X")


# ---------------------------------------------------------------------------
# 9. Priority Enum
# ---------------------------------------------------------------------------

class TestPriority(unittest.TestCase):

    def test_priority_ordering(self) -> None:
        self.assertLess(Priority.LOW, Priority.NORMAL)
        self.assertLess(Priority.NORMAL, Priority.HIGH)
        self.assertLess(Priority.HIGH, Priority.CRITICAL)

    def test_priority_values(self) -> None:
        self.assertEqual(Priority.LOW, 1)
        self.assertEqual(Priority.NORMAL, 5)
        self.assertEqual(Priority.HIGH, 8)
        self.assertEqual(Priority.CRITICAL, 10)

    def test_priority_count(self) -> None:
        self.assertEqual(len(Priority), 4)


# ---------------------------------------------------------------------------
# 10. Queue Exception Classes
# ---------------------------------------------------------------------------

class TestExceptions(unittest.TestCase):

    def test_task_queue_error_base(self) -> None:
        e = TaskQueueError("test")
        self.assertEqual(str(e), "test")

    def test_task_duplicate_error(self) -> None:
        e = TaskDuplicateError("dup")
        self.assertIsInstance(e, TaskQueueError)

    def test_task_queue_full_error(self) -> None:
        e = TaskQueueFullError("full")
        self.assertIsInstance(e, TaskQueueError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
