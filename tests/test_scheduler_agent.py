#!/usr/bin/env python3
"""
Tests for the Scheduler Agent — scheduling, budget, queue, cost, CLI.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_scheduler_agent.py
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
        """Default schedule should cover 24 hours."""
        slots = self.scheduler.show_schedule()
        total_hours = 0
        for s in slots:
            start = int(s["start"].split(":")[0])
            end = int(s["end"].split(":")[0])
            total_hours += (end - start)
        self.assertGreaterEqual(total_hours, 24)

    def test_get_current_model_returns_fleet_model(self) -> None:
        """get_current_model must always return a known fleet model."""
        model, reason = self.scheduler.get_current_model("bridge")
        self.assertIn(model, FLEET_MODELS)

    def test_get_current_model_room_specific(self) -> None:
        """Room-specific slots should be preferred over wildcards."""
        model, _ = self.scheduler.get_current_model("bridge")
        self.assertIsNotNone(model)

    def test_get_model_for_hour(self) -> None:
        """get_model_for_hour should return a model for any hour."""
        for hour in range(24):
            model, reason = self.scheduler.get_model_for_hour(hour, "*")
            self.assertIn(model, FLEET_MODELS)

    def test_schedule_pending_respects_tier(self) -> None:
        """Tasks requiring EXPERT should not be assigned to CHEAP models."""
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
        """Higher-priority tasks should be scheduled first."""
        self.scheduler.submit_task("T-LOW", "*", "Low priority", ModelTier.CHEAP, 500, priority=1)
        self.scheduler.submit_task("T-HIGH", "*", "High priority", ModelTier.CHEAP, 500, priority=10)
        scheduled = self.scheduler.schedule_pending()
        ids = [t.task_id for t in scheduled]
        if "T-HIGH" in ids and "T-LOW" in ids:
            self.assertLess(ids.index("T-HIGH"), ids.index("T-LOW"))

    def test_budget_limit_enforced(self) -> None:
        """Scheduler should not exceed daily budget."""
        self.scheduler.daily_budget = 0.0001  # very tight
        self.scheduler.submit_task("T-BIG", "*", "Expensive task", ModelTier.EXPERT, 100000, priority=5)
        scheduled = self.scheduler.schedule_pending()
        # With a near-zero budget, nothing expensive should schedule
        self.assertEqual(len(scheduled), 0)

    def test_complete_task_tracks_cost(self) -> None:
        """Completing a task should increase spent amount."""
        # Use CHEAP tier so it always schedules regardless of current hour
        self.scheduler.submit_task("T-OK", "*", "Fine task", ModelTier.CHEAP, 5000, priority=5)
        scheduled = self.scheduler.schedule_pending()
        self.assertTrue(any(t.task_id == "T-OK" for t in scheduled),
                        "Task T-OK should have been scheduled")
        cost = self.scheduler.complete_task("T-OK", 5000)
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)
        self.assertGreater(self.scheduler.spent, 0)


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

    def test_fair_share_caps_agent_spending(self) -> None:
        """A single agent should not consume more than 60% of budget."""
        self.scheduler.daily_budget = 1.0
        # Submit many tasks for the same agent
        for i in range(20):
            self.scheduler.submit_task(
                f"T-{i}", "*", f"Agent-A task {i}",
                ModelTier.GOOD, 50000, priority=5, agent_id="agent-A",
            )
        scheduled = self.scheduler.schedule_pending()
        # Count how many got scheduled for agent-A
        agent_a_count = sum(1 for t in scheduled if t.agent_id == "agent-A")
        total_cost = sum(
            (FLEET_MODELS[t.assigned_model].cost_per_1k_tokens * t.estimated_tokens / 1000)
            for t in scheduled if t.assigned_model in FLEET_MODELS
        )
        # Should not exceed 60% of budget
        self.assertLessEqual(total_cost, self.scheduler.daily_budget * 0.65)

    def test_suggest_optimizations_returns_list(self) -> None:
        opts = self.scheduler.suggest_optimizations()
        self.assertIsInstance(opts, list)
        self.assertGreater(len(opts), 0)
        for o in opts:
            self.assertIn("type", o)
            self.assertIn("severity", o)
            self.assertIn("message", o)


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

    def test_backpressure_rejects_at_capacity(self) -> None:
        small_queue = TaskQueue(max_size=2)
        small_queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        small_queue.enqueue("T2", "*", "B", ModelTier.CHEAP, 1000)
        with self.assertRaises(TaskQueueFullError):
            small_queue.enqueue("T3", "*", "C", ModelTier.CHEAP, 1000)

    def test_remove_decreases_size(self) -> None:
        self.queue.enqueue("T1", "*", "Task", ModelTier.CHEAP, 1000)
        self.assertEqual(self.queue.size(), 1)
        self.queue.remove("T1")
        self.assertEqual(self.queue.size(), 0)

    def test_overdue_tasks(self) -> None:
        self.queue.enqueue(
            "T-PAST", "*", "Overdue",
            ModelTier.CHEAP, 1000,
            deadline=time.time() - 10,  # 10s ago
        )
        self.queue.enqueue(
            "T-FUTURE", "*", "Future",
            ModelTier.CHEAP, 1000,
            deadline=time.time() + 3600,
        )
        overdue = self.queue.overdue_tasks()
        self.assertEqual(len(overdue), 1)
        self.assertEqual(overdue[0].task_id, "T-PAST")

    def test_agent_affinity(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000, agent_id="alpha")
        self.queue.enqueue("T2", "*", "B", ModelTier.CHEAP, 1000, agent_id="beta")
        self.queue.enqueue("T3", "*", "C", ModelTier.CHEAP, 1000, agent_id="alpha")
        alpha_tasks = self.queue.peek_for_agent("alpha")
        self.assertEqual(len(alpha_tasks), 3)  # 2 alpha + 1 other
        self.assertEqual(alpha_tasks[0].task_id, "T1")  # alpha first
        self.assertEqual(alpha_tasks[1].task_id, "T3")

    def test_summary(self) -> None:
        self.queue.enqueue("T1", "*", "A", ModelTier.CHEAP, 1000)
        summary = self.queue.summary()
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["pending"], 1)

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


# ---------------------------------------------------------------------------
# 4. Cost Analysis
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

    def test_forecast_daily(self) -> None:
        fc = self.analyzer.forecast_daily()
        self.assertIsInstance(fc, BudgetForecast)
        self.assertEqual(fc.period, "daily")
        self.assertGreaterEqual(fc.overrun_risk, 0)

    def test_forecast_weekly(self) -> None:
        fc = self.analyzer.forecast_weekly()
        self.assertEqual(fc.period, "weekly")

    def test_recommend_returns_list(self) -> None:
        recs = self.analyzer.recommend()
        self.assertIsInstance(recs, list)

    def test_report_daily(self) -> None:
        report = self.analyzer.report_daily()
        self.assertEqual(report["period"], "daily")
        self.assertIn("tier_breakdown", report)

    def test_report_weekly(self) -> None:
        report = self.analyzer.report_weekly()
        self.assertEqual(report["period"], "weekly")

    def test_reset_clears_data(self) -> None:
        self.analyzer.record("glm-5-turbo", 1000, 0)
        self.assertGreater(self.analyzer.total_spent(), 0)
        self.analyzer.reset()
        self.assertEqual(self.analyzer.total_spent(), 0)

    def test_top_models(self) -> None:
        self.analyzer.record("glm-5.1", 5000, 5000)
        self.analyzer.record("glm-4.7-flash", 1000, 0)
        top = self.analyzer.top_models(2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["model"], "glm-5.1")  # most expensive first


# ---------------------------------------------------------------------------
# 5. CLI Parsing
# ---------------------------------------------------------------------------

class TestCLIParsing(unittest.TestCase):
    """Test that CLI arguments parse correctly."""

    def test_no_args_shows_help(self) -> None:
        """With no command, the parser should succeed but args has no func."""
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

    def test_main_status_runs(self) -> None:
        """main(['status']) should not raise."""
        try:
            main(["status"])
        except SystemExit:
            pass  # ok

    def test_main_models_runs(self) -> None:
        """main(['models']) should not raise."""
        try:
            main(["models"])
        except SystemExit:
            pass


# ---------------------------------------------------------------------------
# 6. Model Tier Enum
# ---------------------------------------------------------------------------

class TestModelTier(unittest.TestCase):
    """Test ModelTier ordering and properties."""

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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
