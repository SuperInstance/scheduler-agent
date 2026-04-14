#!/usr/bin/env python3
"""
Scheduler Agent CLI — command-line interface for the fleet scheduler.

Subcommands
-----------
serve      Start the scheduler service (blocking loop).
submit     Submit a task to the scheduler.
schedule   Display the current time-of-day schedule.
models     List available model tiers and profiles.
cost       Show cost analysis and spending reports.
optimize   Suggest cost optimizations.
onboard    Set up the agent with default config.
status     Show live scheduler status.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import signal
from typing import List, Optional

from models import ModelTier, FLEET_MODELS
from scheduler import FleetScheduler
from task_queue import TaskQueue, Priority
from cost_analyzer import CostAnalyzer


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.expanduser("~/.scheduler-agent")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")


def _ensure_config() -> None:
    """Create default config if it doesn't exist."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        return
    default = {
        "daily_budget": 1.0,
        "max_queue_size": 200,
        "dedup_enabled": True,
        "schedule": [
            {"start_hour": 0, "end_hour": 6, "model": "glm-4.7-flash",
             "reason": "night bulk", "rooms": ["*"]},
            {"start_hour": 6, "end_hour": 10, "model": "glm-5-turbo",
             "reason": "morning driver", "rooms": ["*"]},
            {"start_hour": 10, "end_hour": 14, "model": "glm-5.1",
             "reason": "peak expert", "rooms": ["bridge", "nav"]},
            {"start_hour": 10, "end_hour": 14, "model": "glm-5-turbo",
             "reason": "peak runner", "rooms": ["*"]},
            {"start_hour": 14, "end_hour": 18, "model": "glm-5-turbo",
             "reason": "afternoon", "rooms": ["*"]},
            {"start_hour": 18, "end_hour": 22, "model": "glm-4.7",
             "reason": "evening review", "rooms": ["*"]},
            {"start_hour": 22, "end_hour": 24, "model": "glm-4.7-flash",
             "reason": "late night bulk", "rooms": ["*"]},
        ],
    }
    try:
        import yaml
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(default, f, default_flow_style=False)
    except ImportError:
        with open(CONFIG_FILE, "w") as f:
            json.dump(default, f, indent=2)


def _load_config() -> dict:
    """Load config, returning defaults if unavailable."""
    _ensure_config()
    try:
        import yaml
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        with open(CONFIG_FILE) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Build scheduler from config
# ---------------------------------------------------------------------------

def _build_scheduler(cfg: Optional[dict] = None) -> FleetScheduler:
    """Create a :class:`FleetScheduler` wired with queue and cost analyzer."""
    cfg = cfg or _load_config()
    queue = TaskQueue(
        max_size=cfg.get("max_queue_size", 200),
        dedup_enabled=cfg.get("dedup_enabled", True),
    )
    sched = FleetScheduler(
        daily_budget=cfg.get("daily_budget", 1.0),
        task_queue=queue,
    )
    custom_schedule = cfg.get("schedule")
    if custom_schedule:
        sched.load_schedule(custom_schedule)
    return sched


def _build_analyzer(cfg: Optional[dict] = None) -> CostAnalyzer:
    cfg = cfg or _load_config()
    return CostAnalyzer(daily_budget=cfg.get("daily_budget", 1.0))


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the scheduler service (polling loop)."""
    print("╔══════════════════════════════════════════════════╗")
    print("║  Scheduler Agent — Fleet Task Scheduler          ║")
    print("║  Ctrl-C to stop                                   ║")
    print("╚══════════════════════════════════════════════════╝\n")

    sched = _build_scheduler()
    analyzer = _build_analyzer()
    interval = args.interval

    running = True

    def _shutdown(sig, frame):
        nonlocal running
        print("\n⏹  Shutting down scheduler...")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    tick = 0
    while running:
        scheduled = sched.schedule_pending()
        for task in scheduled:
            print(f"  ▶ {task.task_id}: {task.description[:50]:50s} → "
                  f"{task.assigned_model}")

        # Simulate completions for demo
        if tick % 10 == 0 and scheduled:
            for t in scheduled[:2]:
                cost = sched.complete_task(t.task_id, t.estimated_tokens)
                if cost is not None:
                    analyzer.record(t.assigned_model or "", 0, t.estimated_tokens, t.task_id)
                    print(f"  ✓ {t.task_id} done (${cost:.6f})")

        if tick % 5 == 0:
            st = sched.status()
            print(f"  [{st['pending_tasks']} pending | "
                  f"${st['budget_used']:.4f} spent | "
                  f"${st['budget_remaining']:.4f} remaining]")

        time.sleep(interval)
        tick += 1

    print(f"\nFinal: {json.dumps(sched.status(), indent=2)}")
    sys.exit(0)


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit a single task to the scheduler."""
    sched = _build_scheduler()
    tier_map = {t.name: t for t in ModelTier}

    tier_name = args.tier.upper()
    if tier_name not in tier_map:
        print(f"Error: unknown tier '{args.tier}'. "
              f"Choose from: {', '.join(tier_map)}", file=sys.stderr)
        sys.exit(1)

    deadline = None
    if args.deadline_hours:
        deadline = time.time() + args.deadline_hours * 3600

    qt = sched.submit_task(
        task_id=args.task_id or f"T{int(time.time() * 1000) % 100000}",
        room_id=args.room,
        description=args.task,
        required_tier=tier_map[tier_name],
        est_tokens=args.tokens,
        priority=args.priority,
        deadline=deadline,
        agent_id=args.agent,
    )

    print(f"✓ Task enqueued: {qt.task_id}")
    print(f"  Description : {qt.description}")
    print(f"  Tier        : {qt.required_tier.name}")
    print(f"  Priority    : {qt.priority}")
    print(f"  Est. tokens : {qt.estimated_tokens}")
    if qt.deadline:
        print(f"  Deadline    : {qt.deadline}")
    print(f"  Queue size  : {sched.task_queue.size()}")


def cmd_schedule(args: argparse.Namespace) -> None:
    """Display the current time-of-day schedule."""
    sched = _build_scheduler()
    slots = sched.show_schedule()

    print("\n═══ Daily Schedule (UTC) ═══\n")
    print(f"  {'Window':12s}  {'Model':20s}  {'Reason':22s}  Rooms")
    print(f"  {'─' * 12}  {'─' * 20}  {'─' * 22}  {'─' * 20}")
    for s in slots:
        rooms = ", ".join(s["rooms"])
        print(f"  {s['start']:>5s}-{s['end']:<5s}  "
              f"{s['model']:20s}  {s['reason']:22s}  {rooms}")
    print()


def cmd_models(args: argparse.Namespace) -> None:
    """List available model tiers."""
    sched = _build_scheduler()
    tiers = sched.list_model_tiers()

    print("\n═══ Fleet Model Tiers ═══\n")
    print(f"  {'Model':20s}  {'Tier':10s}  {'$/1k tok':>9s}  "
          f"{'tok/s':>6s}  {'Quality':>7s}  Best For")
    print(f"  {'─' * 20}  {'─' * 10}  {'─' * 9}  "
          f"{'─' * 6}  {'─' * 7}  {'─' * 20}")
    for m in tiers:
        best = ", ".join(m["best_for"])
        print(f"  {m['name']:20s}  {m['tier']:10s}  "
              f"{m['cost_per_1k']:9.4f}  {m['speed_tps']:6.0f}  "
              f"{m['quality']:7.2f}  {best}")
    print()


def cmd_cost(args: argparse.Namespace) -> None:
    """Show cost analysis."""
    analyzer = _build_analyzer()
    report_type = args.report or "daily"

    if report_type == "weekly":
        report = analyzer.report_weekly()
    else:
        report = analyzer.report_daily()

    print(f"\n═══ {report_type.capitalize()} Cost Report ═══\n")
    print(f"  Spent        : ${report.get('spent_today', report.get('spent_7d', 0)):.6f}")
    print(f"  Budget limit : ${report.get('budget_limit', report.get('budget_7d', 0)):.6f}")
    print(f"  Remaining    : ${report.get('remaining', report.get('budget_7d', 0) - report.get('spent_7d', 0)):.6f}")

    fc = report["forecast"]
    print(f"\n  Forecast ({fc.period}):")
    print(f"    Projected : ${fc.projected_spend:.6f}")
    print(f"    Overrun   : {fc.overrun_risk:.0%}")

    if report.get("tier_breakdown"):
        print(f"\n  Tier Breakdown:")
        for tier, cost in report["tier_breakdown"].items():
            bar = "█" * int(cost * 2000)
            print(f"    {tier:10s}  ${cost:.6f}  {bar}")

    if report.get("recommendations"):
        print(f"\n  Recommendations:")
        for r in report["recommendations"]:
            print(f"    [{r['severity'].upper()}] {r['message']}")

    print()


def cmd_optimize(args: argparse.Namespace) -> None:
    """Suggest cost optimizations."""
    sched = _build_scheduler()
    analyzer = _build_analyzer()

    scheduler_opts = sched.suggest_optimizations()
    cost_opts = analyzer.recommend()

    print("\n═══ Cost Optimizations ═══\n")

    print("  Scheduler Optimizations:")
    for o in scheduler_opts:
        print(f"    [{o['severity'].upper()}] {o['message']}")

    print("\n  Cost Optimizations:")
    for o in cost_opts:
        print(f"    [{o['severity'].upper()}] {o['message']}")

    print()


def cmd_onboard(args: argparse.Namespace) -> None:
    """Set up the agent with default configuration."""
    _ensure_config()
    cfg = _load_config()

    print("╔══════════════════════════════════════════════════╗")
    print("║  Scheduler Agent — Onboarding                    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    if args.budget:
        cfg["daily_budget"] = args.budget
        try:
            import yaml
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        except ImportError:
            with open(CONFIG_FILE, "w") as f:
                json.dump(cfg, f, indent=2)
        print(f"  ✓ Daily budget set to ${args.budget}")

    sched = _build_scheduler(cfg)

    print(f"  ✓ Config written to {CONFIG_FILE}")
    print(f"  ✓ {len(sched.schedule)} schedule slots configured")
    print(f"  ✓ {len(FLEET_MODELS)} models available")
    print(f"  ✓ Queue capacity: {sched.task_queue.max_size}")
    print(f"  ✓ Dedup: {'enabled' if sched.task_queue.dedup_enabled else 'disabled'}")
    print()
    print("  Ready! Run 'python cli.py serve' to start.\n")


def cmd_status(args: argparse.Namespace) -> None:
    """Show live scheduler status."""
    sched = _build_scheduler()
    st = sched.status()

    print("\n═══ Scheduler Status ═══\n")
    print(f"  Time (UTC)      : {st['current_time_utc']}")
    print(f"  Active Model    : {st['current_model']}")
    print(f"  Schedule Reason : {st['schedule_reason']}")
    print(f"  Pending Tasks   : {st['pending_tasks']}")
    print(f"  Completed Today : {st['completed_today']}")
    print(f"  Budget Used     : ${st['budget_used']:.6f}")
    print(f"  Budget Remaining: ${st['budget_remaining']:.6f}")
    print(f"  Budget Limit    : ${st['budget_limit']:.6f}")

    if st.get("tier_usage"):
        print(f"\n  Tier Usage:")
        for tier, count in st["tier_usage"].items():
            print(f"    {tier:10s}  {count}×")

    if st.get("agent_shares"):
        print(f"\n  Agent Budget Shares:")
        for agent, share in st["agent_shares"].items():
            print(f"    {agent:20s}  ${share:.6f}")

    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scheduler-agent",
        description="Fleet Task Scheduler — the right model at the right time.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start the scheduler service")
    p_serve.add_argument("--interval", type=float, default=2.0,
                         help="Polling interval in seconds (default: 2)")
    p_serve.set_defaults(func=cmd_serve)

    # submit
    p_sub = sub.add_parser("submit", help="Submit a task")
    p_sub.add_argument("task", help="Task description")
    p_sub.add_argument("--priority", type=int, default=5,
                       help="Priority 1-10 (default: 5)")
    p_sub.add_argument("--tier", default="GOOD",
                       help="Required tier (default: GOOD)")
    p_sub.add_argument("--tokens", type=int, default=1000,
                       help="Estimated tokens (default: 1000)")
    p_sub.add_argument("--budget", type=float, default=None,
                       help="Task budget override")
    p_sub.add_argument("--room", default="*",
                       help="Target room (default: *)")
    p_sub.add_argument("--task-id", default=None,
                       help="Custom task ID")
    p_sub.add_argument("--agent", default=None,
                       help="Owning agent ID")
    p_sub.add_argument("--deadline-hours", type=float, default=None,
                       help="Deadline in hours from now")
    p_sub.set_defaults(func=cmd_submit)

    # schedule
    p_sched = sub.add_parser("schedule", help="Show current schedule")
    p_sched.set_defaults(func=cmd_schedule)

    # models
    p_models = sub.add_parser("models", help="List model tiers")
    p_models.set_defaults(func=cmd_models)

    # cost
    p_cost = sub.add_parser("cost", help="Show cost analysis")
    p_cost.add_argument("--report", choices=["daily", "weekly"], default="daily",
                        help="Report period (default: daily)")
    p_cost.set_defaults(func=cmd_cost)

    # optimize
    p_opt = sub.add_parser("optimize", help="Suggest cost optimizations")
    p_opt.set_defaults(func=cmd_optimize)

    # onboard
    p_onb = sub.add_parser("onboard", help="Set up the agent")
    p_onb.add_argument("--budget", type=float, default=None,
                       help="Set daily budget")
    p_onb.set_defaults(func=cmd_onboard)

    # status
    p_stat = sub.add_parser("status", help="Show scheduler status")
    p_stat.set_defaults(func=cmd_status)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
