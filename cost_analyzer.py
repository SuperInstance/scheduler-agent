#!/usr/bin/env python3
"""
Cost Analyzer — track API costs, forecast spending, recommend optimisations.

Plugs into FleetScheduler to provide financial visibility.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field

from models import FLEET_MODELS, ModelTier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CostEntry:
    """A single recorded API cost event."""
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    tier: ModelTier = ModelTier.CHEAP
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    task_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class BudgetForecast:
    """Projected spending for a future window."""
    period: str                  # e.g. "daily", "weekly"
    projected_spend: float
    budget_limit: float
    overrun_risk: float          # 0.0 – 1.0
    recommendation: str


# ---------------------------------------------------------------------------
# Cost Analyzer
# ---------------------------------------------------------------------------

class CostAnalyzer:
    """Track, analyse, and forecast API costs across model tiers.

    Parameters
    ----------
    daily_budget
        The hard cap for spending in a single UTC day.
    """

    def __init__(self, daily_budget: float = 1.0) -> None:
        self.daily_budget: float = daily_budget
        self._entries: List[CostEntry] = []
        self._tier_spend: Dict[ModelTier, float] = {t: 0.0 for t in ModelTier}
        self._agent_spend: Dict[str, float] = {}

    # ---- recording --------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> CostEntry:
        """Record a cost event and return the entry."""
        profile = FLEET_MODELS.get(model)
        if profile is None:
            return CostEntry()  # unknown model — no cost

        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * profile.cost_per_1k_tokens

        entry = CostEntry(
            model=model,
            tier=profile.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_id=task_id,
            agent_id=agent_id,
        )

        self._entries.append(entry)
        self._tier_spend[profile.tier] += cost
        if agent_id:
            self._agent_spend[agent_id] = (
                self._agent_spend.get(agent_id, 0.0) + cost
            )
        return entry

    # ---- queries ----------------------------------------------------------

    def total_spent(self) -> float:
        return sum(e.cost for e in self._entries)

    def spent_today(self) -> float:
        """Sum of costs whose timestamp falls in the current UTC day."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        return sum(e.cost for e in self._entries if e.timestamp >= today_start)

    def spent_in_last(self, hours: float = 24.0) -> float:
        cutoff = time.time() - hours * 3600
        return sum(e.cost for e in self._entries if e.timestamp >= cutoff)

    def tier_breakdown(self) -> Dict[str, float]:
        """Return spending per tier as ``{tier_name: cost}``."""
        return {t.name: round(c, 6) for t, c in self._tier_spend.items()}

    def agent_breakdown(self) -> Dict[str, float]:
        """Return spending per agent."""
        return {k: round(v, 6) for k, v in self._agent_spend.items()}

    def top_models(self, n: int = 5) -> List[Dict[str, Any]]:
        """Most expensive models by aggregate spend."""
        model_totals: Dict[str, float] = {}
        model_counts: Dict[str, int] = {}
        for e in self._entries:
            model_totals[e.model] = model_totals.get(e.model, 0.0) + e.cost
            model_counts[e.model] = model_counts.get(e.model, 0) + 1

        ranked = sorted(model_totals.items(), key=lambda kv: -kv[1])[:n]
        return [
            {
                "model": name,
                "total_cost": round(cost, 6),
                "calls": model_counts[name],
                "avg_cost": round(cost / model_counts[name], 6),
            }
            for name, cost in ranked
        ]

    # ---- forecasting ------------------------------------------------------

    def forecast_daily(self) -> BudgetForecast:
        """Project end-of-day spend based on current rate."""
        spent = self.spent_today()
        now = datetime.now(timezone.utc)
        day_fraction = (now.hour * 60 + now.minute) / 1440.0
        if day_fraction < 0.01:
            projected = self.daily_budget  # not enough data yet
        else:
            projected = spent / day_fraction

        overrun = max(0.0, (projected - self.daily_budget) / self.daily_budget)
        recommendation = self._forecast_recommendation(projected, self.daily_budget)

        return BudgetForecast(
            period="daily",
            projected_spend=round(projected, 6),
            budget_limit=self.daily_budget,
            overrun_risk=min(overrun, 1.0),
            recommendation=recommendation,
        )

    def forecast_weekly(self) -> BudgetForecast:
        """Project weekly spend based on today's rate."""
        daily = self.forecast_daily()
        weekly_projected = daily.projected_spend * 7
        weekly_budget = self.daily_budget * 7
        overrun = max(0.0, (weekly_projected - weekly_budget) / weekly_budget)
        recommendation = self._forecast_recommendation(weekly_projected, weekly_budget)

        return BudgetForecast(
            period="weekly",
            projected_spend=round(weekly_projected, 6),
            budget_limit=weekly_budget,
            overrun_risk=min(overrun, 1.0),
            recommendation=recommendation,
        )

    # ---- optimisation recommendations -------------------------------------

    def recommend(self) -> List[Dict[str, Any]]:
        """Generate a list of cost-optimization recommendations."""
        recs: List[Dict[str, Any]] = []

        # 1. Tier-based saving potential
        for tier in [ModelTier.EXPERT, ModelTier.REASONER]:
            spent = self._tier_spend[tier]
            if spent < 0.001:
                continue
            cheaper = ModelTier(tier - 1) if tier.value > 0 else tier
            cheaper_profile = next(
                (p for p in FLEET_MODELS.values() if p.tier == cheaper), None
            )
            if cheaper_profile is None:
                continue
            ratio = cheaper_profile.cost_per_1k_tokens / next(
                p.cost_per_1k_tokens
                for p in FLEET_MODELS.values() if p.tier == tier
            )
            savings = spent * (1 - ratio)
            recs.append({
                "type": "tier_downgrade",
                "severity": "medium",
                "from_tier": tier.name,
                "to_tier": cheaper.name,
                "potential_savings": round(savings, 6),
                "message": (
                    f"Downgrading {tier.name} → {cheaper.name} could save "
                    f"${savings:.4f} with acceptable quality trade-off."
                ),
            })

        # 2. Agent cost concentration
        for agent, spend in self._agent_spend.items():
            share = spend / self.daily_budget if self.daily_budget > 0 else 0
            if share > 0.5:
                recs.append({
                    "type": "redistribute_agents",
                    "severity": "medium",
                    "agent": agent,
                    "share": round(share, 3),
                    "message": (
                        f"Agent '{agent}' accounts for {share:.0%} of daily "
                        f"budget. Consider load balancing."
                    ),
                })

        # 3. Budget pacing
        forecast = self.forecast_daily()
        if forecast.overrun_risk > 0.2:
            recs.append({
                "type": "pace_spending",
                "severity": "high",
                "overrun_risk": round(forecast.overrun_risk, 3),
                "message": forecast.recommendation,
            })

        if not recs:
            recs.append({
                "type": "all_clear",
                "severity": "info",
                "message": "Spending is healthy — no action needed.",
            })

        return recs

    # ---- reports ----------------------------------------------------------

    def report_daily(self) -> Dict[str, Any]:
        """Full daily cost report."""
        return {
            "period": "daily",
            "spent_today": round(self.spent_today(), 6),
            "budget_limit": self.daily_budget,
            "remaining": round(self.daily_budget - self.spent_today(), 6),
            "forecast": self.forecast_daily(),
            "tier_breakdown": self.tier_breakdown(),
            "agent_breakdown": self.agent_breakdown(),
            "top_models": self.top_models(5),
            "recommendations": self.recommend(),
        }

    def report_weekly(self) -> Dict[str, Any]:
        """Full weekly cost report."""
        return {
            "period": "weekly",
            "spent_7d": round(self.spent_in_last(168), 6),
            "budget_7d": self.daily_budget * 7,
            "forecast": self.forecast_weekly(),
            "tier_breakdown": self.tier_breakdown(),
            "agent_breakdown": self.agent_breakdown(),
            "top_models": self.top_models(5),
            "recommendations": self.recommend(),
        }

    # ---- reset ------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded entries (e.g. at day boundary)."""
        self._entries.clear()
        self._tier_spend = {t: 0.0 for t in ModelTier}
        self._agent_spend.clear()

    # ---- internal ---------------------------------------------------------

    @staticmethod
    def _forecast_recommendation(projected: float, budget: float) -> str:
        ratio = projected / budget if budget > 0 else 0
        if ratio < 0.5:
            return "Under-utilised — opportunity to increase task throughput."
        if ratio < 0.8:
            return "On track — spending is within healthy range."
        if ratio < 1.0:
            return "Approaching limit — consider deferring low-priority tasks."
        return (
            f"Over budget! Projected ${projected:.4f} vs ${budget:.4f}. "
            "Immediately defer non-critical tasks or reduce model tiers."
        )
