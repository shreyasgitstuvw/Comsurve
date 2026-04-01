"""
Daily request counter for news APIs with strict quotas (e.g. newsdata.io: 200 req/day).
Stores the counter in a small SQLite-backed JSON file to survive restarts.
Raises RateLimitExceeded before hitting the API so quota is never exceeded.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path


class RateLimitExceeded(Exception):
    pass


class DailyRateLimiter:
    """
    Tracks API call counts per day per source in a flat JSON file.
    Thread-safe for single-process use (APScheduler runs jobs sequentially by default).
    """

    def __init__(self, source: str, daily_limit: int, state_file: str = ".rate_limit_state.json"):
        self.source = source
        self.daily_limit = daily_limit
        self.state_file = Path(state_file)

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2))

    def _today(self) -> str:
        return date.today().isoformat()

    def get_used(self) -> int:
        state = self._load()
        return state.get(self.source, {}).get(self._today(), 0)

    def get_remaining(self) -> int:
        return max(0, self.daily_limit - self.get_used())

    def check_and_increment(self, count: int = 1) -> None:
        """
        Check if `count` requests can be made. If yes, increment counter.
        Raises RateLimitExceeded if quota would be exceeded.
        """
        state = self._load()
        today = self._today()
        source_state = state.setdefault(self.source, {})
        used = source_state.get(today, 0)

        if used + count > self.daily_limit:
            raise RateLimitExceeded(
                f"{self.source}: daily limit {self.daily_limit} reached "
                f"(used {used}, requested {count})"
            )

        source_state[today] = used + count
        self._save(state)

    def reset(self) -> None:
        """Manually reset counter (for testing)."""
        state = self._load()
        state[self.source] = {self._today(): 0}
        self._save(state)
