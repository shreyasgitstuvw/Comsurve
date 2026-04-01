"""
Single source of truth for all configuration and secrets.
All services import `settings` from here — never use os.getenv directly.

Secret tiers:
  CRITICAL  — system cannot function without these; startup is aborted
  DEGRADED  — one data source is disabled without these; warnings only
  OPTIONAL  — quality improvements; silently skipped when absent
"""

import sys

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Secret tier definitions ───────────────────────────────────────────────────
# Keyed by .env variable name → human label used in startup diagnostics.

CRITICAL_KEYS: dict[str, str] = {
    "gemini_api_key": "GEMINI_API_KEY (AI embedding + causality — core pipeline)",
}

DEGRADED_KEYS: dict[str, str] = {
    "newsdata_api_key":    "NEWSDATA_API_KEY (primary news source)",
    "aisstream_api_key":   "AISSTREAM_API_KEY (AIS vessel tracking)",
    "fred_api_key":        "FRED_API_KEY (historical price data)",
    "eia_api_key":         "EIA_API_KEY (energy price data)",
}

OPTIONAL_KEYS: dict[str, str] = {
    "newsapi_ai_key":        "NEWSAPI_AI_KEY (news fallback)",
    "commodities_api_key":   "COMMODITIES_API_KEY (price fallback)",
    "opensky_client_id":     "OPENSKY_CLIENT_ID (authenticated aircraft data)",
    "opensky_client_secret": "OPENSKY_CLIENT_SECRET (authenticated aircraft data)",
    "cdse_username":         "CDSE_USERNAME (satellite product download)",
    "cdse_password":         "CDSE_PASSWORD (satellite product download)",
    "mcei_api_key":          "MCEI_API_KEY (API authentication — required in production)",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── News ──────────────────────────────────────────────────────────────────
    newsdata_api_key: str = ""
    newsapi_ai_key: str = ""

    # ── Price Real-Time ───────────────────────────────────────────────────────
    commodities_api_key: str = ""

    # ── Price Historical ──────────────────────────────────────────────────────
    fred_api_key: str = ""
    eia_api_key: str = ""

    # ── Ship Tracking (aisstream.io) ──────────────────────────────────────────
    aisstream_api_key: str = ""

    # ── Aircraft ──────────────────────────────────────────────────────────────
    opensky_client_id: str = ""
    opensky_client_secret: str = ""

    # ── Satellite ─────────────────────────────────────────────────────────────
    cdse_username: str = ""
    cdse_password: str = ""

    # ── AI ────────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── Application ───────────────────────────────────────────────────────────
    db_path: str = "mcei.db"
    log_level: str = "INFO"
    # Set to a file path to enable rotating file logging (10 MB, 5 backups).
    # Leave empty to log to stdout only (default, suitable for containers).
    log_file: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    qdrant_path: str = "./qdrant_data"

    # ── Alerting ──────────────────────────────────────────────────────────────
    # POST webhook called on consecutive job failures or quota exhaustion.
    # Leave empty to disable (alerts still appear in logs).
    mcei_alert_webhook_url: str = ""

    # ── Security ──────────────────────────────────────────────────────────────
    # Set to a strong random string in production (e.g. openssl rand -hex 32).
    # Empty string disables auth (development only).
    mcei_api_key: str = ""

    # Comma-separated list of allowed CORS origins.
    # Defaults to Streamlit localhost; override in production.
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def validate_secrets(abort_on_critical: bool = True) -> dict[str, list[str]]:
    """
    Check all secret tiers against the loaded settings.
    Returns {"critical": [...missing], "degraded": [...missing], "optional": [...missing]}.
    If abort_on_critical=True and any CRITICAL keys are missing, prints a clear
    error and exits the process (prevents silent misconfiguration in production).
    """
    missing: dict[str, list[str]] = {"critical": [], "degraded": [], "optional": []}

    for key, label in CRITICAL_KEYS.items():
        if not getattr(settings, key, ""):
            missing["critical"].append(label)

    for key, label in DEGRADED_KEYS.items():
        if not getattr(settings, key, ""):
            missing["degraded"].append(label)

    for key, label in OPTIONAL_KEYS.items():
        if not getattr(settings, key, ""):
            missing["optional"].append(label)

    # Print diagnostics regardless
    if missing["critical"]:
        print("\n[MCEI] STARTUP FAILED — missing CRITICAL secrets:", file=sys.stderr)
        for label in missing["critical"]:
            print(f"  ✗  {label}", file=sys.stderr)
        print("[MCEI] Add these to your .env file and restart.\n", file=sys.stderr)
        if abort_on_critical:
            sys.exit(1)

    if missing["degraded"]:
        print("[MCEI] WARNING — missing keys (affected data sources will be skipped):",
              file=sys.stderr)
        for label in missing["degraded"]:
            print(f"  !  {label}", file=sys.stderr)

    if missing["optional"]:
        print("[MCEI] INFO — optional keys not set (reduced functionality):",
              file=sys.stderr)
        for label in missing["optional"]:
            print(f"  -  {label}", file=sys.stderr)

    return missing


settings = Settings()
