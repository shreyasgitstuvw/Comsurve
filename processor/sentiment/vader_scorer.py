"""
Thin wrapper around vaderSentiment.
Input: article text (title + description + content concatenated).
Output: compound score in [-1.0, 1.0].

Commodity keyword boosting: domain-specific terms are added to the VADER
lexicon before scoring so that disruption language scores more strongly
negative and supply-normalisation language scores more strongly positive.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Per-commodity lexicon additions.
# Positive float → boosts positive sentiment; negative → boosts negative.
# Values are in VADER's [-4, +4] range.
_COMMODITY_LEXICON: dict[str, dict[str, float]] = {
    "lng": {
        "shortage": -2.8,
        "disruption": -2.5,
        "outage": -2.5,
        "shutdown": -2.3,
        "explosion": -3.0,
        "sabotage": -3.0,
        "freeze": -2.0,
        "unplanned": -1.8,
        "maintenance": -1.2,
        "export": 1.5,
        "surplus": 1.8,
        "restart": 2.0,
        "resumed": 2.0,
        "recovery": 1.8,
    },
    "copper": {
        "strike": -2.5,
        "stoppage": -2.5,
        "shutdown": -2.3,
        "flooding": -2.0,
        "collapse": -3.0,
        "ore": 0.5,
        "inventory": 0.8,
        "deficit": -2.2,
        "surplus": 1.8,
        "mine": 0.3,
        "smelter": 0.3,
        "resumption": 2.0,
        "ramp-up": 1.8,
    },
    "soybeans": {
        "drought": -2.8,
        "frost": -2.5,
        "flood": -2.3,
        "crop": 0.5,
        "harvest": 1.5,
        "bumper": 2.0,
        "export": 1.5,
        "ban": -2.5,
        "embargo": -2.8,
        "record": 1.2,
        "disease": -2.5,
        "infestation": -2.8,
        "dryness": -2.0,
        "rainfall": 1.2,
    },
}


def score(text: str, commodity: str | None = None) -> float:
    """
    Returns VADER compound score for the given text.
    compound is in [-1.0, 1.0]:
      >= 0.05  → positive
      <= -0.05 → negative
      else     → neutral

    If commodity is provided, domain-specific term weights are temporarily
    injected into the VADER lexicon before scoring.
    """
    if not text or not text.strip():
        return 0.0

    boosts = _COMMODITY_LEXICON.get(commodity or "", {})
    if boosts:
        # Inject commodity-specific terms
        original = {}
        for term, val in boosts.items():
            original[term] = _analyzer.lexicon.get(term)
            _analyzer.lexicon[term] = val
        try:
            result = _analyzer.polarity_scores(text)["compound"]
        finally:
            # Restore original lexicon state
            for term, orig_val in original.items():
                if orig_val is None:
                    _analyzer.lexicon.pop(term, None)
                else:
                    _analyzer.lexicon[term] = orig_val
        return result

    return _analyzer.polarity_scores(text)["compound"]


def score_article(title: str, description: str = "", content: str = "",
                  commodity: str | None = None) -> float:
    """Combine article fields, score the concatenation with optional commodity boosting."""
    parts = [p for p in [title, description, content] if p]
    full_text = " ".join(parts)
    return score(full_text, commodity=commodity)
