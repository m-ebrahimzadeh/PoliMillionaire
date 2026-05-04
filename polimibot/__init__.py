"""PoliMiBot — chatbot for the PoliMillionaire quiz, this package becomes."""
from .config import CATEGORIES, PATHS, RUNTIME, Category, CompetitionInfo, PathConfig, RuntimeConfig
from .game import AnswerOutcome, GameAdapter, GameQuestion, GameSummary

__version__ = "0.0.2"
__all__ = [
    # config
    "CATEGORIES", "PATHS", "RUNTIME",
    "Category", "CompetitionInfo", "PathConfig", "RuntimeConfig",
    # game
    "GameAdapter", "GameQuestion", "AnswerOutcome", "GameSummary",
]