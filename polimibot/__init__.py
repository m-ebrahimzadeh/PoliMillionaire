"""PoliMiBot — chatbot for the PoliMillionaire quiz, this package becomes."""
from .config import CATEGORIES, PATHS, RUNTIME, Category, CompetitionInfo, PathConfig, RuntimeConfig
from .game import AnswerOutcome, GameAdapter, GameQuestion, GameSummary
from .logging_utils import GameSummaryRecord, QuestionRecord, RunLogger, RunManifest, load_jsonl

__version__ = "0.0.3"
__all__ = [
    "CATEGORIES", "PATHS", "RUNTIME",
    "Category", "CompetitionInfo", "PathConfig", "RuntimeConfig",
    "GameAdapter", "GameQuestion", "AnswerOutcome", "GameSummary",
    "RunLogger", "RunManifest", "QuestionRecord", "GameSummaryRecord", "load_jsonl",
]