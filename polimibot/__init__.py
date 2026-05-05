"""PoliMiBot — chatbot for the PoliMillionaire quiz, this package becomes."""
from .config import CATEGORIES, PATHS, RUNTIME, Category, CompetitionInfo, PathConfig, RuntimeConfig
from .game import AnswerOutcome, GameAdapter, GameQuestion, GameSummary
from .logging_utils import GameSummaryRecord, QuestionRecord, RunLogger, RunManifest, load_jsonl
from .strategies import RandomStrategy, Strategy, StrategyInput, StrategyOutput
from .runner import GameResult, play_game
from .logging_utils import NullLogger

__version__ = "0.0.5"
__all__ = [
    "CATEGORIES", "PATHS", "RUNTIME",
    "Category", "CompetitionInfo", "PathConfig", "RuntimeConfig",
    "GameAdapter", "GameQuestion", "AnswerOutcome", "GameSummary",
    "RunLogger", "RunManifest", "NullLogger", "QuestionRecord", "GameSummaryRecord", "load_jsonl",
    "Strategy", "StrategyInput", "StrategyOutput", "RandomStrategy",
    "GameResult", "play_game",
]