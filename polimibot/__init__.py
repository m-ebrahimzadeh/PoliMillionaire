"""PoliMiBot — chatbot for the PoliMillionaire quiz, this package becomes."""
from .config import CATEGORIES, PATHS, RUNTIME, Category, CompetitionInfo, PathConfig, RuntimeConfig
from .game import AnswerOutcome, GameAdapter, GameQuestion, SessionRecord
from .logging_utils import GameSummaryRecord, QuestionRecord, RunLogger, RunManifest, load_jsonl
from .strategies import RandomStrategy, Strategy, StrategyInput, StrategyOutput
from .runner import GameResult, play_game
from .logging_utils import NullLogger
from .models import LLM, LLMSpec, AnswerProbabilities

__version__ = "0.1.0"
__all__ = [
    "CATEGORIES", "PATHS", "RUNTIME",
    "Category", "CompetitionInfo", "PathConfig", "RuntimeConfig",
    "GameAdapter", "GameQuestion", "AnswerOutcome", "SessionRecord",
    "RunLogger", "RunManifest", "NullLogger", "QuestionRecord", "GameSummaryRecord", "load_jsonl",
    "Strategy", "StrategyInput", "StrategyOutput", "RandomStrategy",
    "GameResult", "play_game",
    "LLM", "LLMSpec", "AnswerProbabilities"
]