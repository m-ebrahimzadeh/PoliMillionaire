"""PoliMiBot — chatbot for the PoliMillionaire quiz, this package becomes."""
from .game import AnswerOutcome, GameAdapter, GameQuestion, GameSummary

__version__ = "0.0.1"
__all__ = ["GameAdapter", "GameQuestion", "AnswerOutcome", "GameSummary"]