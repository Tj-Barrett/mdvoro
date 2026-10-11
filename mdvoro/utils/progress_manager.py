"""
Progress manager for MDvoro
Provides cleaner, consolidated progress output using rich
"""
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from contextlib import contextmanager

class ProgressManager:
    """Singleton progress manager for consistent output across MDvoro"""

    _instance = None
    _progress = None
    _verbose = True

    @classmethod
    def set_verbose(cls, verbose=True):
        """Enable or disable verbose output"""
        cls._verbose = verbose

    @classmethod
    @contextmanager
    def get_progress(cls):
        """Get or create a Progress instance"""
        if cls._progress is None:
            cls._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                transient=False
            )

        with cls._progress as progress:
            yield progress

    @classmethod
    def print(cls, message, level="info"):
        """Print a message if verbose is enabled"""
        if cls._verbose:
            prefix = {
                "info": "  ",
                "success": "  ✓ ",
                "warning": "  ⚠ ",
                "error": "  ✗ "
            }.get(level, "  ")
            print(f"{prefix}{message}")

# Global convenience function
def get_progress_manager():
    """Get the global ProgressManager instance"""
    return ProgressManager
