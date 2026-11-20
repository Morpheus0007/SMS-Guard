"""Root re-export for jury auto-evaluation (`from detector import check`)."""

from src.detector import check, check_detailed, load_model_info, reload_model

__all__ = ["check", "check_detailed", "load_model_info", "reload_model"]
