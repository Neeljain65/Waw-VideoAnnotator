from functools import lru_cache

from .analyzer import VideoAnnotator


@lru_cache(maxsize=1)
def get_video_annotator() -> VideoAnnotator:
    """Return a singleton video annotator instance."""
    return VideoAnnotator()
