"""
Utils Module - Utility functions and helper classes

This module provides utility functions for:
- Video processing and analysis
- Playwright script conversion
- Data processing and validation
"""

# Use simplified imports to avoid NLTK issues
try:
    from .video_processor import VideoProcessor
    print("✅ VideoProcessor imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import VideoProcessor: {e}")
    VideoProcessor = None

try:
    from .playwright_converter import PlaywrightConverter
    print("✅ PlaywrightConverter imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import PlaywrightConverter: {e}")
    PlaywrightConverter = None

# Only export available components
__all__ = []

if VideoProcessor:
    __all__.append("VideoProcessor")
if PlaywrightConverter:
    __all__.append("PlaywrightConverter")

# Utility configuration
DEFAULT_CHUNK_SIZE = 30  # seconds
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "auto"]