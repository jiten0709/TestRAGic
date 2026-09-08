import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

def load_environment():
    """Load environment variables from .env file"""
    # Load from project root .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Also try from src/.env
    src_env_path = Path(__file__).parent.parent / ".env"
    if src_env_path.exists():
        load_dotenv(src_env_path)

def _session_get(key: str):
    """Read session state defensively -- these helpers also run outside Streamlit."""
    try:
        if key in st.session_state and st.session_state[key]:
            return st.session_state[key]
    except Exception:
        pass
    return None

def _session_set(key: str, value):
    try:
        st.session_state[key] = value
    except Exception:
        pass

def get_llm_api_key():
    """API key for direct-OpenAI mode: session state, then OPENAI_API_KEY."""
    return _session_get('openai_api_key') or os.getenv('OPENAI_API_KEY')

def set_llm_api_key(api_key: str):
    """Store the key in session state and the environment (never on disk)."""
    _session_set('openai_api_key', api_key)
    os.environ["OPENAI_API_KEY"] = api_key

def get_llm_model():
    """Chat model override: session state, then $OMNIROUTE_LLM_MODEL. May be None."""
    return _session_get('llm_model') or os.getenv('OMNIROUTE_LLM_MODEL')

def set_llm_model(model: str):
    _session_set('llm_model', model)
    os.environ["OMNIROUTE_LLM_MODEL"] = model

def get_embedding_model():
    """Local embedding model override: session state, then $EMBEDDING_MODEL.

    Not an OMNIROUTE_* variable: embeddings never touch the gateway. See
    src/utils/embeddings.py."""
    return _session_get('embedding_model') or os.getenv('EMBEDDING_MODEL')

def set_embedding_model(model: str):
    _session_set('embedding_model', model)
    os.environ["EMBEDDING_MODEL"] = model

def get_vision_model():
    """Model for captioning video frames: session state, then $OMNIROUTE_VISION_MODEL.

    None means the OCR path reads text only and never calls a vision model. There is
    no default on purpose -- see `video_ocr.vision_model` for why guessing an
    `auto/*vision` route is worse than leaving it off."""
    return _session_get('vision_model') or os.getenv('OMNIROUTE_VISION_MODEL')

def set_vision_model(model: str):
    _session_set('vision_model', model)
    if model:
        os.environ["OMNIROUTE_VISION_MODEL"] = model
    else:
        os.environ.pop("OMNIROUTE_VISION_MODEL", None)

def get_gateway_base_url():
    """OmniRoute gateway URL. Its presence is what selects OmniRoute mode."""
    return _session_get('omniroute_base_url') or os.getenv('OMNIROUTE_BASE_URL')

def set_gateway_base_url(base_url: str):
    _session_set('omniroute_base_url', base_url)
    if base_url:
        os.environ["OMNIROUTE_BASE_URL"] = base_url
    else:
        os.environ.pop("OMNIROUTE_BASE_URL", None)

def get_gateway_api_key():
    return _session_get('omniroute_api_key') or os.getenv('OMNIROUTE_API_KEY')

def set_gateway_api_key(api_key: str):
    _session_set('omniroute_api_key', api_key)
    if api_key:
        os.environ["OMNIROUTE_API_KEY"] = api_key
    else:
        os.environ.pop("OMNIROUTE_API_KEY", None)

# Backward-compatible aliases -- existing call sites in app.py and sidebar.py.
get_openai_api_key = get_llm_api_key
set_openai_api_key = set_llm_api_key
