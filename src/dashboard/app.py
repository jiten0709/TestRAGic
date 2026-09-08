import datetime
import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from src.agents import data_ingestion
from src.agents.data_ingestion import DataIngestionAgent
from src.agents.test_generator import (
    MAX_TEST_CASES, TestGeneratorAgent, allocate, normalise_cases,
)
from src.agents.test_executor import TestExecutorAgent
from src.utils.playwright_converter import PlaywrightConverter
from src.utils.config import (
    load_environment, get_openai_api_key, set_openai_api_key,
    get_gateway_base_url, set_gateway_base_url,
    get_gateway_api_key, set_gateway_api_key,
    set_llm_model,
)
from src.utils import embeddings as embeddings_mod
from src.utils import provider
import traceback
from src.dashboard.components.sidebar import render_sidebar
from src.utils.logging_setup import get_logger, run_context

logger = get_logger(__name__, log_file="dashboard.log")

def main():
    # Load environment variables
    load_environment()
    
    # Initialize session state
    initialize_session_state()
    
    st.set_page_config(
        page_title="TestRAGic - Turning Inputs into Test Cases — Like Magic, but Smarter",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Either an OmniRoute gateway or an OpenAI key is enough to run.
    if not provider.is_configured():
        show_api_key_warning()
        return
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🤖 TestRAGic - Turning Inputs into Test Cases</div>', unsafe_allow_html=True)
    st.markdown("*Automated end-to-end frontend test case generation, execution, and reporting*")
    
    # USE THE COMPREHENSIVE SIDEBAR COMPONENT
    selected = render_sidebar()
    
    # Map sidebar selection to page routing
    page_mapping = {
        "generation": "🎯 Test Generation",
        "execution": "🚀 Test Execution", 
        "results": "📊 Results & Reports",
        "settings": "⚙️ Settings"
    }
    
    selected_page = page_mapping.get(selected, "🎯 Test Generation")
    
    # Route to appropriate page
    if selected_page == "🎯 Test Generation":
        render_test_generation_page()
    elif selected_page == "🚀 Test Execution":
        render_test_execution_page()
    elif selected_page == "📊 Results & Reports":
        render_results_page()
    else:
        render_settings_page()

def initialize_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = get_openai_api_key() or ""
    
    if 'video_url' not in st.session_state:
        st.session_state.video_url = ""
    
    if 'test_categories' not in st.session_state:
        st.session_state.test_categories = ["Core User Flows", "Edge Cases"]
    
    if 'priority_levels' not in st.session_state:
        st.session_state.priority_levels = ["Critical", "High"]

    if 'max_test_cases' not in st.session_state:
        st.session_state.max_test_cases = 10

    if 'priority_distribution' not in st.session_state:
        st.session_state.priority_distribution = "Even"
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = provider.default_chat_model()

    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = embeddings_mod.default_model()
    
    if 'default_browser' not in st.session_state:
        st.session_state.default_browser = "Chromium"
    
    if 'default_timeout' not in st.session_state:
        st.session_state.default_timeout = 30
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {
            'in_progress': False,
            'step': '',
            'progress': 0
        }

def show_api_key_warning():
    """Explain both ways to configure a provider. Neither is set."""
    st.error("🔌 No AI provider configured")
    st.markdown("""
    TestRAGic needs somewhere to send its requests. Pick either:

    **A. OmniRoute gateway (recommended — 350+ providers, one endpoint)**

    ```bash
    npm i -g omniroute && omniroute      # boots on http://localhost:20128
    ```
    Then set `OMNIROUTE_BASE_URL=http://localhost:20128/v1` in your `.env`.
    A local instance needs no API key.

    **B. OpenAI directly** — set `OPENAI_API_KEY` in your `.env`, or enter it below.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OmniRoute gateway")
        base_url = st.text_input(
            "Gateway base URL:",
            value=get_gateway_base_url() or "",
            placeholder=provider.DEFAULT_GATEWAY_URL,
        )
        if st.button("🔌 Use gateway", use_container_width=True):
            if base_url:
                set_gateway_base_url(base_url)
                provider.reset_clients()
                st.success("✅ Gateway configured! Reloading…")
                st.rerun()
            else:
                st.error("Please enter a gateway URL")

    with col2:
        st.subheader("OpenAI direct")
        api_key_input = st.text_input(
            "OpenAI API Key:",
            type="password",
            placeholder="sk-proj-...",
        )
        if st.button("💾 Save API key", type="primary", use_container_width=True):
            if api_key_input:
                set_openai_api_key(api_key_input)
                provider.reset_clients()
                st.success("✅ API Key saved! Reloading…")
                st.rerun()
            else:
                st.error("Please enter a valid API key")

def test_with_mock_data():
    """Test test generation with mock video data"""
    return {
        "success": True,
        "video_info": {
            "video_id": "test123",
            "title": "Recruiter.ai Demo - User Registration Process",
            "description": "This video shows how to register on Recruiter.ai platform"
        },
        "transcript": """
        Welcome to Recruiter.ai. Today I'll show you how to create an account.
        First, navigate to the signup page by clicking the 'Sign Up' button.
        Enter your email address in the email field.
        Create a strong password and confirm it.
        Select your account type - either Recruiter or Job Seeker.
        Fill in your profile information including name and company.
        Click the 'Create Account' button to complete registration.
        You'll receive a confirmation email to verify your account.
        Once verified, you can log in and start using the platform.
        """,
        "chunks": ["Navigation to signup", "Email entry", "Password creation", "Account type selection", "Profile completion", "Account creation"],
        "vector_store_info": {"status": "created", "chunks_count": 6}
    }

def validate_youtube_url(url):
    """Validate YouTube URL format"""
    import re
    
    if url.lower() == "test":
        return True, "Mock data URL"
    
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?(www\.)?youtu\.be/',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True, "Valid YouTube URL"
    
    return False, "Invalid YouTube URL format"

def debug_video_processing(url):
    """Debug video processing step by step"""
    st.write("🔍 Debug: Starting video processing...")
    
    data_agent = DataIngestionAgent()
    
    # Step 1: Test download
    st.write("Step 1: Testing download...")
    download_result = data_agent._download_video(url)
    st.json(download_result)
    
    if download_result.get("success"):
        # Step 2: Test transcript
        st.write("Step 2: Testing transcript...")
        video_id = download_result.get("video_id")
        video_path = download_result.get("video_path")
        transcript_result = data_agent._extract_transcript(video_id, video_path)
        st.json(transcript_result)
        
        if transcript_result.get("success"):
            # Step 3: Test chunking
            st.write("Step 3: Testing chunking...")
            chunks = data_agent._intelligent_chunking(transcript_result)
            st.write(f"Created {len(chunks)} chunks")
            st.json(chunks[:2])  # Show first 2 chunks

def fix_duplicate_test_ids():
    """Fix duplicate test case IDs in existing files"""
    st.subheader("🔧 Fix Duplicate Test IDs")
    
    test_files = load_available_test_files()
    
    if not test_files:
        st.warning("No test files found to fix.")
        return
    
    # Show current files with potential issues
    st.write("**Available test files:**")
    for test_file in test_files:
        if test_file['path'] != 'session_state':
            st.write(f"📄 {test_file['name']} - {test_file['test_count']} test cases")
    
    if st.button("🔄 Fix Duplicate IDs in All Files", type="primary"):
        fixed_count = 0
        
        for test_file in test_files:
            if test_file['path'] != 'session_state':
                try:
                    # Load the file
                    with open(test_file['path'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Fix duplicate IDs
                    test_cases = data.get('test_cases', [])
                    
                    # Create backup
                    backup_path = f"{test_file['path']}.backup"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    # Generate unique IDs
                    for i, test_case in enumerate(test_cases, 1):
                        test_case['ID'] = f"TC{i:03d}"
                        test_case['id'] = f"TC{i:03d}"  # Also fix lowercase 'id'
                    
                    # Update metadata
                    if 'metadata' not in data:
                        data['metadata'] = {}
                    
                    data['metadata']['fixed_at'] = datetime.datetime.now().isoformat()
                    data['metadata']['total_cases'] = len(test_cases)
                    data['metadata']['fixed_duplicate_ids'] = True
                    
                    # Save back to file
                    with open(test_file['path'], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    fixed_count += 1
                    st.success(f"✅ Fixed {test_file['name']} (backup saved as {backup_path})")
                    
                except Exception as e:
                    st.error(f"❌ Error fixing {test_file['name']}: {e}")
        
        if fixed_count > 0:
            st.success(f"🎉 Successfully fixed {fixed_count} test files!")
            st.info("💡 Backup files were created with .backup extension")
            st.rerun()
        else:
            st.warning("No files were fixed.")

def save_generated_tests_to_file(test_cases, video_info):
    """Save generated test cases to file for future use"""
    try:
        # Create test cases directory
        test_cases_dir = Path("src/data/test_cases")
        test_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_title = video_info.get('title', 'Unknown').replace(' ', '_')[:30]
        filename = f"{video_title}_{timestamp}.json"
        
        # Save file
        file_path = test_cases_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        st.success(f"💾 Test cases saved to: {filename}")
        logger.info("saved %d test cases to %s", len(test_cases.get('test_cases', [])), file_path)
        
        # Update recent generations
        if 'recent_generations' not in st.session_state:
            st.session_state.recent_generations = []
        
        st.session_state.recent_generations.append({
            'name': filename,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'test_count': len(test_cases.get('test_cases', []))
        })
        
        return str(file_path)
        
    except Exception as e:
        logger.error("could not save test cases: %s", e, exc_info=True)
        st.warning(f"Could not save test cases to file: {e}")
        return None

def render_test_generation_page():
    st.header("🎯 Test Case Generation")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Input")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["YouTube URL", "Upload Video File"])
        
        if input_method == "YouTube URL":
            # Use session state for persistent value
            video_url = st.text_input(
                "Enter YouTube URL:",
                value=st.session_state.video_url,
                placeholder="https://youtube.com/watch?v=example or type 'test' for demo",
                help="Paste the URL of the YouTube video or type 'test' for mock data",
                key="video_url_input"
            )
            # Update session state when value changes
            if video_url != st.session_state.video_url:
                st.session_state.video_url = video_url
            
            # Validate URL in real-time
            if video_url:
                is_valid, message = validate_youtube_url(video_url)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        else:
            uploaded_file = st.file_uploader(
                "Upload video file:",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file (max 200MB)"
            )
        
        # Advanced options with session state
        with st.expander("🔧 Advanced Options"):
            test_categories = st.multiselect(
                "Test Categories to Generate:",
                ["Core User Flows", "Edge Cases", "Cross-browser", "Mobile", "Accessibility", "Performance"],
                default=st.session_state.test_categories,
                key="test_categories_input"
            )
            st.session_state.test_categories = test_categories
            
            priority_levels = st.multiselect(
                "Priority Levels:",
                ["Critical", "High", "Medium", "Low"],
                default=st.session_state.priority_levels,
                key="priority_levels_input"
            )
            st.session_state.priority_levels = priority_levels

            # One call per category costs money and time, so the budget is for the
            # whole suite. The floor tracks the category count: every selected
            # category must get at least one case, and st.slider raises rather
            # than clamps when its value falls below min_value.
            floor = max(1, len(test_categories))
            st.session_state.max_test_cases = st.slider(
                "Max test cases (total):",
                floor, MAX_TEST_CASES,
                value=max(st.session_state.max_test_cases, floor),
                key="max_test_cases_input",
                help="Split evenly across the selected categories.",
            )
            st.session_state.priority_distribution = st.radio(
                "Priority distribution:",
                ["Even", "Weighted"],
                horizontal=True,
                index=["Even", "Weighted"].index(st.session_state.priority_distribution),
                key="priority_distribution_input",
                help="How each category's share is split across the priority levels "
                     "above. Weighted: critical 4, high 3, medium 2, low 1.",
            )
            if test_categories:
                st.caption("Per category: " + " · ".join(
                    f"{c} {n}" for c, n in
                    allocate(st.session_state.max_test_cases, test_categories).items()))

            chat_models, catalog_warning = provider.list_chat_models()
            model_options = [m["id"] for m in chat_models]
            if catalog_warning:
                st.warning(f"⚠️ {catalog_warning}")

            col_model, col_refresh = st.columns([6, 1])
            with col_model:
                llm_model = st.selectbox(
                    "LLM Model:",
                    model_options,
                    # option_index, not .index(): a dynamic catalog makes the bare
                    # call a guaranteed ValueError once the default drops out.
                    index=provider.option_index(model_options, st.session_state.llm_model),
                    key="llm_model_input",
                    help="Served by the configured provider. 'auto' lets OmniRoute choose.",
                )
            with col_refresh:
                st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
                if st.button("🔄", help="Refresh the model catalog"):
                    provider.list_chat_models(refresh=True)
                    st.rerun()
            st.session_state.llm_model = llm_model
        
        # Generation button
        if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
            # Validate provider configuration
            if not provider.is_configured():
                st.error("❌ No AI provider configured. Set a gateway URL or an API key in Settings.")
                return
            
            # Validate inputs before processing
            if not st.session_state.test_categories:
                st.error("❌ Please select at least one test category.")
                return
            
            if not st.session_state.priority_levels:
                st.error("❌ Please select at least one priority level.")
                return
            
            # One run id per click, tagging every line the pipeline logs across all
            # five modules -- concurrent Streamlit sessions interleave otherwise.
            if input_method == "YouTube URL" and st.session_state.video_url:
                with run_context(f"url={st.session_state.video_url}"):
                    generate_test_cases_from_url(
                        st.session_state.video_url, 
                        st.session_state.test_categories, 
                        st.session_state.priority_levels, 
                        st.session_state.llm_model,
                        st.session_state.max_test_cases,
                        st.session_state.priority_distribution.lower(),
                    )
            elif input_method == "Upload Video File" and uploaded_file:
                with run_context(f"file={uploaded_file.name}"):
                    generate_test_cases_from_file(
                        uploaded_file, 
                        st.session_state.test_categories, 
                        st.session_state.priority_levels, 
                        st.session_state.llm_model,
                        st.session_state.max_test_cases,
                        st.session_state.priority_distribution.lower(),
                    )
            else:
                st.error("Please provide a video URL or upload a video file.")
    
    with col2:
        st.subheader("📊 Generation Statistics")
        
        # Metrics placeholder
        if 'generation_stats' in st.session_state:
            stats = st.session_state.generation_stats
            st.metric("Total Test Cases", stats.get('total_cases', 0))
            st.metric("Core Flows", stats.get('core_flows', 0))
            st.metric("Edge Cases", stats.get('edge_cases', 0))
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.1f}s")
        else:
            st.info("Generate test cases to see statistics")
        
        # Recent generations
        st.subheader("📝 Recent Generations")
        if 'recent_generations' in st.session_state and st.session_state.recent_generations:
            for gen in st.session_state.recent_generations[-5:]:
                with st.container():
                    col_name, col_count = st.columns([3, 1])
                    with col_name:
                        st.write(f"📄 {gen['name']}")
                        st.caption(f"⏰ {gen['timestamp']}")
                    with col_count:
                        st.metric("Tests", gen.get('test_count', 0))
        else:
            st.info("No recent generations")

        # Debug section
        st.markdown("---")
        st.subheader("🔍 Debug Tools")
        
        # Debug video processing
        if st.button("🔍 Debug Video Processing", use_container_width=True):
            if st.session_state.video_url:
                debug_video_processing(st.session_state.video_url)
            else:
                st.error("Please enter a video URL first")
        
        # Load mock data button
        if st.button("🧪 Load Mock Data", use_container_width=True):
            st.session_state.video_url = "test"
            mock_data = test_with_mock_data()
            st.success("✅ Mock data loaded!")
            with st.expander("📄 Mock Data Preview"):
                st.json({
                    "title": mock_data["video_info"]["title"],
                    "transcript_preview": mock_data["transcript"][:200] + "...",
                    "chunks_count": len(mock_data["chunks"])
                })
            st.rerun()

        # Fix duplicate test IDs button
        if st.button("🔧 Fix Duplicate Test IDs", use_container_width=True):
            fix_duplicate_test_ids()
        
        # Clear session data
        if st.button("🗑️ Clear Session", use_container_width=True):
            # Clear relevant session state
            keys_to_clear = ['generated_tests', 'generation_stats', 'recent_generations', 'execution_status', 'execution_log']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Session data cleared!")
            st.rerun()

# Transcript methods that produce a fixed placeholder instead of the video's words.
SYNTHETIC_TRANSCRIPT_METHODS = ("basic_fallback", "basic_fallback_file")


def warn_if_synthetic_transcript(video_content):
    """Say so when the transcript is a placeholder, not the video.

    A silent video (no audio stream), a failed download, or a missing ffmpeg all end
    at `_create_basic_transcript*`, which returns fixed boilerplate that knows only
    the filename. Every later stage succeeds on it: it chunks, it embeds, it
    retrieves, and the model writes confident test cases about a video it was never
    shown. Nothing else catches this -- `wire_retriever` stays quiet because a store
    *was* built, just from placeholder text.
    """
    if video_content.get('transcript_method') not in SYNTHETIC_TRANSCRIPT_METHODS:
        return

    reason = video_content.get('transcript_reason')
    logger.warning("synthetic transcript: method=%s cause=%s reason=%s",
                   video_content.get('transcript_method'),
                   video_content.get('transcript_cause'), reason)
    # Switch on the code, not on the reason text: the reason is trimmed for display
    # and the decisive line is exactly what gets trimmed out.
    hint = {
        data_ingestion.NO_AUDIO:
            "\n\nThis video has **no audio track**, so there is nothing to transcribe. "
            "The pipeline reads speech only — it never looks at the picture — so a silent "
            "screen recording cannot produce real test cases.",
        data_ingestion.FFMPEG_MISSING:
            "\n\n`ffmpeg` is missing; Whisper needs it to read the audio. "
            "Install it (`brew install ffmpeg`) and re-run.",
    }.get(video_content.get('transcript_cause'), "")

    st.warning(
        "📝 **The transcript is a placeholder, not this video.** Transcription failed, so "
        "generation ran on fixed boilerplate that knows only the file name. The test cases "
        "below are therefore generic and not derived from your video."
        + (f"\n\nReason: `{reason}`" if reason else "")
        + hint
    )


def wire_retriever(test_agent, data_agent, video_content):
    """Hand this run's vector store to the generator, and say so when there isn't one.

    Both generation paths route through here because the failure is otherwise
    invisible: no store means no retriever, the generator silently falls back to
    `transcript[:2000]` for every category, and the run still reports success --
    just with markedly weaker prompts. Embeddings are local now, so the usual cause
    is a model that would not load or a width mismatch against the existing store.
    """
    store_key = video_content.get('vector_store_key')
    retriever = data_agent.setup_retrieval_chain(store_key)
    test_agent.set_retriever(retriever)
    if retriever is not None:
        logger.info("RAG on, store=%s", store_key)
        return

    reason = (video_content.get('vector_store_info') or {}).get('error')
    logger.warning("RAG off, store=%s: %s -- every category prompted with transcript head",
                   store_key, reason or "no reason reported")
    st.warning(
        "🔍 **RAG is off for this run.** No vector store was available, so every "
        "category was prompted with the first 2000 characters of the transcript "
        "instead of chunks retrieved for it."
        + (f"\n\nReason: `{reason}`" if reason else "")
        + "\n\nEmbeddings run locally — check the app log for the model load, and "
        "Settings → AI Provider for the active embedding model and device."
    )


def generate_test_cases_from_url(url, categories, priorities, model,
                                 max_cases=MAX_TEST_CASES, distribution="even"):
    """Generate test cases from YouTube URL"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Validate inputs first
        if not url or url.strip() == "":
            st.error("❌ Please enter a valid YouTube URL")
            return
        
        if not categories:
            st.error("❌ Please select at least one test category")
            return
        
        # Either a gateway URL or an API key will do. There is no key format to
        # police: a local OmniRoute instance is keyless by default, and a gateway
        # key is whatever its operator chose.
        if not provider.is_configured():
            st.error("❌ No AI provider configured")
            st.info("Set an OmniRoute gateway URL or an OpenAI API key in Settings")
            return

        endpoint = provider.resolve_endpoint()
        logger.info("generating from url: categories=%s priorities=%s model=%s via %s",
                    categories, priorities, model or "default", endpoint.gateway)
        if endpoint.gateway == "openai" and endpoint.api_key:
            os.environ["OPENAI_API_KEY"] = endpoint.api_key
        
        # Initialize agents
        status_text.text("🔧 Initializing AI agents...")
        progress_bar.progress(10)
        
        data_agent = DataIngestionAgent()
        
        # Initialize TestGeneratorAgent safely
        try:
            test_agent = TestGeneratorAgent(model=model)
            st.success(f"✅ Test agent initialized with model: {model}")
        except Exception as e:
            st.error(f"❌ Failed to initialize TestGeneratorAgent: {e}")
            return
        
        # Step 1: Process video
        status_text.text("🎬 Processing video...")
        progress_bar.progress(25)
        
        # Mock input is opt-in by typing exactly "test". This used to be
        # `"test" in url.lower()`, which silently swapped canned data in for any
        # real URL containing that substring (youtube.com/watch?v=...test...).
        used_mock_input = url.strip().lower() == "test"
        if used_mock_input:
            video_content = test_with_mock_data()
            st.warning(
                "🧪 Mock input — the transcript below is canned, not from this URL. "
                "Test cases are still model-generated, but from fake source material."
            )
        else:
            video_content = data_agent.process_video_content(url)
        
        if not video_content.get("success"):
            logger.error("video processing failed: %s", video_content.get('error'))
            st.error(f"❌ Video processing failed: {video_content.get('error')}")
            return
        
        # Feed the freshly-built vector store to the generator so each category is
        # prompted with chunks relevant to it. Warns when there is no store (the
        # mock path, or a gateway that cannot embed) rather than degrading quietly.
        wire_retriever(test_agent, data_agent, video_content)
        warn_if_synthetic_transcript(video_content)
        
        # Step 2: Generate test cases
        status_text.text("🤖 Generating test cases...")
        progress_bar.progress(50)
        
        # One generation entry point. The cascade that used to sit here tried
        # generate_test_cases() and generate_tests(), neither of which exists.
        test_cases = None
        try:
            test_cases = test_agent.generate_comprehensive_tests(
                video_content, categories, priorities, max_cases, distribution
            )
        except Exception as e:
            logger.warning("generate_comprehensive_tests failed: %s", e, exc_info=True)
            st.warning(f"generate_comprehensive_tests failed: {e}")

        # If generation produced nothing, create a basic test case structure
        if not test_cases:
            logger.warning("no generator method produced cases; using canned fallbacks")
            st.warning("Using fallback test generation...")
            test_cases = create_fallback_test_cases(video_content, categories,
                                                    priorities, max_cases)
        
        # Step 3: Format and save
        status_text.text("💾 Formatting and saving...")
        progress_bar.progress(75)
        
        # Check if format method exists
        if hasattr(test_agent, 'format_test_cases'):
            try:
                formatted_cases = test_agent.format_test_cases(test_cases)
            except Exception as e:
                st.warning(f"format_test_cases failed: {e}")
                formatted_cases = format_test_cases_fallback(test_cases, test_agent)
        else:
            formatted_cases = format_test_cases_fallback(test_cases, test_agent)
        
        # Step 4: Display results
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        
        # Fold in which model embedded the transcript (from the ingestion step).
        store_info = video_content.get('vector_store_info') or {}
        embedding_attribution = store_info.get('embedding_attribution')
        if embedding_attribution and isinstance(formatted_cases, dict):
            attribution = formatted_cases.setdefault('metadata', {}).setdefault('attribution', {})
            attribution['embedding'] = {
                'provider': embedding_attribution.get('provider'),
                'model': embedding_attribution.get('model'),
                'dimensions': store_info.get('embedding_dimensions'),
            }

        # Show results
        display_generated_test_cases(formatted_cases, used_mock_input=used_mock_input)
        
        # Update session state
        total_cases = len(formatted_cases.get('test_cases', [])) if isinstance(formatted_cases, dict) else len(formatted_cases) if isinstance(formatted_cases, list) else 0
        
        logger.info("generated %d test cases from url", total_cases)

        st.session_state.generation_stats = {
            'total_cases': total_cases,
            'core_flows': len([tc for tc in formatted_cases.get('test_cases', []) if 'core' in tc.get('category', '').lower()]),
            'edge_cases': len([tc for tc in formatted_cases.get('test_cases', []) if 'edge' in tc.get('category', '').lower()]),
            'processing_time': 45.2
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        logger.error("url generation failed: %s", e, exc_info=True)
        st.error(f"❌ Error generating test cases: {str(e)}")
        
        # Show detailed error for debugging
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Suggest solutions
        st.info("""
        **Possible solutions:**
        1. Try using "test" as URL to use mock data
        2. Check if your OpenAI API key is valid
        3. Try a different YouTube URL
        4. Check your internet connection
        """)

def create_fallback_test_cases(video_content, categories, priorities,
                               max_cases=MAX_TEST_CASES):
    """Create basic test cases when the TestGeneratorAgent fails"""
    
    # Extract basic info from video content
    video_title = video_content.get('video_info', {}).get('title', 'Unknown Video')
    transcript = video_content.get('transcript', '')
    
    # Create basic test cases based on categories
    test_cases = []
    
    if 'Core User Flows' in categories:
        test_cases.extend([
            {
                'id': 'TC001',
                'title': 'Basic Navigation Test',
                'description': f'Test basic navigation functionality shown in {video_title}',
                'category': 'core_flow',
                'priority': 'critical',
                'steps': [
                    {'step': 1, 'action': 'Open application', 'expected': 'Application loads successfully'},
                    {'step': 2, 'action': 'Navigate to main page', 'expected': 'Main page displays correctly'},
                    {'step': 3, 'action': 'Verify key elements', 'expected': 'All navigation elements are visible'}
                ],
                'assertions': ['Page title is correct', 'Navigation menu is present']
            },
            {
                'id': 'TC002',
                'title': 'User Authentication Flow',
                'description': 'Test user login/signup process',
                'category': 'core_flow',
                'priority': 'critical',
                'steps': [
                    {'step': 1, 'action': 'Navigate to login page', 'expected': 'Login form is displayed'},
                    {'step': 2, 'action': 'Enter valid credentials', 'expected': 'Credentials are accepted'},
                    {'step': 3, 'action': 'Click login button', 'expected': 'User is logged in successfully'}
                ],
                'assertions': ['Login form validates input', 'Success message is shown']
            }
        ])
    
    if 'Edge Cases' in categories:
        test_cases.extend([
            {
                'id': 'TC003',
                'title': 'Invalid Input Handling',
                'description': 'Test application behavior with invalid inputs',
                'category': 'edge_case',
                'priority': 'high',
                'steps': [
                    {'step': 1, 'action': 'Enter invalid data', 'expected': 'Error message is displayed'},
                    {'step': 2, 'action': 'Verify error handling', 'expected': 'Application remains stable'}
                ],
                'assertions': ['Error messages are user-friendly', 'No system crashes occur']
            }
        ])
    
    if 'Cross-browser' in categories:
        test_cases.append({
            'id': 'TC004',
            'title': 'Cross-browser Compatibility',
            'description': 'Verify functionality across different browsers',
            'category': 'cross_browser',
            'priority': 'medium',
            'steps': [
                {'step': 1, 'action': 'Test in Chrome', 'expected': 'Functionality works in Chrome'},
                {'step': 2, 'action': 'Test in Firefox', 'expected': 'Functionality works in Firefox'},
                {'step': 3, 'action': 'Test in Safari', 'expected': 'Functionality works in Safari'}
            ],
            'assertions': ['UI renders consistently', 'All features work across browsers']
        })

    # Canonical shape, like every other path -- these are displayed and executed
    # by the same code as generated cases.
    return normalise_cases(test_cases)[:max_cases]

def render_attribution(metadata, prefix="🤖 Generated by"):
    """Render who actually answered.

    One rendering rule for every surface: the model that *served* the request is
    the headline; a requested-but-unavailable model appears only in the note. A
    requested model is never presented as if it were confirmed.
    """
    attribution = (metadata or {}).get('attribution')

    if not attribution:
        st.caption("🤖 Generated by: Unknown (saved before attribution tracking)")
        return

    note = attribution.get('note')

    # No model produced this (mock data, or an exhausted fallback chain).
    if not attribution.get('requested_model'):
        st.caption(f"⚠️ {attribution.get('display') or 'no model was called'}"
                   + (f" — {note}" if note else ""))
        return

    icon = "⚠️" if attribution.get('fallback') else "🤖"
    st.caption(f"{icon} {prefix}: {attribution.get('display') or 'unknown'}")
    if note:
        st.caption(f"↳ {note}")

    per_category = attribution.get('per_category') or []
    if len(per_category) > 1:
        with st.expander("Model details"):
            st.dataframe(pd.DataFrame(per_category), use_container_width=True)


def format_test_cases_fallback(test_cases, agent=None):
    """Format test cases when the agent's format method is not available.

    Carries attribution too, so the fallback path is not an attribution hole.
    """
    attribution = provider.summarize(getattr(agent, '_attributions', None) or [])

    if isinstance(test_cases, dict) and 'test_cases' in test_cases:
        test_cases['test_cases'] = normalise_cases(test_cases['test_cases'])
        test_cases.setdefault('metadata', {}).setdefault('attribution', attribution)
        return test_cases

    cases = normalise_cases(test_cases if isinstance(test_cases, list)
                            else ([test_cases] if test_cases else []))
    return {
        'test_cases': cases,
        'metadata': {
            'generated_at': datetime.datetime.now().isoformat(),
            'total_cases': len(cases),
            'generator': 'fallback',
            'attribution': attribution,
        }
    }

def generate_test_cases_from_file(uploaded_file, categories, priorities, model,
                                  max_cases=MAX_TEST_CASES, distribution="even"):
    """Generate test cases from uploaded video file"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize agents
        data_agent = DataIngestionAgent()
        test_agent = TestGeneratorAgent(model=model)
        
        # Step 1: Process video
        status_text.text("🎬 Processing video file...")
        progress_bar.progress(25)
        logger.info("generating from upload: %s (%d bytes) categories=%s model=%s",
                    uploaded_file.name, temp_file_path.stat().st_size, categories,
                    model or "default")
        video_content = data_agent.process_video_file(str(temp_file_path))
        
        # See the URL path: retrieval-backed context, transcript head if unavailable.
        wire_retriever(test_agent, data_agent, video_content)
        warn_if_synthetic_transcript(video_content)
        
        # Step 2: Generate test cases
        status_text.text("🤖 Generating test cases...")
        progress_bar.progress(50)
        test_cases = test_agent.generate_comprehensive_tests(
            video_content, categories, priorities, max_cases, distribution
        )
        
        # Step 3: Format and save
        status_text.text("💾 Formatting and saving...")
        progress_bar.progress(75)
        formatted_cases = test_agent.format_test_cases(test_cases)
        
        # Step 4: Display results
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        
        # Show results
        logger.info("generated %d test cases from upload",
                    len(formatted_cases.get('test_cases', [])) if isinstance(formatted_cases, dict) else 0)
        display_generated_test_cases(formatted_cases)
        
        # Cleanup
        temp_file_path.unlink()
        
    except Exception as e:
        logger.error("upload generation failed: %s", e, exc_info=True)
        st.error(f"Error generating test cases: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_generated_test_cases(test_cases, used_mock_input=False):
    """Display generated test cases in a user-friendly format"""
    st.success("✅ Test cases generated successfully!")

    metadata = test_cases.get('metadata', {})
    render_attribution(metadata)
    if used_mock_input:
        st.caption("🧪 Source material was mock data, not the supplied URL")

    embedding = (metadata.get('attribution') or {}).get('embedding')
    if embedding and embedding.get('model'):
        dims = embedding.get('dimensions')
        st.caption(f"🔡 Embedded by: {provider.pretty(embedding.get('provider'), embedding.get('model'))}"
                   + (f" ({dims} dims)" if dims else ""))

    # Auto-save to file
    video_info = metadata
    saved_file = save_generated_tests_to_file(test_cases, video_info)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", len(test_cases.get('test_cases', [])))
    with col2:
        critical_cases = len([tc for tc in test_cases.get('test_cases', []) if tc.get('priority') == 'critical'])
        st.metric("Critical", critical_cases)
    with col3:
        high_cases = len([tc for tc in test_cases.get('test_cases', []) if tc.get('priority') == 'high'])
        st.metric("High Priority", high_cases)
    with col4:
        # Calculate unique suites
        suites = set(tc.get('suite', tc.get('category', 'default')) for tc in test_cases.get('test_cases', []))
        st.metric("Test Suites", len(suites))
    
    # Store in session state
    st.session_state.generated_tests = test_cases
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Test Cases", "📄 JSON Format", "📝 Markdown"])
    
    with tab1:
        # One key schema: normalise_cases() guarantees lowercase keys, string
        # steps and unique ids, so no .get('ID', .get('id', ...)) cascade here.
        for i, tc in enumerate(test_cases.get('test_cases', [])):
            with st.expander(f"{tc['id']}: {tc['title']}", expanded=i < 3):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Description:** {tc['description'] or 'No description'}")
                    st.write(f"**Category:** {tc['category'] or 'Unknown'}")

                    if tc['steps']:
                        st.write("**Steps:**")
                        for j, step in enumerate(tc['steps'], 1):
                            st.write(f"  {j}. {step}")

                    if tc['expected_result']:
                        st.write(f"**Expected result:** {tc['expected_result']}")

                    if tc['assertions']:
                        st.write("**Assertions:**")
                        for assertion in tc['assertions']:
                            st.write(f"  • {assertion}")

                with col2:
                    priority_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    st.write(f"**Priority:** {priority_color.get(tc['priority'], '⚪')} "
                             f"{tc['priority'].title()}")
                    st.write(f"**ID:** {tc['id']}")
    
    with tab2:
        st.json(test_cases)
        
        # Download button
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(test_cases, indent=2),
            file_name=f"test_cases_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with tab3:
        markdown_content = convert_to_markdown(test_cases)
        st.markdown(markdown_content)
        
        # Download button
        st.download_button(
            label="📥 Download Markdown",
            data=markdown_content,
            file_name=f"test_cases_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def convert_to_markdown(test_cases):
    """Convert test cases to markdown format"""
    markdown = "# Test Cases\n\n"
    
    for tc in normalise_cases(test_cases.get('test_cases', [])):
        markdown += f"## {tc['id']}: {tc['title']}\n\n"
        markdown += f"**Description:** {tc['description'] or 'No description'}\n\n"
        markdown += f"**Category:** {tc['category'] or 'Unknown'}\n\n"
        markdown += f"**Priority:** {tc['priority'].title()}\n\n"

        if tc['steps']:
            markdown += "**Steps:**\n"
            for number, step in enumerate(tc['steps'], 1):
                markdown += f"{number}. {step}\n"
            markdown += "\n"

        if tc['expected_result']:
            markdown += f"**Expected result:** {tc['expected_result']}\n\n"
        
        if tc['assertions']:
            markdown += "**Assertions:**\n"
            for assertion in tc['assertions']:
                markdown += f"- {assertion}\n"
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown

def render_test_execution_page():
    st.header("🚀 Test Execution")
    
    # Load available test cases
    test_files = load_available_test_files()
    
    if not test_files:
        st.warning("No test cases found. Please generate test cases first.")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Go to Test Generation", type="primary"):
                st.session_state.page = "🎯 Test Generation"
                st.rerun()
        
        with col2:
            st.info("Generate test cases first to enable execution")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📂 Select Test Cases")
        
        # Convert test_files to display format
        file_options = [f['name'] for f in test_files]
        selected_files = st.multiselect(
            "Choose test files to execute:",
            file_options,
            default=file_options[:1] if file_options else []
        )
        
        # The application under test. Test cases describe a flow, not a host, so
        # without this every generated `page.goto()` lands on about:blank.
        base_url = st.text_input(
            "Application URL (tests run against this):",
            value=os.getenv("BASE_URL", "http://localhost:3000"),
            help="Playwright opens this URL. Relative paths in a step are resolved against it.",
        )

        # Execution options
        with st.expander("🔧 Execution Options"):
            # Exactly the three engines pytest-playwright's --browser accepts;
            # the old list offered device profiles that were never wired up.
            browsers = st.multiselect(
                "Target Browsers:",
                list(BROWSER_ENGINES),
                default=["Chromium"]
            )

            headless = st.checkbox("Run in headless mode", value=True)

            capture_options = st.multiselect(
                "Capture on failure:",
                ["Screenshots", "Videos", "Traces"],
                default=["Screenshots", "Traces"]
            )

        # Execute button
        if st.button("🎬 Execute Tests", type="primary", use_container_width=True):
            if not selected_files:
                st.error("Please select at least one test file.")
            elif not base_url.strip():
                st.error("Enter the URL of the application to test.")
            else:
                execute_playwright_tests(selected_files, browsers, headless,
                                         base_url.strip(), capture_options)
    
    with col2:
        st.subheader("📊 Execution Status")
        
        if 'execution_status' in st.session_state:
            status = st.session_state.execution_status
            st.metric("Tests Executed", status.get('executed', 0))
            st.metric("Passed", status.get('passed', 0))
            st.metric("Failed", status.get('failed', 0))
            st.metric("Execution Time", f"{status.get('duration', 0):.1f}s")
        else:
            st.info("No execution status available")
        
        # Real-time execution log
        st.subheader("📝 Execution Log")
        if 'execution_log' in st.session_state:
            log_container = st.container()
            with log_container:
                for log_entry in st.session_state.execution_log[-10:]:
                    st.text(log_entry)
        else:
            st.info("No execution logs available")

def save_test_execution_results(execution_results, test_files):
    """Save test execution results to file"""
    try:
        # Create results directory
        results_dir = Path("src/data/test_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_results_{timestamp}.json"
        
        # Prepare results data
        results_data = {
            "execution_id": f"exec_{timestamp}",
            "timestamp": datetime.datetime.now().isoformat(),
            "test_files": test_files,
            "summary": execution_results,
            # The per-test rows the Results page tabulates. This used to save the
            # log strings under this key, so a reloaded run showed log lines in
            # the results table.
            "detailed_results": execution_results.get('detailed_results', []),
            "execution_log": st.session_state.get('execution_log', []),
            "browser_results": {},
            "performance_metrics": {
                "total_duration": execution_results.get('duration', 0),
                "avg_test_time": execution_results.get('duration', 0) / max(execution_results.get('executed', 1), 1)
            }
        }
        
        # Save file
        file_path = results_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        st.success(f"💾 Execution results saved to: {filename}")
        
        # Update session state with saved results
        if 'saved_results' not in st.session_state:
            st.session_state.saved_results = []
        
        st.session_state.saved_results.append({
            'filename': filename,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'total_tests': execution_results.get('executed', 0),
            'passed': execution_results.get('passed', 0),
            'failed': execution_results.get('failed', 0)
        })
        
        return str(file_path)
        
    except Exception as e:
        st.warning(f"Could not save execution results: {e}")
        return None
    
# Streamlit's browser labels -> the engines pytest-playwright actually accepts.
BROWSER_ENGINES = {"Chromium": "chromium", "Firefox": "firefox", "WebKit": "webkit"}


def execute_playwright_tests(selected_files, browsers, headless, base_url, capture_options):
    """Convert the selected test cases to a Playwright suite and run it for real.

    Three stages, each able to fail visibly: normalise the saved cases, convert
    them to pytest modules (PlaywrightConverter), run those modules in a pytest
    subprocess (TestExecutorAgent). What lands in session state is what the
    browser did -- this function used to `time.sleep(0.3)` and
    `random.choice([True]*9 + [False])`, so every chart on the Results page was
    fabricated.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📖 Loading test cases...")
        raw_cases = []
        for file_name in selected_files:
            raw_cases.extend(load_test_cases_from_file(file_name))

        # Saved files predate the canonical schema, so normalise on the way in.
        all_test_cases = normalise_cases(raw_cases)
        if not all_test_cases:
            logger.error("execution aborted: no test cases in %s", selected_files)
            st.error("No test cases found in selected files")
            return

        engines = [BROWSER_ENGINES[b] for b in browsers if b in BROWSER_ENGINES] or ["chromium"]

        status_text.text("🧪 Converting test cases to Playwright...")
        progress_bar.progress(10)
        converter = PlaywrightConverter()
        suite_name = Path(selected_files[0]).stem if selected_files else "suite"
        suite = converter.generate_complete_test_suite(all_test_cases, suite_name, base_url)

        if not suite.get("success") or not suite.get("test_files"):
            logger.error("conversion produced no runnable suite: %s", suite.get("error"))
            st.error(f"❌ Could not convert test cases to Playwright: {suite.get('error', 'no files written')}")
            return

        case_index = suite.get("case_index", {})
        st.caption(f"🧾 Wrote {len(suite['test_files'])} suite file(s) to "
                   f"`{converter.generated_tests_dir}`")

        capture = tuple(option.lower() for option in capture_options)
        executor = TestExecutorAgent()

        executed = passed = failed = skipped = 0
        duration = 0.0
        execution_log = []
        detailed_results = []

        for position, engine in enumerate(engines, 1):
            status_text.text(f"🎬 Running {len(all_test_cases)} test cases on {engine}...")
            run = executor.execute_tests(suite["test_files"], browser=engine,
                                         headless=headless, base_url=base_url,
                                         capture=capture)
            duration += run.get("duration", 0.0)

            if run.get("status") != "completed":
                message = run.get("error", "unknown error")
                execution_log.append(f"⛔️ {engine}: {message}")
                st.error(f"❌ {engine}: {message}")

            for case in run.get("test_results", []):
                origin = case_index.get(case["name"], {})
                test_id = origin.get("id") or case["name"]
                title = origin.get("title") or case["name"]
                symbol = {"passed": "✅", "failed": "❌"}.get(case["status"], "⏭️")

                executed += 1
                if case["status"] == "passed":
                    passed += 1
                elif case["status"] == "failed":
                    failed += 1
                else:
                    skipped += 1

                execution_log.append(
                    f"{symbol} {test_id} - {engine}: {case['status'].upper()} "
                    f"({case['duration']:.1f}s)" + (f" - {case['error']}" if case['error'] else "")
                )
                detailed_results.append({
                    "Test Case": test_id,
                    "Test Title": title,
                    "Status": case["status"].title(),
                    "Browser": engine.title(),
                    "Priority": (origin.get("priority") or "medium").title(),
                    "Duration": f"{case['duration']:.1f}s",
                    "Error": case["error"],
                })

            progress_bar.progress(min(10 + int(position / len(engines) * 90), 100))

        status_text.text("✅ Test execution completed!")
        progress_bar.progress(100)

        st.session_state.execution_status = {
            'executed': executed,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'duration': duration,
            'total_test_cases': len(all_test_cases),
            'browsers_tested': len(engines),
            'simulated': False,
        }
        st.session_state.execution_log = execution_log
        st.session_state.detailed_results = detailed_results

        save_test_execution_results({
            'executed': executed,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'duration': duration,
            'simulated': False,
            'base_url': base_url,
            'suite_files': suite["test_files"],
            'detailed_results': detailed_results,
        }, selected_files)

        logger.info("execution finished: %d passed, %d failed, %d skipped of %d runs",
                    passed, failed, skipped, executed)
        if failed:
            st.warning(f"Test execution completed: {passed} passed, {failed} failed, {skipped} skipped")
        elif executed:
            st.success(f"All {passed} executed tests passed! 🎉")
        if skipped:
            st.info(f"⏭️ {skipped} test(s) were skipped — a step or assertion could not be "
                    "converted to Playwright. The skip message names the step.")

    except Exception as e:
        logger.error("test execution failed: %s", e, exc_info=True)
        st.error(f"Error executing tests: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def load_test_cases_from_file(file_name):
    """Load test cases from a specific file"""
    try:
        # Handle session state tests
        if file_name == "Generated Tests (Session)":
            if hasattr(st.session_state, 'generated_tests'):
                return st.session_state.generated_tests.get('test_cases', [])
            return []
        
        # Load from file
        test_cases_dir = Path("src/data/test_cases")
        for file_path in test_cases_dir.glob("*.json"):
            if file_path.stem == file_name:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('test_cases', [])
        
        return []
        
    except Exception as e:
        st.error(f"Error loading test cases from {file_name}: {e}")
        return []

def latest_test_case_metadata():
    """Metadata for the suite most recently generated, session first then disk."""
    session = getattr(st.session_state, 'generated_tests', None)
    if isinstance(session, dict) and session.get('metadata'):
        return session['metadata']

    test_cases_dir = Path("src/data/test_cases")
    files = sorted(test_cases_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('metadata', {})
        except (OSError, ValueError):
            continue
    return None


def render_results_page():
    st.header("📊 Test Results & Reports")

    # Attribution for the generated suite. Files written before this change have no
    # attribution key and render as Unknown rather than silently blank.
    metadata = latest_test_case_metadata()
    if metadata is not None:
        render_attribution(metadata, prefix="🤖 Test cases generated by")

    # Load test results
    results = load_test_results()
    
    if not results:
        st.warning("No test results found. Please execute tests first.")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Go to Test Execution", type="primary"):
                st.session_state.page = "🚀 Test Execution"
                st.rerun()
        
        with col2:
            st.info("Execute tests first to see results")
        return
    
    if results.get('simulated', True):
        st.warning(
            "🎲 **These results are simulated.** They were recorded before test "
            "execution ran a real browser (pass/fail was `random.choice`). Re-run "
            "the suite from the Execution page to replace them with real results."
        )

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tests", results.get('total_tests', 0))
    with col2:
        passed = results.get('passed', 0)
        passed_delta = results.get('passed_delta', 0)
        st.metric("Passed", passed, delta=passed_delta)
    with col3:
        failed = results.get('failed', 0)
        failed_delta = results.get('failed_delta', 0)
        st.metric("Failed", failed, delta=failed_delta)
    with col4:
        success_rate = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%",
                  help="Skipped tests are excluded; a skip means a step could not "
                       "be converted to Playwright.")

    if results.get('skipped'):
        st.info(f"⏭️ {results['skipped']} test(s) skipped — see the Error column for "
                "the step or assertion that could not be converted.")
    
    # Charts
    st.subheader("📈 Test Results Trends")
    
    # Create sample trend data
    trend_data = create_trend_chart_data(results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Line chart for trends
        fig_line = px.line(
            trend_data, 
            x='date', 
            y=['passed', 'failed'], 
            title='Test Results Over Time',
            labels={'value': 'Number of Tests', 'date': 'Date'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        # Pie chart for current results
        fig_pie = px.pie(
            values=[passed, failed],
            names=['Passed', 'Failed'],
            title='Current Test Status',
            color_discrete_map={'Passed': '#00CC96', 'Failed': '#FF6B6B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Test Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Filter by Status:", ["All", "Passed", "Failed", "Skipped"])
    
    with col2:
        browser_filter = st.selectbox("Filter by Browser:", ["All", "Chromium", "Firefox", "Safari"])
    
    with col3:
        priority_filter = st.selectbox("Filter by Priority:", ["All", "Critical", "High", "Medium", "Low"])
    
    # Display filtered results
    filtered_results = apply_filters(results.get('detailed_results', []), status_filter, browser_filter, priority_filter)
    
    if filtered_results:
        df = pd.DataFrame(filtered_results)
        st.dataframe(df, use_container_width=True)
        
        # Download reports
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Generate PDF report (placeholder)
            st.button("📄 Generate PDF Report", help="PDF report generation coming soon!")
    else:
        st.info("No results match the selected filters.")

def render_ai_provider_settings():
    """The single place the provider, the models and the credentials are chosen."""
    st.subheader("🔌 AI Provider")

    endpoint = provider.resolve_endpoint()
    if endpoint.gateway == "omniroute":
        st.success(f"Mode: **OmniRoute gateway** — `{endpoint.base_url}`")
    else:
        st.info("Mode: **OpenAI direct** — set a gateway URL below to route through OmniRoute.")

    with st.expander("Gateway & credentials", expanded=True):
        base_url = st.text_input(
            "OmniRoute base URL:",
            value=get_gateway_base_url() or "",
            placeholder=provider.DEFAULT_GATEWAY_URL,
            help="Leave empty to talk to OpenAI directly. A local instance is keyless by default.",
            key="omniroute_base_url_setting",
        )
        gateway_key = st.text_input(
            "OmniRoute API key (optional):",
            value=get_gateway_api_key() or "",
            type="password",
            help="Only needed when the gateway runs with REQUIRE_API_KEY=true.",
            key="omniroute_key_setting",
        )
        openai_key = st.text_input(
            "OpenAI API Key (direct mode / fallback):",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            key="openai_key_setting",
        )

        if st.button("💾 Save provider settings", type="primary"):
            set_gateway_base_url(base_url.strip())
            set_gateway_api_key(gateway_key.strip())
            if openai_key:
                set_openai_api_key(openai_key)
            provider.reset_clients()
            st.success("✅ Provider settings saved for this session.")
            st.rerun()

    with st.expander("Models", expanded=True):
        chat_models, chat_warning = provider.list_chat_models()
        chat_options = [m["id"] for m in chat_models]
        if chat_warning:
            st.warning(f"⚠️ {chat_warning}")

        default_llm = st.selectbox(
            "Default LLM model:",
            chat_options,
            index=provider.option_index(chat_options, st.session_state.get('llm_model')),
            key="settings_llm_model",
        )

        # Embeddings do not come from the gateway -- it serves chat only. They are
        # produced in-process, so there is one model and it is set by $EMBEDDING_MODEL.
        default_embedding = embeddings_mod.default_model()
        selected_dims = embeddings_mod.model_dimensions(default_embedding)
        st.caption(
            f"🔡 Embeddings: `{default_embedding}` on **{embeddings_mod.resolve_device()}**"
            + (f" — {selected_dims} dims" if selected_dims else "")
            + " (local, in-process; set `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` to change)"
        )

        col_save, col_test = st.columns(2)
        with col_save:
            if st.button("💾 Save model defaults", use_container_width=True):
                set_llm_model(default_llm)
                st.success("✅ Model defaults saved for this session.")
                st.rerun()
        with col_test:
            if st.button("🔍 Test connection", use_container_width=True):
                test_llm_connection(default_llm)

    render_vector_store_settings()

    with st.expander("Make it permanent (.env)"):
        st.caption("The UI keeps choices for this session only, exactly like the API key. "
                   "Paste this into `src/.env` for a persistent default:")
        st.code(
            f"OMNIROUTE_BASE_URL={base_url.strip() or provider.DEFAULT_GATEWAY_URL}\n"
            f"OMNIROUTE_LLM_MODEL={default_llm}\n"
            f"OMNIROUTE_TIMEOUT={int(provider.timeout_seconds())}\n"
            f"EMBEDDING_MODEL={default_embedding}\n"
            f"EMBEDDING_DEVICE={embeddings_mod.resolve_device()}\n"
            "# OMNIROUTE_API_KEY=only-if-REQUIRE_API_KEY-is-true",
            language="bash",
        )


def render_vector_store_settings():
    """One store per source, listed. Vector width is a correctness constraint:
    mixing widths corrupts a store, so a stale-width one is called out per row."""
    root = Path("src/data/vector_store")
    stores = sorted(
        (d for d in root.glob("*") if d.is_dir() and embeddings_mod.read_store_meta(d)),
        key=lambda d: d.name,
    )
    if not stores:
        return

    configured = embeddings_mod.model_dimensions(embeddings_mod.default_model())
    with st.expander(f"🗂️ Vector stores ({len(stores)})", expanded=False):
        st.caption("One store per video, keyed by YouTube id or file name. "
                   "Re-ingesting the same source reuses its vectors unless the "
                   "transcript changed.")
        for store in stores:
            stored = embeddings_mod.read_store_meta(store) or {}
            st.markdown(f"**`{store.name}`** — {stored.get('source') or 'unknown source'}")
            st.caption(
                f"{provider.pretty(stored.get('provider'), stored.get('model'))}"
                f" — {stored.get('dimensions')} dims, {stored.get('written_at', 'unknown date')}"
            )
            if configured and stored.get('dimensions') and configured != stored['dimensions']:
                st.error(
                    f"⚠️ The selected embedding model produces {configured}-dim vectors but "
                    f"this store is {stored['dimensions']}-dim. Vectors of different widths "
                    "are not comparable — clear it before re-ingesting."
                )
            if st.button("🗑️ Clear", key=f"clear_store_{store.name}"):
                for item in store.glob("*"):
                    item.unlink()
                store.rmdir()
                st.success(f"✅ Cleared `{store.name}`. Re-ingest to rebuild it.")
                st.rerun()


def render_settings_page():
    st.header("⚙️ Settings")
    
    render_ai_provider_settings()
    
    # Playwright Configuration
    st.subheader("🎭 Playwright Configuration")
    
    with st.expander("Browser Settings"):
        default_browser = st.selectbox(
            "Default Browser:", 
            ["Chromium", "Firefox", "Safari"],
            index=["Chromium", "Firefox", "Safari"].index(st.session_state.default_browser),
            key="browser_setting"
        )
        
        default_viewport = st.selectbox(
            "Default Viewport:", 
            ["1920x1080", "1366x768", "1280x720", "Mobile"],
            key="viewport_setting"
        )
        
        default_timeout = st.slider(
            "Default Timeout (seconds):", 
            5, 60, 
            st.session_state.default_timeout,
            key="timeout_setting"
        )
    
    # Save all settings
    if st.button("💾 Save All Settings", type="primary"):
        st.session_state.default_browser = default_browser
        st.session_state.default_timeout = default_timeout
        st.success("✅ All settings saved successfully!")

# Helper Functions

def load_available_test_files():
    """Load available test case files from the test_cases directory"""
    test_files = []
    
    # Create directory if it doesn't exist
    test_cases_dir = Path("src/data/test_cases")
    test_cases_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON test files
    for file_path in test_cases_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            test_files.append({
                'name': file_path.stem,
                'path': str(file_path),
                'type': 'json',
                'test_count': len(test_data.get('test_cases', [])),
                'created': datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {str(e)}")
    
    # Load YAML test files
    for file_path in test_cases_dir.glob("*.yaml"):
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = yaml.safe_load(f)
            
            test_files.append({
                'name': file_path.stem,
                'path': str(file_path),
                'type': 'yaml',
                'test_count': len(test_data.get('test_cases', [])),
                'created': datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {str(e)}")
    
    # Also check if there are test cases in session state
    if hasattr(st.session_state, 'generated_tests') and st.session_state.generated_tests:
        test_files.append({
            'name': 'Generated Tests (Session)',
            'path': 'session_state',
            'type': 'session',
            'test_count': len(st.session_state.generated_tests.get('test_cases', [])),
            'created': datetime.datetime.now()
        })
    
    return test_files

def load_test_results():
    """Load test execution results from files and session state"""
    
    # First check session state
    if 'execution_status' in st.session_state:
        session_results = st.session_state.execution_status
        # USE REAL DETAILED RESULTS instead of generated samples
        detailed_results = st.session_state.get('detailed_results', [])
    else:
        session_results = None
        detailed_results = []
    
    # Load from files
    results_dir = Path("src/data/test_results")
    file_results = []
    
    if results_dir.exists():
        for file_path in results_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                file_results.append(result_data)
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")
    
    file_results.sort(key=lambda r: r.get('timestamp', ''))

    # Return most recent or session results
    if session_results:
        return {
            'total_tests': session_results.get('executed', 0),
            'passed': session_results.get('passed', 0),
            'failed': session_results.get('failed', 0),
            'skipped': session_results.get('skipped', 0),
            'passed_delta': 0,
            'failed_delta': 0,
            'simulated': session_results.get('simulated', True),
            'detailed_results': detailed_results,
            'historical_results': file_results
        }
    elif file_results:
        latest = file_results[-1]  # Most recent
        summary = latest.get('summary', {})
        return {
            'total_tests': summary.get('executed', 0),
            'passed': summary.get('passed', 0),
            'failed': summary.get('failed', 0),
            'skipped': summary.get('skipped', 0),
            'passed_delta': 0,
            'failed_delta': 0,
            # Runs written before execution was real carry no flag, and every one
            # of them was simulated. Absence means simulated, never "assume real".
            'simulated': summary.get('simulated', True),
            'detailed_results': latest.get('detailed_results', []),
            'historical_results': file_results
        }

    return None

def create_trend_chart_data(results):
    """Trend over the runs actually saved in src/data/test_results.

    This used to return five hardcoded numbers plus the current run, so the
    "Test Results Over Time" chart showed history that never happened.
    """
    rows = []
    for record in results.get('historical_results', []):
        summary = record.get('summary', {})
        rows.append({
            'date': (record.get('timestamp') or '')[:19].replace('T', ' '),
            'passed': summary.get('passed', 0),
            'failed': summary.get('failed', 0),
            'simulated': summary.get('simulated', True),
        })

    if not rows:
        rows = [{'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'passed': results.get('passed', 0),
                 'failed': results.get('failed', 0),
                 'simulated': results.get('simulated', True)}]

    return pd.DataFrame(rows)

def apply_filters(results, status_filter, browser_filter, priority_filter):
    """Apply filters to test results"""
    filtered = results
    
    if status_filter != "All":
        filtered = [r for r in filtered if r.get('Status') == status_filter]
    
    if browser_filter != "All":
        filtered = [r for r in filtered if r.get('Browser') == browser_filter]
    
    if priority_filter != "All":
        filtered = [r for r in filtered if r.get('Priority') == priority_filter]
    
    return filtered

def test_llm_connection(model=None):
    """Ping the configured provider and report which model actually answered."""
    endpoint = provider.resolve_endpoint()
    ok, attribution, error = provider.test_connection(model)

    if not ok:
        logger.error("provider ping failed via %s at %s: %s",
                     endpoint.gateway, endpoint.base_url, error)
        st.error(f"❌ Connection failed via {endpoint.gateway} at {endpoint.base_url}")
        st.caption(error or "no detail reported")
        if endpoint.gateway == "omniroute":
            st.info("Is the gateway running? `npm i -g omniroute && omniroute`")
        return False

    latency = f" ({attribution.latency_ms}ms)" if attribution.latency_ms else ""
    st.success(f"✅ Connected — answered by {attribution.display}{latency}")
    if attribution.note:
        st.caption(f"↳ {attribution.note}")
    if attribution.source == "unknown":
        st.caption("↳ the gateway did not report which model served this")
    return True

if __name__ == "__main__":
    main()