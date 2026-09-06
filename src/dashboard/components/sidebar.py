"""
This comprehensive sidebar component provides:

Main Navigation - Clean menu with icons and proper styling
System Status - Real-time status indicators for key components
Quick Stats - Essential metrics display with delta values
Settings Panel - API key configuration and data management
Progress Indicators - For multi-step processes
Notifications System - Centralized notification management
File Browser - Simple file exploration capability
Responsive Design - Works well on different screen sizes
"""

import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
from src.utils import provider
from typing import Dict

# Global constants
SIDEBAR_TITLE = "� TestRAGic"
SIDEBAR_SUBTITLE = "QA automation platform"
LOGO_URL = "src/assets/images/agent-qa-banner-image.png"

# Navigation menu items
MENU_ITEMS = {
    "generation": {"icon": "🎯", "label": "Test Generation"},
    "execution": {"icon": "🚀", "label": "Test Execution"}, 
    "results": {"icon": "📊", "label": "Results & Reports"},
    "settings": {"icon": "⚙️", "label": "Settings"}
}

# Status indicators
STATUS_COLORS = {
    "success": "#28a745",
    "warning": "#ffc107", 
    "error": "#dc3545",
    "info": "#17a2b8",
    "inactive": "#6c757d"
}

def render_sidebar() -> str:
    """
    Render the enhanced sidebar navigation with modern styling
    Returns the selected menu item key
    """
    with st.sidebar:
        # Custom CSS for prettier sidebar
        st.markdown("""
        <style>
        .sidebar-logo {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 1rem;
            color: white;
            text-align: center;
        }
        
        .sidebar-title {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0.5rem 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .sidebar-subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            margin: 0;
        }
        
        .status-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 0.8rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            color: white;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .status-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .status-error {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        
        .stat-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            color: #444;
            margin: 1.5rem 0 0.5rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .notification-item {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 3px solid #17a2b8;
            font-size: 0.85rem;
        }
        
        .file-item {
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: 6px;
            margin: 0.2rem 0;
            font-size: 0.8rem;
            color: #666;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced Logo and title section
        render_enhanced_logo()
        
        # Enhanced API Key status
        render_enhanced_api_status()
        
        # Main navigation menu with better styling
        selected = option_menu(
            menu_title=None,
            options=[MENU_ITEMS[key]["label"] for key in MENU_ITEMS.keys()],
            icons=["target", "rocket-takeoff", "graph-up", "gear"],  # Better icons
            menu_icon="grid-3x3-gap",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {
                    "padding": "0!important", 
                    "background-color": "transparent",
                    "border-radius": "10px"
                },
                "icon": {
                    "color": "#667eea", 
                    "font-size": "20px"
                }, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "2px 0",
                    "padding": "12px 16px",
                    "border-radius": "10px",
                    "background-color": "transparent",
                    "--hover-color": "#f0f0f0",
                    "transition": "all 0.3s ease"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "color": "white",
                    "font-weight": "600",
                    "box-shadow": "0 4px 8px rgba(102, 126, 234, 0.3)"
                },
            }
        )
        
        # Enhanced Quick Stats
        # render_enhanced_quick_stats()
        
        # Enhanced System Status
        # render_enhanced_system_status()
        
        # Enhanced Notifications
        # render_enhanced_notifications()
        
        # Enhanced File Browser
        render_enhanced_file_browser()
        
        # Convert selected label back to key
        selected_key = None
        for key, item in MENU_ITEMS.items():
            if item["label"] == selected:
                selected_key = key
                break
        
        return selected_key or "generation"

def render_enhanced_logo():
    st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</div>
            <div class="sidebar-title">TestRAGic</div>
            <div class="sidebar-subtitle">QA automation platform</div>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_api_status():
    """Render which AI provider is configured, and the model it will use"""
    if provider.is_configured():
        endpoint = provider.resolve_endpoint()
        label = "OmniRoute" if endpoint.gateway == "omniroute" else "OpenAI"
        st.markdown(f"""
        <div class="status-card status-success">
            <div style="font-size: 1.2rem;">✅</div>
            <div style="font-weight: 600; margin-top: 0.3rem;">AI Provider Ready</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">{label} · {provider.default_chat_model()}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-error">
            <div style="font-size: 1.2rem;">❌</div>
            <div style="font-weight: 600; margin-top: 0.3rem;">No AI Provider</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">Configure in Settings</div>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_quick_stats():
    """Render enhanced quick statistics with modern cards"""
    st.markdown('<div class="section-header">📈 Quick Stats</div>', unsafe_allow_html=True)
    
    stats = _get_quick_stats()
    
    # Row 1: Tests Generated & Executed
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('tests_generated', 0)}</div>
            <div class="stat-label">Tests Generated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('tests_executed', 0)}</div>
            <div class="stat-label">Tests Executed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Success Rate & Duration
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('success_rate', 0)}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('avg_duration', 0)}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_system_status():
    """Render enhanced system status with modern indicators"""
    st.markdown('<div class="section-header">🔍 System Status</div>', unsafe_allow_html=True)
    
    status_items = _get_system_status()
    
    for item, status in status_items.items():
        level = status["level"]
        
        if level == "success":
            icon = "🟢"
            color = "#28a745"
        elif level == "warning":
            icon = "🟡"
            color = "#ffc107"
        elif level == "error":
            icon = "🔴"
            color = "#dc3545"
        else:
            icon = "⚪"
            color = "#6c757d"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 0.5rem; margin: 0.2rem 0; 
                    background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <span style="font-size: 1rem; margin-right: 0.8rem;">{icon}</span>
            <span style="font-size: 0.9rem; color: #444; flex: 1;">{item}</span>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_notifications():
    """Render enhanced notifications with modern styling"""
    st.markdown('<div class="section-header">🔔 Notifications</div>', unsafe_allow_html=True)
    
    notifications = _get_notifications()
    
    if not notifications:
        st.markdown("""
        <div class="notification-item" style="text-align: center; color: #666;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔕</div>
            <div>No new notifications</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for notification in notifications[-3:]:  # Show last 3 notifications
        level = notification.get("level", "info")
        message = notification.get("message", "")
        timestamp = notification.get("timestamp", "")
        
        if level == "success":
            icon = "✅"
            border_color = "#28a745"
        elif level == "warning":
            icon = "⚠️"
            border_color = "#ffc107"
        elif level == "error":
            icon = "❌"
            border_color = "#dc3545"
        else:
            icon = "ℹ️"
            border_color = "#17a2b8"
        
        st.markdown(f"""
        <div class="notification-item" style="border-left-color: {border_color};">
            <div style="display: flex; align-items: flex-start;">
                <span style="margin-right: 0.5rem;">{icon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #333;">{message}</div>
                    <div style="font-size: 0.7rem; color: #999; margin-top: 0.2rem;">{timestamp}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_file_browser():
    """Render enhanced file browser with modern styling"""
    st.markdown('<div class="section-header">📁 Recent Files</div>', unsafe_allow_html=True)
    
    try:
        # Check test cases directory
        test_cases_dir = Path("src/data/test_cases")
        if test_cases_dir.exists():
            files = list(test_cases_dir.glob("*.json"))
            if files:
                # Show latest 3 files
                for file_path in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    file_size = file_path.stat().st_size / 1024  # KB
                    st.markdown(f"""
                    <div class="file-item">
                        <div style="display: flex; justify-content: space-between;">
                            <span>📄 {file_path.name[:20]}...</span>
                            <span>{file_size:.1f}KB</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="file-item" style="text-align: center; color: #999;">
                    No test files found
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="file-item" style="text-align: center; color: #999;">
                Test cases directory not found
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown("""
        <div class="file-item" style="text-align: center; color: #999;">
            Error loading files
        </div>
        """, unsafe_allow_html=True)

def _get_quick_stats() -> Dict:
    """Get quick statistics for display"""
    # These would typically come from session state or database
    # For now, return default/example values
    
    return {
        "tests_generated": st.session_state.get("total_tests_generated", 0),
        "tests_delta": st.session_state.get("tests_delta", 0),
        "tests_executed": st.session_state.get("total_tests_executed", 0),
        "execution_delta": st.session_state.get("execution_delta", 0),
        "success_rate": st.session_state.get("success_rate", 0),
        "success_delta": st.session_state.get("success_delta", 0),
        "avg_duration": st.session_state.get("avg_duration", 0),
        "duration_delta": st.session_state.get("duration_delta", 0)
    }

def _get_system_status() -> Dict:
    """Get system status for display"""
    status_items = {}
    
    # Check AI provider
    if provider.is_configured():
        endpoint = provider.resolve_endpoint()
        label = "OmniRoute" if endpoint.gateway == "omniroute" else "OpenAI"
        status_items["AI Provider"] = {"level": "success", "message": label}
    else:
        status_items["AI Provider"] = {"level": "error", "message": "Not configured"}
    
    # Check data directories
    try:
        data_dir = Path("src/data")
        if data_dir.exists():
            status_items["Data Directory"] = {"level": "success", "message": "Available"}
        else:
            status_items["Data Directory"] = {"level": "warning", "message": "Not found"}
    except Exception:
        status_items["Data Directory"] = {"level": "error", "message": "Error"}
    
    # Check test cases directory
    try:
        test_cases_dir = Path("src/data/test_cases")
        if test_cases_dir.exists():
            file_count = len(list(test_cases_dir.glob("*.json")))
            if file_count > 0:
                status_items["Test Cases"] = {"level": "success", "message": f"{file_count} files"}
            else:
                status_items["Test Cases"] = {"level": "warning", "message": "No files"}
        else:
            status_items["Test Cases"] = {"level": "warning", "message": "Directory missing"}
    except Exception:
        status_items["Test Cases"] = {"level": "error", "message": "Error"}
    
    # Check vector store
    try:
        vector_store_dir = Path("src/data/vector_store")
        if vector_store_dir.exists():
            status_items["Vector Store"] = {"level": "success", "message": "Available"}
        else:
            status_items["Vector Store"] = {"level": "warning", "message": "Not created"}
    except Exception:
        status_items["Vector Store"] = {"level": "error", "message": "Error"}
    
    # Check session state
    if hasattr(st.session_state, 'generated_tests'):
        status_items["Session Data"] = {"level": "success", "message": "Active"}
    else:
        status_items["Session Data"] = {"level": "inactive", "message": "Empty"}
    
    return status_items

def _get_notifications() -> list:
    """Get notifications for display"""
    # Initialize notifications in session state if not exists
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    # Add some sample notifications if empty
    if not st.session_state.notifications:
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M")
        
        sample_notifications = [
            {
                "level": "info",
                "message": "Welcome to TestRAGic!",
                "timestamp": current_time
            }
        ]
        
        # Add provider notification if missing
        if not provider.is_configured():
            sample_notifications.append({
                "level": "warning",
                "message": "Configure an AI provider (OmniRoute gateway or OpenAI key)",
                "timestamp": current_time
            })
        
        # Add success notification if tests exist
        if hasattr(st.session_state, 'generated_tests'):
            sample_notifications.append({
                "level": "success", 
                "message": "Test cases ready for execution",
                "timestamp": current_time
            })
        
        st.session_state.notifications = sample_notifications
    
    return st.session_state.notifications

def add_notification(level: str, message: str):
    """Add a notification to the system"""
    import datetime
    
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notification = {
        "level": level,
        "message": message,
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    }
    
    st.session_state.notifications.append(notification)
    
    # Keep only last 10 notifications
    if len(st.session_state.notifications) > 10:
        st.session_state.notifications = st.session_state.notifications[-10:]

def clear_notifications():
    """Clear all notifications"""
    if 'notifications' in st.session_state:
        st.session_state.notifications = []

def update_stats(stat_name: str, value: any):
    """Update a statistic in session state"""
    if stat_name == "tests_generated":
        st.session_state.total_tests_generated = value
    elif stat_name == "tests_executed":
        st.session_state.total_tests_executed = value
    elif stat_name == "success_rate":
        st.session_state.success_rate = value
    elif stat_name == "avg_duration":
        st.session_state.avg_duration = value

def render_system_status():
    """Render system status section (alias for compatibility)"""
    render_enhanced_system_status()

def render_quick_stats():
    """Render quick stats section (alias for compatibility)"""
    render_enhanced_quick_stats()

def render_settings_section():
    """Render settings section"""
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # AI provider status
    if provider.is_configured():
        st.markdown("""
        <div class="status-card status-success">
            <div>✅ AI Provider Configured</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-error">
            <div>❌ Configure API Key in Settings</div>
        </div>
        """, unsafe_allow_html=True)

def render_progress_indicator(progress: float, message: str = ""):
    """Render a progress indicator"""
    st.markdown('<div class="section-header">🚀 Progress</div>', unsafe_allow_html=True)
    
    st.progress(progress)
    if message:
        st.markdown(f"""
        <div class="notification-item">
            <div style="text-align: center; color: #666;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_notifications():
    """Render notifications section (alias for compatibility)"""
    render_enhanced_notifications()

def render_file_browser():
    """Render file browser section (alias for compatibility)"""
    render_enhanced_file_browser()
