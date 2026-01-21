import streamlit as st
import streamlit.components.v1 as components

# =============================================================================
# GLOBAL PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="FraudHunter Pro 🔍",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix browser translation prompt
components.html(
    """
    <script>
        window.parent.document.documentElement.lang = 'es';
        window.parent.document.documentElement.setAttribute('translate', 'no');
        var meta = window.parent.document.createElement('meta');
        meta.name = 'google';
        meta.content = 'notranslate';
        if (!window.parent.document.querySelector('meta[name="google"]')) {
            window.parent.document.head.appendChild(meta);
        }
    </script>
    """,
    height=0,
    width=0
)

# =============================================================================
# UNIFIED NAVIGATION
# =============================================================================

# Define pages
legacy_page = st.Page(
    "views/legacy_v2.py", 
    title="Legacy v2.0", 
    icon="🏛️", 
    default=True
)
os_v3_page = st.Page(
    "views/os_v3.py", 
    title="FraudHunter OS v3.0", 
    icon="🕹️"
)

# Create navigation
pg = st.navigation([legacy_page, os_v3_page])

# Run navigation
pg.run()
