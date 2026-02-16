import streamlit as st

st.set_page_config(
    page_title="Dose Escalation Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Programmatic navigation (so sidebar labels are exactly what you want)
pages = [
    st.Page("pages/essentials.py", title="Essentials", icon="⚙️"),
    st.Page("pages/playground.py", title="Playground", icon="🧪"),
]
nav = st.navigation(pages)
nav.run()
