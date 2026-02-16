import streamlit as st

st.set_page_config(
    page_title="CRM simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optioneel: compacte layout
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# In multipage apps (pages/) is sim.py de "home".
# Jij wil eigenlijk alleen Essentials + Playground. Daarom: direct switchen naar Essentials.
if "did_autoswitch" not in st.session_state:
    st.session_state["did_autoswitch"] = True
    try:
        st.switch_page("pages/essentials.py")
    except Exception:
        # Als switch_page niet beschikbaar is in jouw Streamlit versie, laat dan een simpele home zien.
        st.title("CRM simulator")
        st.write("Open links in de sidebar: Essentials of Playground.")
