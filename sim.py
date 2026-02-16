# sim.py
import streamlit as st

from core import init_state

st.set_page_config(page_title="CRM Simulator", layout="wide")

init_state()

st.title("CRM Simulator")
st.caption("Use the sidebar to switch between Essentials and Playground.")
