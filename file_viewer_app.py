"""
file_viewer_app.py — minimal Streamlit file upload & viewer

Upload a text file (.txt, .csv, .md, ...) or an Excel file (.xlsx, .xls)
and its contents are displayed immediately.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="File Viewer", page_icon="📄")

st.title("File Viewer")
st.write("Upload a text file or an Excel file to see its contents.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["txt", "csv", "md", "log", "json", "xlsx", "xls"],
)

if uploaded_file is not None:
    st.subheader(uploaded_file.name)

    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        for sheet_name, df in sheets.items():
            st.markdown(f"**Sheet: {sheet_name}**")
            st.dataframe(df)
    else:
        text = uploaded_file.read().decode("utf-8", errors="replace")
        st.text(text)
