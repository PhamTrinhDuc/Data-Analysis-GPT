import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer


def render():
    st.set_page_config(
        page_title="📈 Interactive Visualization Tool",
        page_icon="📈",
        layout="wide",
    )

    st.header("📈 Interactive Visualization Tool")
    st.write(" Welcome to interactive visualization tool. Please enjoy !")

    if st.session_state.get("df") is not None:
        pyg_app = StreamlitRenderer(
            st.session_state.df
        )
        pyg_app.explorer()
    else:
        st.info("Please upload a dataset to begin using the interactive visualization tools")