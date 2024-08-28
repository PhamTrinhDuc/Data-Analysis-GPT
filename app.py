import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from source.model import load_openai_model
from source.utils import excutor_code
from logs.logger import set_logging_terminal, set_logging_error
from source.tool_visualize import render

logger_error = set_logging_error()
logger_terminal = set_logging_terminal()


def process_query(agent: create_pandas_dataframe_agent, query: str):
    response  = agent(query)

    action = response['intermediate_steps'][-1][0].tool_input["query"]

    if "plt" in action:
        st.write(response['output'])
        fig = excutor_code(action, df = st.session_state.df)
        if fig:
            st.pyplot(fig)
        
        st.write("**Executed code:**")
        st.code(action)

        to_display_string = response['output'] + "\n" + f"```python\{action}\n```"
        st.session_state.history.append((query, to_display_string))
    
    else:
        st.write(response['output'])
        st.seasion_state.history((query, response['output']))

def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")

def main():

    render()
    # st.set_page_config(page_title="📊 Smart Data Analysis Tool", page_icon="📊", layout="centered")
    # st.header("📊 Smart Data Analysis Tool")
    # st.write(
    #     "### Welcome to our data analysis tool. This tools can assist your daily data analysis tasks. Please enjoy !"
    # )

    llm = load_openai_model()
    logger_terminal.info("Model loaded successfully")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    # Initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write(st.session_state.df)
        st.write("### Data loaded successfully", st.session_state.df.head())

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
        )

        logger_terminal.info("Agent created successfully")

        if st.button("Run query"):
                query = st.text_area("Enter your query here")
                process_query(agent, query)

        if st.button("Show history"):
            display_chat_history()

if __name__ == "__main__":
    main()