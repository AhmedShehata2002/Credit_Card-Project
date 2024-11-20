import pandas as pd
import streamlit as st
import os
import pandasai
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser



class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


# Define a function to load data
def load_data():
    data = pd.read_csv('physio_cycles.csv')  # Path to your CSV file
    return data

# Load and display the data
data = load_data()
st.write("# Whoop Data 🧘🏻‍♀️:")
st.dataframe(data)  # Displays the data in a table format

query = st.text_area("🗣️ Chat with Dataframe")

if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(
        data,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
        },
    )

    answer = query_engine.chat(query)
    st.write(answer)
