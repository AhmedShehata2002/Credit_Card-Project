import pandas as pd
import streamlit as st
import os
import pandasai
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Define a function to load data
def load_data():
    data = pd.read_csv('Oura.csv')  # Path to your CSV file
    return data

# Load and display the data
data = load_data()
st.write("# Health Data 🧘🏻‍♀️:")
st.dataframe(data)  # Displays the data in a table format

query = st.text_area("🗣️ Chat with Dataframe")

if query:
    llm = OpenAI(api_token=os.environ['OPENAI_API_KEY'])
    query_engine = SmartDataframe(data, config={"llm": llm})

    answer = query_engine.chat(query)
    st.write(answer)
