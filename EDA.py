#imports
import os
import time
from apikey import apikey

import streamlit as st
import pandas as pd

from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

#api
os.environ['OPENAI_API_KEY'] = apikey

#intro
st.title("AI Data Analysis Assistant")

st.write("Built by Max Luo")

#sidebar
with st.sidebar:
    st.write("Begin with a CSV Upload")
    st.caption('''This program analysis the data set you upload using EDA and provides insightful 
               answers and reccomendations by the power of AI.''')
    #divider
    st.divider()
    st.caption('<p style="text-align:center"> made with 🥸 by Max</p>', unsafe_allow_html=True)

#buttons
if 'clicked' not in st.session_state:
    st.session_state.clicked={1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click= clicked, args=[1])
if st.session_state.clicked[1]: 
    #file uploadeder
    user_csv = st.file_uploader("Upload your files here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        llm = OpenAI(temperature=0)

        @st.cache_data
        def what_eda():
            what_eda = llm("What is EDA?")
            return what_eda

        @st.cache_data
        def steps_eda():
            steps_eda = llm("What are the steps of EDA? show only 5")
            return steps_eda

        #pands_agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        #functions
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset looks like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        def function_question_variable():
            st.line_chart(df, y =[user_question_var])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_var}, round to the nearest 2 decimal places")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_var}, round to the nearest 2 decimal places")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_var}, round to the nearest 2 decimal places")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_var}, round to the nearest 2 decimal places")
            st.write(trends)
            return

        def function_question_data():
            data_info = pandas_agent.run(user_question_data)
            st.write(data_info)
            return

        #end of functions

        #Main
        st.header('Exploratory Data Analysis')

        with st.sidebar:
            with st.expander("What is EDA"):
                st.caption(what_eda())
            with st.expander("What are the steps of EDA"):
                st.caption(steps_eda())

        function_agent()

        st.subheader('Variables')
        user_question_var = st.text_input("What variables are you interested in?")
        if user_question_var is not None and user_question_var != "":
            function_question_variable()
        
        st.subheader('Questions')
        user_question_data = st.text_input('What other questions do you have regarding this data set?')
        if user_question_data is not None and user_question_data != "":
            function_question_data()






