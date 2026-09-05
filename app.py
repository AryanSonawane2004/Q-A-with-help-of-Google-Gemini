import streamlit as st
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Provide detailed, well-structured, and comprehensive answers to the user's query."),
        ("user", "Question : {question}"),
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):
    genai.Client(api_key = api_key)     
    llm = ChatGoogleGenerativeAI(
        model = llm,
        temperature = temperature,
        max_tokens = max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question" : question})
    return answer       

st.title("Enhanced Q & A Chatbot with Google Gemini")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API Key", type = "password")

llm = st.sidebar.selectbox("Select your model", ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite","gemini-3.1-flash-lite"])

temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 100, max_value = 4096, value = 1500, step = 50)

st.write("Go ahead and ask any question.")

user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write("Bot: ", response) 
else:
    response = "Please ask a question."

