import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# Safely set LangSmith environment variables if available
for key in ["LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]:
    val = os.getenv(key) or (st.secrets.get(key) if hasattr(st, "secrets") and key in st.secrets else None)
    if val:
        os.environ[key] = str(val)

# Also check Streamlit secrets for GOOGLE_API_KEY if present
if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Provide detailed, well-structured, and comprehensive answers to the user's query."),
        ("user", "Question : {question}"),
    ]
)


def generate_response(question, api_key, model_name, temperature, max_tokens):
    effective_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not effective_api_key:
        return "⚠️ Please enter your Google API Key in the sidebar (or configure GOOGLE_API_KEY in your Streamlit secrets)."

    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=effective_api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | chat_model | output_parser
    answer = chain.invoke({"question": question})
    return answer       

st.title("Enhanced Q & A Chatbot with Google Gemini")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API Key", type="password")

llm = st.sidebar.selectbox("Select your model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4096, value=1500, step=50)

st.write("Go ahead and ask any question.")

user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write("Bot: ", response) 
else:
    response = "Please ask a question."

