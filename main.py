import streamlit as st
from langchain_helper import get_few_shot_db_chain

# Set the title and page layout
st.set_page_config(page_title="LLM ClosetKeeper: Smart Inventory Management & SQL Expert for Fashion", layout="wide")

# Add a title and description
st.title("LLM ClosetKeeper: Smart Inventory Management & SQL Expert for Fashion")
st.subheader("Ask questions about your Stocks inventory database!")

# Create a text input for the question
question = st.text_input("Ask a question about your Stocks inventory:")

# Check if a question is provided
if question:
    # Get the response from the database chain
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    # Display the question and answer
    st.subheader("Your Question:")
    st.write(question)
    st.subheader("Answer:")
    st.write(response)
