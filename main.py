import streamlit as st
from langchain_helper import check_and_create_vector_db, ask_question_from_qa_chain

# Use Streamlit session state to ensure check_and_create_vector_db runs only once per session
if 'vector_db_initialized' not in st.session_state:
    check_and_create_vector_db()
    st.session_state.vector_db_initialized = True

st.title("Chat with Suraj Upadhyay - Resume Assistant")

st.markdown(
    "Hello! You are now chatting directly with Suraj Upadhyay. "
    "Feel free to ask me any questions about my professional experience, skills, and background. "
    "Just type your question below and I'll answer based on my resume information."
)

question = st.text_input('Enter your question here:', placeholder='e.g., What are Suraj\'s key skills?')

if question:
    try:
        with st.spinner('Getting answer...'):
            answer = ask_question_from_qa_chain(question=question)
        st.header('Answer')
        st.write(answer)
    except Exception as e:
        st.error(f"Error getting answer: {e}")
