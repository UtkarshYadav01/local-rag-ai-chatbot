import os

import streamlit as st

from config import DATA_PATH
from llm_clients import ask_gpt
from rag_pipeline import run_pipeline

st.title("Chatbot Assistant via RAG LLM")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# --- SIDEBAR SECTION ---
st.sidebar.header("📁 Upload a File")

# delete button
# if st.sidebar.button("🧹 Reset Database"):
#     try:
#         clear_database()
#         st.sidebar.success("✅ Database cleared successfully!")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error: {str(e)}")

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    # Create the folder if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)

    # Define save path
    save_path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.info("File saved for processing.")

    # Button to run pipeline
    if st.sidebar.button("🚀 Process File"):
        try:
            st.sidebar.info("Processing started...")
            run_pipeline()
            st.sidebar.success("✅ Processing complete!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# --------------------------------

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_prompt := st.chat_input("Ask something..."):
    st.session_state.chat.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # You can modify query_rag to take uploaded_file as an optional argument
                # reply = query_rag(user_prompt)
                reply = ask_gpt(user_prompt)
            except Exception as e:
                reply = f"❌ {str(e)}"
            st.markdown(reply)

    st.session_state.chat.append({"role": "assistant", "content": reply})
