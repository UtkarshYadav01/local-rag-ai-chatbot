import os

import streamlit as st

from populate_database_docs_updated import run_pipeline, query_rag

st.title("Chat with LLaMA via Ollama")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# --- SIDEBAR SECTION ---
st.sidebar.header("📁 Upload a File")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    # Create the folder if it doesn't exist
    save_dir = "data from ui"
    os.makedirs(save_dir, exist_ok=True)

    # Define save path
    save_path = os.path.join(save_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.info("File saved for processing.")

    # Button to run pipeline
    if st.sidebar.button("🚀 Process File"):
        st.sidebar.info("Processing started...")
        run_pipeline()
        st.sidebar.success("✅ Processing complete!")

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
                reply = query_rag(user_prompt)
            except Exception as e:
                reply = f"❌ {str(e)}"
            st.markdown(reply)

    st.session_state.chat.append({"role": "assistant", "content": reply})
