import streamlit as st

from query_data import query_rag

st.title("Chat with LLaMA via Ollama")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask something..."):
    st.session_state.chat.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = query_rag(user_prompt)# call
            except Exception as e:
                reply = f"❌ {str(e)}"
            st.markdown(reply)

    st.session_state.chat.append({"role": "assistant", "content": reply})
