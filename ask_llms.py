from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from openai import OpenAI

from config import API_KEY, AZURE_AI_ENDPOINT, LLM_MODEL
from rag_pipeline import get_chroma_db


# 1. Ask Local Ollama
def query_rag(query_text: str, chat_history: list = None):
    db = get_chroma_db()

    PROMPT_TEMPLATE = """
    You are a helpful assistant having a conversation with the user.
    Use both the chat history and the provided context to answer.
    If you are unsure, say "I don't know" — do not fabricate.

    Chat history:
    {history}

    Context:
    {context}

    User question:
    {question}
    """

    # a. Retrieve top relevant chunks
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # b. Convert chat history into readable text
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]]
        )

    # c. Fill in the prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
        history=history_text or "No previous conversation."
    )

    # d.invoke llm and Generate response
    model = OllamaLLM(model=LLM_MODEL)
    response_text = model.invoke(prompt)

    # e. get the original source
    sources = [doc.metadata.get("id") for doc, _ in results]
    formated_response = f"{response_text}\n\n**Sources:** {sources}"
    # logging.info(formated_response)

    return response_text


# 1. Ask Azure openAI
def ask_gpt(query_text: str,
            endpoint: str = AZURE_AI_ENDPOINT,
            api_key: str = API_KEY,
            chat_history: list = None,
            deployment_name: str = "gpt-5-mini") -> str:
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    db = get_chroma_db()

    PROMPT_TEMPLATE = """
        You are a helpful assistant having a conversation with the user.
        Use both the chat history and the provided context to answer.
        If you are unsure, say "I don't know" — do not fabricate.

        Chat history:
        {history}

        Context:
        {context}

        User question:
        {question}
        """

    # a. Retrieve top relevant chunks
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # b. Convert chat history into readable text
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]]
        )

        # c. Fill in the prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
        history=history_text or "No previous conversation."
    )

    response_text = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Return the message text directly
    result = response_text.choices[0].message.content

    # e. get the original source
    sources = [doc.metadata.get("id") for doc, _ in results]
    formated_response = f"{result}\n\n**Sources:** {sources}"
    # logging.info(formated_response)
    return result


# Example usage:
if __name__ == "__main__":
    response = ask_gpt("Hi")
    print(response)
