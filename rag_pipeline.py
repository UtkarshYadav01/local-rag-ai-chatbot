import os
import shutil
import logging
from datetime import datetime
from functools import lru_cache
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredFileLoader,
    JSONLoader, TextLoader, CSVLoader
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import CHROMA_PATH, DATA_PATH, LLM_MODEL, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    run_pipeline()


# 1. Load documents
def load_documents(folder_path: str = DATA_PATH):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()

            try:
                if ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif ext == '.docx':
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif ext == '.json':
                    loader = JSONLoader(file_path, jq_schema=".", text_content=True)
                elif ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif ext == '.csv':
                    loader = CSVLoader(file_path)
                else:
                    loader = UnstructuredFileLoader(file_path)

                docs.extend(loader.load())
            except Exception as e:
                logging.warning(f"❌ Failed to load {file_path}: {e}")

    logging.info(f"📃 Loaded {len(docs)} documents from 📂 {folder_path}")
    return docs


# 2. Split data
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"✂️ Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# 3. Generate Unique IDs
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 1)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
        chunk.metadata["filename"] = os.path.basename(source)
        chunk.metadata["processed_at"] = datetime.now().isoformat()

    logging.info(f"🆔 Assigned unique IDs to {len(chunks)} chunk(s)")
    return chunks


# 4. Embed data
@lru_cache(maxsize=1)
def get_embedding_function():
    logging.info(f"🚀 Initializing embedding model: {EMBED_MODEL}")
    return OllamaEmbeddings(model=EMBED_MODEL)


# 5. Cached DB instance
@lru_cache(maxsize=1)
def get_chroma_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


# 6. Store in DB
def add_to_chroma(chunks: list[Document]):
    db = get_chroma_db()
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        logging.info(f"✅ Added {len(new_chunks)} new chunks to Chroma DB")
    else:
        logging.info("ℹ️ No new chunks to add — everything already embedded.")


# 7. Reset DB (safe)
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info(f"🧹 Cleared Chroma DB at {CHROMA_PATH}")
    else:
        logging.info("No Chroma DB found to clear.")


# 8. Ask
def query_rag(query_text: str):
    # a. Prepare the DB.
    db = get_chroma_db()

    # b. template var
    PROMPT_TEMPLATE = """
    You are a helpful assistant. Answer the question using the context below.
    If the answer is not in the context, say you don't know — do not fabricate.

    Context:
    {context}

    Question:
    {question}
    """

    # c. Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # d. generate complete prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )
    # logging.info(prompt)

    # e.invoke llm
    model = OllamaLLM(model=LLM_MODEL)
    response_text = model.invoke(prompt)

    # f. get the original source
    sources = [doc.metadata.get("id") for doc, _ in results]
    formated_response = f"{response_text}\n\n**Sources:** {sources}"
    # logging.info(formated_response)

    return response_text


# 9. Run full pipeline
def run_pipeline():
    logging.info("\n================ RAG Pipeline ================\n")
    chunks = split_documents(load_documents())
    add_to_chroma(chunks)
    logging.info("\n=============================================\n")


if __name__ == "__main__":
    main()
