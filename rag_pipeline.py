import argparse
import os
import shutil
import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredFileLoader, \
    JSONLoader, TextLoader, CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHROMA_PATH = "chroma"
DATA_PATH = "data"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logging.info("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    # query_rag(query_text)


# helper method to start processing pipeline (loading → splitting → embedding)
def run_pipeline():
    logging.info("\n================ RAG Pipeline ================\n")
    chunks = split_documents(load_documents())
    add_to_chroma(chunks)
    logging.info("\n=============================================\n")


"""
def run_pipeline(file_path):
    # 1. Load document
    doc = load_documents(file_path)

    # 2. Preprocess / Clean / OCR
    clean_doc = preprocess_document(doc)

    # 3. Structure / Tag Sections
    structured_doc = structure_document(clean_doc)

    # 4. Split into chunks
    chunks = split_documents(structured_doc)

    # 5. Generate unique IDs + metadata
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 6. Embed chunks
    embeddings = get_embedding_function(chunks_with_ids)

    # 7. Store in DB
    add_to_chroma(embeddings)

    # 8. (Optional) Update Knowledge Base
    update_knowledge_base(chunks_with_ids)

    # Done
    return "Processing Complete"
"""

# 1.load data v1
"""def load_documents():
    document_loader = UnstructuredWordDocumentLoader(DATA_PATH)
    return document_loader.load()"""

# 1.load data v2
"""def load_documents(folder_path="data"):
    documents = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):  # only load .docx files
            file_path = os.path.join(folder_path, filename)
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            documents.extend(docs)  # add all documents from this file to the list

    return documents"""


# 1.load data v3
def load_documents(folder_path: str = DATA_PATH):
    """
    Load all documents (PDF, DOCX, JSON, TXT, CSV, etc.) from a folder into LangChain Document objects.
    """

    docs = []
    supported_exts = {'.pdf', '.docx', '.json', '.txt', '.csv'}
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
                    loader = JSONLoader(
                        file_path,
                        jq_schema=".",  # You can change this if JSON has a specific key
                        text_content=True
                    )
                elif ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif ext == '.csv':
                    loader = CSVLoader(file_path)
                else:
                    # fallback for other formats
                    loader = UnstructuredFileLoader(file_path)

                docs.extend(loader.load())

            except Exception as e:
                logging.info(f"❌ Failed to load {file_path}: {e}")

    logging.info(f"📃 Loaded {len(docs)} documents from 📂 {folder_path}")
    return docs


# 2. split data
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"✂️ Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# 3. Generating Unique IDs for Each Chunk
def calculate_chunk_ids(chunks):
    # This will create IDs like "data/sample.docx:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 1)
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
        chunk.metadata["filename"] = os.path.basename(source)
        chunk.metadata["processed_at"] = datetime.now().isoformat()

    logging.info(f"🆔 Assigned unique IDs to {len(chunks)} chunk(s)")
    return chunks


# 4. embed data
def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    logging.info(f"🚀 Embedding model initialized: {EMBED_MODEL}")
    return embeddings


# 5. store in db
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logging.info(f"📚 Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        logging.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logging.info("✅ No new documents to add")


# 6. reset db(optional) v1
"""def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)"""


# 6. reset db(optional) v2
def clear_database():
    paths_to_clear = [CHROMA_PATH, DATA_PATH]  # List of directories to clear

    for path in paths_to_clear:
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Deleted: {path}")
        else:
            logging.info(f"Path does not exist: {path}")
    logging.info("🧹 Database cleared")


# 1. ask v1
def query_rag(query_text: str):
    # 2. Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 3. template var
    PROMPT_TEMPLATE = """
    You are a helpful assistant. Answer the question using the context below.
    If the answer is not in the context, say you don't know — do not fabricate.

    Context:
    {context}

    Question:
    {question}
    """

    # 4. Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # 5. generate complete prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # logging.info(prompt)

    # 6.invoke llm
    model = OllamaLLM(model=LLM_MODEL)
    response_text = model.invoke(prompt)

    # 7. get the original source
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # logging.info(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
