import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
DATA_PATH = os.getenv("DATA_PATH", "data")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
API_KEY = os.getenv("API_KEY", "")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "")