import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config.settings import settings


def load_retriever(k: int = 3):
    """
    Load the FAISS vector store and return a retriever.
    Note: For querying, we don't need rate limiting as it's single queries.
    
    :param k: number of results to fetch
    :return: LangChain retriever
    """
    # Initialize Google Gemini embeddings (no rate limiting needed for queries)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Latest embedding model
        google_api_key=settings.GOOGLE_API_KEY
    )

    # Check if FAISS index exists
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {settings.FAISS_INDEX_PATH}. "
            f"Run train.py to generate embeddings first."
        )

    # Load FAISS vector store
    vectorstore = FAISS.load_local(
        folder_path=settings.FAISS_INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # required for FAISS
    )

    # Return a retriever with the specified number of results
    return vectorstore.as_retriever(search_kwargs={"k": k})