import os
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config.settings import settings
from src.ingestion.rate_limited_embeddings import create_rate_limited_embeddings

logger = logging.getLogger(__name__)


def build_faiss_index(
    documents: List[str], 
    index_path: str, 
    embedding_model: str = "models/text-embedding-004",
    batch_size: int = 10,
    delay_between_batches: float = 5.0
) -> FAISS:
    """
    Build a FAISS vector store from given documents and save it locally.

    Args:
        documents (List[str]): Preprocessed document texts to index.
        index_path (str): Directory path to save FAISS index.
        embedding_model (str): Google Gemini embedding model name.
                              Default: "models/text-embedding-004" (latest)
        batch_size (int): Number of chunks to embed per batch (default: 10)
        delay_between_batches (float): Seconds to wait between batches (default: 5.0)

    Returns:
        FAISS: The built FAISS vector store object.
    """

    # Convert raw strings to Document objects
    doc_objects = [Document(page_content=d) for d in documents]
    logger.info(f"📌 Total documents: {len(doc_objects)}")

    # Split documents into chunks
    logger.info("📌 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(doc_objects)
    logger.info(f"📌 Total chunks after splitting: {len(split_docs)}")

    # Create rate-limited embeddings
    logger.info("📌 Creating embeddings with Google Gemini...")
    logger.info(f"   Using rate limiting: batch_size={batch_size}, delay={delay_between_batches}s")
    
    embeddings = create_rate_limited_embeddings(
        google_api_key=settings.GOOGLE_API_KEY,
        model=embedding_model,
        batch_size=batch_size,
        delay=delay_between_batches
    )

    # Build FAISS index with rate-limited embeddings
    logger.info("📌 Building FAISS index with rate limiting...")
    logger.info("⏱️  This may take a few minutes due to API rate limits...")
    
    # Extract texts and metadata
    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    
    # Get embeddings with automatic batching and rate limiting
    logger.info(f"📌 Starting embedding generation for {len(texts)} chunks...")
    text_embeddings = embeddings.embed_documents(texts)
    
    logger.info(f"📌 Generated {len(text_embeddings)} embeddings successfully!")
    
    # Create FAISS index from embeddings
    logger.info("📌 Building FAISS vector store...")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, text_embeddings)),
        embedding=embeddings.embeddings,  # Use the underlying embeddings for future queries
        metadatas=metadatas
    )

    # Ensure directory exists and save
    os.makedirs(index_path, exist_ok=True)
    logger.info(f"📌 Saving FAISS index to: {index_path}")
    vectorstore.save_local(index_path)

    logger.info("🎉 FAISS index built and saved successfully!")
    return vectorstore


if __name__ == "__main__":
    # Standalone usage example
    from src.ingestion.preprocess import preprocess_dataframe
    from src.ingestion.load_data import load_complaint_data

    df = load_complaint_data()
    docs = preprocess_dataframe(df)
    
    build_faiss_index(
        documents=docs,
        index_path=settings.FAISS_INDEX_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        batch_size=10,  # Adjust based on your API limits
        delay_between_batches=5.0  # Adjust based on your API limits
    )