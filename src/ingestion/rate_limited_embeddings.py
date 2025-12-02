import time
import logging
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)


class RateLimitedGoogleEmbeddings:
    """
    Wrapper around GoogleGenerativeAIEmbeddings with intelligent rate limiting.
    
    Features:
    - Automatic batching
    - Configurable delays between batches
    - Exponential backoff on rate limit errors
    - Progress tracking
    """
    
    def __init__(
        self,
        model: str = "models/text-embedding-004",
        google_api_key: str = None,
        batch_size: int = 10,
        delay_between_batches: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 60.0
    ):
        """
        Initialize rate-limited embeddings.
        
        Args:
            model: Google embedding model name
            google_api_key: Google API key
            batch_size: Number of documents to embed per batch (default: 10)
            delay_between_batches: Seconds to wait between batches (default: 5)
            max_retries: Maximum retry attempts on rate limit (default: 3)
            retry_delay: Initial retry delay in seconds (default: 60)
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=google_api_key
        )
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized RateLimitedGoogleEmbeddings:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Delay between batches: {delay_between_batches}s")
        logger.info(f"  - Max retries: {max_retries}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with automatic batching and rate limiting.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Starting embedding process for {len(texts)} documents")
        logger.info(f"Will process in {total_batches} batches")
        
        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_texts = texts[i:i + self.batch_size]
            
            logger.info(f"📌 Batch {batch_num}/{total_batches}: Processing {len(batch_texts)} documents...")
            
            # Try to embed with retries
            batch_embeddings = self._embed_with_retry(batch_texts, batch_num)
            all_embeddings.extend(batch_embeddings)
            
            # Delay between batches (except for last batch)
            if i + self.batch_size < len(texts):
                logger.info(f"   ⏳ Waiting {self.delay_between_batches}s before next batch...")
                time.sleep(self.delay_between_batches)
        
        logger.info(f"✅ Successfully embedded {len(all_embeddings)} documents")
        return all_embeddings
    
    def _embed_with_retry(self, texts: List[str], batch_num: int) -> List[List[float]]:
        """
        Embed a batch with exponential backoff retry logic.
        
        Args:
            texts: Batch of texts to embed
            batch_num: Current batch number (for logging)
            
        Returns:
            List of embedding vectors
        """
        retry_count = 0
        current_retry_delay = self.retry_delay
        
        while retry_count <= self.max_retries:
            try:
                embeddings = self.embeddings.embed_documents(texts)
                return embeddings
                
            except ResourceExhausted as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    logger.error(f"❌ Max retries ({self.max_retries}) exceeded for batch {batch_num}")
                    logger.error(f"Error: {str(e)}")
                    raise
                
                logger.warning(f"⚠️  Rate limit hit on batch {batch_num}")
                logger.warning(f"   Retry {retry_count}/{self.max_retries} after {current_retry_delay}s...")
                time.sleep(current_retry_delay)
                
                # Exponential backoff: double the delay for next retry
                current_retry_delay *= 2
                
            except Exception as e:
                logger.error(f"❌ Unexpected error in batch {batch_num}: {str(e)}")
                raise
        
        # Should never reach here, but just in case
        raise Exception(f"Failed to embed batch {batch_num} after {self.max_retries} retries")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (no rate limiting needed for single queries).
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)


def create_rate_limited_embeddings(
    google_api_key: str,
    model: str = "models/text-embedding-004",
    batch_size: int = 10,
    delay: float = 5.0
) -> RateLimitedGoogleEmbeddings:
    """
    Factory function to create rate-limited embeddings.
    
    Args:
        google_api_key: Google API key
        model: Embedding model name
        batch_size: Batch size (smaller = safer, larger = faster)
        delay: Delay between batches in seconds
        
    Returns:
        RateLimitedGoogleEmbeddings instance
    """
    return RateLimitedGoogleEmbeddings(
        model=model,
        google_api_key=google_api_key,
        batch_size=batch_size,
        delay_between_batches=delay
    )