import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ingestion.load_data import load_complaint_data
from src.ingestion.preprocess import preprocess_dataframe
from src.ingestion.build_vector_store import build_faiss_index
from src.config.settings import settings

# Configure logging with UTF-8 encoding for Windows
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
# Force UTF-8 for console output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
logger = logging.getLogger(__name__)


def validate_environment():
    """
    Validate that all required environment variables and files exist.
    """
    logger.info("Validating environment...")
    
    # Check Google API key
    if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your-google-api-key-here":
        logger.error("GOOGLE_API_KEY not set in .env file!")
        logger.error("Please create a .env file with: GOOGLE_API_KEY=your-key-here")
        logger.error("Get your API key from: https://aistudio.google.com/app/apikey")
        return False
    
    # Check data file exists
    data_path = Path(settings.DATA_PATH)
    if not data_path.exists():
        logger.error(f"Data file not found: {settings.DATA_PATH}")
        logger.error("Please ensure complaints.csv exists in the data/ directory")
        return False
    
    # Check/create embeddings directory
    embeddings_dir = Path(settings.INDEX_PATH).parent
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Embeddings directory: {embeddings_dir}")
    
    logger.info("[OK] Environment validation passed")
    return True


def print_settings():
    """
    Print current configuration settings.
    """
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"LLM Provider: Google Gemini")
    logger.info(f"Data Path: {settings.DATA_PATH}")
    logger.info(f"Index Path: {settings.INDEX_PATH}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Chunk Size: {settings.CHUNK_SIZE}")
    logger.info(f"Chunk Overlap: {settings.CHUNK_OVERLAP}")
    logger.info(f"Embedding Batch Size: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"Embedding Batch Delay: {settings.EMBEDDING_BATCH_DELAY}s")
    logger.info(f"LLM Model: {settings.MODEL_NAME}")
    logger.info(f"Temperature: {settings.TEMPERATURE}")
    logger.info("="*60)


def train_pipeline(force_rebuild: bool = False, sample_size: int = None):
    """
    Main training pipeline that orchestrates the entire process.
    
    Args:
        force_rebuild: If True, rebuild even if index exists
        sample_size: If provided, only process this many rows (for testing)
    """
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("STARTING TRAINING PIPELINE (Google Gemini)")
    logger.info(f"Start Time: {start_time.isoformat()}")
    logger.info("="*60)
    
    try:
        # Step 0: Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            return False
        
        print_settings()
        
        # Check if index already exists
        index_path = Path(settings.INDEX_PATH)
        if index_path.exists() and not force_rebuild:
            logger.warning(f"Vector store already exists at: {index_path}")
            logger.warning("Use --force to rebuild")
            response = input("Continue and overwrite existing index? (y/N): ")
            if response.lower() != 'y':
                logger.info("Training cancelled by user")
                return False
        
        # Step 1: Load data
        logger.info("\n" + "="*60)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*60)
        df = load_complaint_data()
        
        if df is None or df.empty:
            logger.error("Failed to load data or data is empty")
            return False
        
        logger.info(f"[OK] Loaded {len(df)} complaints")
        
        # Apply sample size if specified (for testing)
        if sample_size and sample_size < len(df):
            logger.info(f"Using sample of {sample_size} complaints for testing")
            df = df.head(sample_size)
        
        # Step 2: Preprocess data
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("="*60)
        documents = preprocess_dataframe(df)
        
        if not documents:
            logger.error("No documents generated from preprocessing")
            return False
        
        logger.info(f"[OK] Generated {len(documents)} document chunks")
        
        # Step 3: Build vector store with Gemini embeddings
        logger.info("\n" + "="*60)
        logger.info("STEP 3: BUILDING VECTOR STORE (Google Gemini Embeddings)")
        logger.info("="*60)
        vector_store = build_faiss_index(
            documents=documents,
            index_path=settings.FAISS_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL
        )
        
        if vector_store is None:
            logger.error("Failed to build vector store")
            return False
        
        logger.info(f"[OK] Vector store saved to: {settings.INDEX_PATH}")
        
        # Training complete
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Start Time: {start_time.isoformat()}")
        logger.info(f"End Time: {end_time.isoformat()}")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total Documents: {len(documents)}")
        logger.info(f"Index Location: {settings.INDEX_PATH}")
        logger.info(f"LLM Provider: Google Gemini")
        logger.info("="*60)
        
        # Print next steps
        logger.info("\nNEXT STEPS:")
        logger.info("1. Test the bot: python -m src.bot.assistant")
        logger.info("2. Start API: python -m src.api.app")
        logger.info("3. View API docs: http://localhost:8000/docs")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}", exc_info=True)
        return False


def verify_index():
    """
    Verify that the index was created successfully and can be loaded.
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFYING INDEX (Google Gemini Embeddings)")
    logger.info("="*60)
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        index_path = Path(settings.INDEX_PATH)
        if not index_path.exists():
            logger.error(f"Index not found at: {index_path}")
            return False
        
        # Try to load the index with Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get index stats
        index_size = vector_store.index.ntotal
        logger.info(f"[OK] Index loaded successfully")
        logger.info(f"[OK] Number of vectors: {index_size}")
        
        # Test a simple query
        test_query = "refund issue"
        results = vector_store.similarity_search(test_query, k=3)
        logger.info(f"[OK] Test query successful, found {len(results)} results")
        
        if results:
            logger.info("\nSample result:")
            logger.info(f"Content: {results[0].page_content[:200]}...")
            logger.info(f"Metadata: {results[0].metadata}")
        
        logger.info("\n[OK] Index verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Index verification failed: {str(e)}", exc_info=True)
        return False


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Train the AI Complaint Bot by building FAISS vector store from complaints data (Google Gemini)"
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Use only a sample of N complaints (for testing)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing index without rebuilding'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify only mode
    if args.verify:
        success = verify_index()
        sys.exit(0 if success else 1)
    
    # Run training pipeline
    success = train_pipeline(
        force_rebuild=args.force,
        sample_size=args.sample
    )
    
    # Verify after training
    if success:
        logger.info("\n")
        verify_index()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()