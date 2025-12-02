from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings


def get_llm(model_name: str = "gemini-2.0-flash"):
    """
    Returns an initialized ChatGoogleGenerativeAI LLM client.

    Default model: gemini-2.0-flash-live (fast + cost-effective for retrieval)
    Other options: 
    - gemini-2.0-flash-live (more capable, slower)
    - gemini-1.5-flash-8b (fastest, most economical)
    """

    if not settings.GOOGLE_API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY missing in environment variables")

    llm = ChatGoogleGenerativeAI(
        model=model_name,                    # Gemini model name
        temperature=0,                       # Deterministic responses for QA
        google_api_key=settings.GOOGLE_API_KEY,
        convert_system_message_to_human=True # Gemini doesn't support system messages natively
    )

    return llm


def load_llm(model_name: str = "gemini-2.0-flash"):
    """
    Backward-compatible wrapper to support older imports.
    Internally calls get_llm().
    """
    return get_llm(model_name)