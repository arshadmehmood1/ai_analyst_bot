"""
assistant.py
Intelligent RAG Complaint Analyst using Gemini + LangChain Core.
"""

import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

from ..llm.model import load_llm
from ..llm.retrieval import load_retriever
from ..config.settings import settings

logger = logging.getLogger(__name__)


class ComplaintAssistant:
    def __init__(self):
        logger.info("Loading intelligent complaint assistant...")

        self.llm = load_llm()
        self.retriever = load_retriever()

        self.rewrite_prompt, self.rag_prompt = self._create_prompts()

        # SMART RAG CHAIN (modern LCEL)
        self.rag_chain = (
            {
                "question": RunnablePassthrough(),
                "re_query": self.rewrite_prompt | self.llm | StrOutputParser(),
            }
            | RunnableMap(
                {
                    "context": lambda x: self.retriever.invoke(x["re_query"]),
                    "question": lambda x: x["question"]
                }
            )
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

    # --------------------------------------------------------------
    # PROMPTS
    # --------------------------------------------------------------
    def _create_prompts(self):

        # 1) Query rewriting (makes your AI "smart")
        rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user query to be clearer and more retrieval-friendly.
Return ONLY the rewritten query.

Original Query:
{question}
""")

        # 2) Actual RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""
You are an **Internal Complaint Analyst**.

Use ONLY the following complaint records:

Context:
{context}

Question:
{question}

If the answer is not explicitly supported by the records, reply:
"Based on the historical complaint data provided, I cannot generate a specific finding."

When possible, include:
- complaint type
- issue summary
- action taken
- final outcome (Completed / Rejected / Resolved)

Respond concisely in an analytical, non-emotional tone.
""")

        return rewrite_prompt, rag_prompt

    # --------------------------------------------------------------
    # MAIN QUERY
    # --------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        try:
            result = self.rag_chain.invoke(question)

            docs = self.retriever.invoke(question)
            src_docs = [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]

            return {
                "answer": result,
                "source_documents": src_docs,
                "metadata": {
                    "num_sources": len(src_docs),
                    "model": settings.MODEL_NAME
                }
            }

        except Exception as e:
            logger.error(f"Error in query: {e}")
            raise

    # For direct calls
    def __call__(self, question: str):
        return self.query(question)


_assistant_instance: ComplaintAssistant = None


def get_assistant():
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = ComplaintAssistant()
    return _assistant_instance


def ask_question(question: str):
    return get_assistant().query(question)["answer"]
