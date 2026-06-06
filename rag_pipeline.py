import os
import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from multiple URL formats."""

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"live/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(
        "Could not extract a valid YouTube video ID from the URL."
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load local embedding model."""

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ------------------------------------------------------------------
# Transcript Loading
# ------------------------------------------------------------------

def load_video(url: str) -> tuple[list[Document], str]:
    """
    Load YouTube transcript and convert it into LangChain documents.
    """

    video_id = extract_video_id(url)

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        raw = [
            {
                "text": item.text,
                "start": item.start,
                "duration": item.duration,
            }
            for item in transcript
        ]

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise ValueError(f"Transcript unavailable: {e}")

    except Exception as e:
        raise ValueError(
            f"Failed to fetch transcript from YouTube: {e}"
        )

    if not raw:
        raise ValueError(
            "Transcript is empty. Cannot build a knowledge base."
        )

    documents = []

    buffer_text = []
    segment_start = raw[0]["start"]
    accumulated = 0.0

    for entry in raw:

        buffer_text.append(entry["text"])
        accumulated += entry.get("duration", 0)

        if accumulated >= 60:

            documents.append(
                Document(
                    page_content=" ".join(buffer_text),
                    metadata={
                        "start": segment_start,
                        "timestamp_url": (
                            f"https://youtu.be/{video_id}"
                            f"?t={int(segment_start)}"
                        ),
                        "video_id": video_id,
                    },
                )
            )

            buffer_text = []
            segment_start = (
                entry["start"] + entry.get("duration", 0)
            )
            accumulated = 0.0

    # Flush remaining transcript

    if buffer_text:
        documents.append(
            Document(
                page_content=" ".join(buffer_text),
                metadata={
                    "start": segment_start,
                    "timestamp_url": (
                        f"https://youtu.be/{video_id}"
                        f"?t={int(segment_start)}"
                    ),
                    "video_id": video_id,
                },
            )
        )

    return documents, video_id


# ------------------------------------------------------------------
# Vector Store
# ------------------------------------------------------------------

def create_or_load_vectorstore(
    documents: list[Document],
    video_id: str,
) -> FAISS:

    cache_path = os.path.join("faiss_cache", video_id)

    embeddings = _get_embeddings()

    if os.path.exists(cache_path):

        return FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    os.makedirs(cache_path, exist_ok=True)

    vectorstore.save_local(cache_path)

    return vectorstore


# ------------------------------------------------------------------
# Question Answering
# ------------------------------------------------------------------

def get_answer(
    vectorstore: FAISS,
    question: str,
    api_key: str,
) -> tuple[str, list[Document]]:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    source_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in source_docs
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant that answers questions strictly
based on a YouTube video transcript.

Rules:
- Use ONLY the transcript context provided below.
- If the answer is not in the context, say exactly:
  "I couldn't find that in this video."
- Be concise and direct.
- Do not make up information.

Transcript context:
{context}
"""
    ),
    ("human", "{question}")
])
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer, source_docs