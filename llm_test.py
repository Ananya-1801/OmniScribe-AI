import os

from dotenv import load_dotenv

from rag_pipeline import (
    load_video,
    create_or_load_vectorstore,
    get_answer,
)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

url = "https://www.youtube.com/watch?v=cOAaonpTLlc"

documents, video_id = load_video(url)

vectorstore = create_or_load_vectorstore(
    documents,
    video_id
)

answer, sources = get_answer(
    vectorstore,
    "What is this video about?",
    api_key
)

print("\nANSWER:\n")
print(answer)