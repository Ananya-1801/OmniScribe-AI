from rag_pipeline import (
    load_video,
    create_or_load_vectorstore,
)

url = "https://www.youtube.com/watch?v=cOAaonpTLlc"

print("Loading video...")

documents, video_id = load_video(url)

print("Documents:", len(documents))

print("Building vectorstore...")

vectorstore = create_or_load_vectorstore(
    documents,
    video_id
)

print("SUCCESS")