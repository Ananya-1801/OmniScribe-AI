import streamlit as st
import concurrent.futures
import traceback

from rag_pipeline import (
    load_video,
    create_or_load_vectorstore,
    get_answer,
)


st.set_page_config(
    page_title="OmniScribe AI",
    page_icon="🗪",
    layout="centered"
)


with st.sidebar:
    st.title("⚙️ Setup")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at groq.com — no credit card needed"
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Enter your free Groq API key")
    st.markdown("2. Paste any YouTube URL")
    st.markdown("3. Click **Process Video**")
    st.markdown("4. Ask anything about the video")
    st.markdown("---")
    st.markdown("**Stack**")
    st.caption("LangChain · FAISS · HuggingFace · Groq LLaMA3 · Streamlit")
    st.caption("No OpenAI credits needed.")


st.title("🗪 OMNISCRIBE AI")
st.caption("Paste a YouTube URL and chat with the video using RAG — no OpenAI needed.")
st.markdown("---")


youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([2, 1])
with col1:
    process_btn = st.button("⚡ Process Video", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.video_title = None
        st.rerun()

 
if process_btn:

    if not groq_api_key:
        st.error("Please enter your Groq API key.")

    elif not youtube_url:
        st.error("Please enter a YouTube URL.")

    else:

        try:
            with st.spinner("📥 Loading transcript..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(load_video, youtube_url)
                    documents, video_id = future.result()
            st.success(
                f"Transcript loaded successfully ({len(documents)} chunks)"
            )

            with st.spinner("Building vector database..."):

                vectorstore = create_or_load_vectorstore(
                    documents,
                    video_id
                )

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []
            st.session_state.video_title = youtube_url
            st.session_state.groq_key = groq_api_key
            st.session_state.video_id = video_id

            st.success(
                f"Video processed successfully. Video ID: {video_id}"
            )

        except Exception as e:

            st.error("Processing failed.")

            st.exception(e)

            with st.expander("Full Traceback"):

                st.code(traceback.format_exc())


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("vectorstore"):
    st.markdown(f"**Chatting with:** {st.session_state.video_title}")
    st.markdown("---")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources — click timestamps to jump to that moment"):
                    for i, doc in enumerate(msg["sources"]):
                        start = doc.metadata.get("start", 0)
                        url = doc.metadata.get("timestamp_url", "")
                        mins, secs = int(start // 60), int(start % 60)
                        st.caption(f"Chunk {i + 1} — [{mins:02d}:{secs:02d}]({url})")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")

    if question := st.chat_input("Ask something about this video..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning..."):
                answer, sources = get_answer(
                    st.session_state.vectorstore,
                    question,
                    st.session_state.groq_key
                )
            st.write(answer)
            with st.expander("📄 Sources — click timestamps to jump to that moment"):
                for i, doc in enumerate(sources):
                    start = doc.metadata.get("start", 0)
                    url = doc.metadata.get("timestamp_url", "")
                    mins, secs = int(start // 60), int(start % 60)
                    st.caption(f"Chunk {i + 1} — [{mins:02d}:{secs:02d}]({url})")
                    st.text(doc.page_content[:300] + "...")
                    st.markdown("---")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    st.info("👆 Enter your Groq API key in the sidebar, paste a YouTube URL above, and hit **Process Video**.")