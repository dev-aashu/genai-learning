"""
Step 09: Streamlit chat UI. Talks only to the FastAPI backend, never to Ollama directly.
"""
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Local AI Chat", page_icon="🦙")
st.title("🦙 Local AI Chat")
st.caption("Streamlit frontend → FastAPI backend → Ollama (runs fully on my Mac)")

# ---------- Sidebar ----------
with st.sidebar:
    model = st.selectbox("Model", ["llama3.2:3b", "qwen3:4b"])
    if st.button("🗑️ New chat"):
        st.session_state.messages = []
        st.rerun()
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=2).raise_for_status()
        st.success("Backend: connected")
    except requests.RequestException:
        st.error("Backend: not running")

# ---------- Chat history (kept across reruns) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- New message ----------
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def stream_answer():
            payload = {
                "messages": st.session_state.messages,
                "model": model,
                "think": model.startswith("qwen3"),
            }
            with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

        try:
            answer = st.write_stream(stream_answer())
        except requests.RequestException as e:
            answer = f"❌ Could not reach backend: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})