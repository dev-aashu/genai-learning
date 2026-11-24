"""
Step 10: Streamlit UI for the agent app.
Talks only to the FastAPI backend (port 8001). The backend runs a LangGraph agent
(local Ollama model) whose tools come from the MCP server (07_mcp/server.py).
"""
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8001"

st.set_page_config(page_title="Local AI Agent", page_icon="🤖")
st.title("🤖 Local AI Agent")
st.caption("Streamlit → FastAPI → LangGraph agent (Ollama) → MCP server tools")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["qwen3:4b", "llama3.2:3b"])

    if st.button("🗑️ New chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    try:
        info = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.success("Backend: connected")
        st.write("**MCP tools:**")
        for t in info.get("tools", []):
            st.write(f"• `{t}`")
    except requests.RequestException:
        st.error("Backend: not running")
        st.caption("Start it with:\n\n`uvicorn backend:app --app-dir 10_agent_app --port 8001`")

    st.divider()
    st.write("**Try asking:**")
    st.caption("• What is 4837 * 2913?")
    st.caption("• Which files are in the data folder?")
    st.caption("• Read notes.txt and tell me when the meeting is.")
    st.caption("• What time is it now?")

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
            payload = {"messages": st.session_state.messages, "model": model}
            with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

        try:
            with st.spinner("Agent is working..."):
                answer = st.write_stream(stream_answer())
        except requests.RequestException as e:
            answer = f"❌ Could not reach backend: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})