import sys
from pathlib import Path

import streamlit as st

# Allow `backend` imports when running from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm import ChatMessage, OpenAIChatProvider, ProviderError  # noqa: E402


st.set_page_config(page_title="yaf-gpt")

st.title("yaf-gpt")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    stream = st.toggle("Stream", value=True)
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    provider = OpenAIChatProvider()

    with st.chat_message("assistant"):
        try:
            if stream:
                placeholder = st.empty()
                content = ""
                chat_messages = [ChatMessage(**m) for m in st.session_state.messages]
                for chunk in provider.stream(
                    chat_messages, model=model, temperature=float(temperature)
                ):
                    content += chunk.content_delta
                    placeholder.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content})
            else:
                chat_messages = [ChatMessage(**m) for m in st.session_state.messages]
                response = provider.generate(
                    chat_messages, model=model, temperature=float(temperature)
                )
                st.markdown(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )
        except ProviderError as exc:
            st.error(str(exc))
