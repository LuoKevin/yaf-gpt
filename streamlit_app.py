import streamlit as st

from yaf_gpt.graph import build_chat_pipeline
from yaf_gpt.core.config import Settings

"""
TODO: hook up graph to this streamlit app
"""

@st.cache_resource
def _load_pipeline():
    settings = Settings()
    return build_chat_pipeline(settings=settings)


def main() -> None:
    st.set_page_config(page_title="YAF-GPT Chat", layout="wide")
    st.title("YAF-GPT — Streamlit Prototype")
    st.markdown("Ask about a passage, generate study notes, and explore the outputs in one place.")

    pipeline = _load_pipeline()

    reference = st.text_input("Passage reference", value="Luke 11:1-13")
    user_question = st.text_area("Question", value="Can you give me study notes for this passage?")

    if st.button("Run Study Helper"):
        with st.spinner("Generating responses..."):
            result = pipeline.invoke({"reference": reference, "question": user_question})

        st.subheader("Study Notes")
        st.write(result.get("study_notes", "No study notes returned."))

        st.subheader("Reflection Questions")
        for question in result.get("questions", []):
            st.write(f"- {question}")

        st.subheader("Life Application")
        for item in result.get("life_application", []):
            st.write(f"- {item}")

        if image := result.get("image"):
            st.subheader("Illustration")
            st.image(image)


if __name__ == "__main__":
    main()
