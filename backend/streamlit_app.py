from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `backend` imports when running from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.schemas import StudyPlanRequest  # noqa: E402
from backend.services.bible_lookup import (  # noqa: E402
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from backend.services.study_plan_service import (  # noqa: E402
    StudyPlanProviderError,
    StudyPlanService,
    StudyPlanValidationError,
)

st.set_page_config(page_title="yaf-gpt Study Plan", layout="wide")
st.title("yaf-gpt: Bible Study Plan Generator")
st.caption("Generate a structured small-group study plan from a Bible passage range.")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("Inputs")
    reference = st.text_input("Passage reference", value="Luke 21:5-28")
    translation = st.selectbox("Translation", options=["WEB", "KJV"], index=0)
    use_override = st.toggle("Use custom passage text", value=False)
    passage_text = ""
    if use_override:
        passage_text = st.text_area(
            "Custom passage text",
            height=220,
            placeholder="Paste passage text here...",
        )
    goals = st.text_area("Optional goals", height=100, placeholder="What should this study focus on?")
    user_notes = st.text_area(
        "Optional leader notes",
        height=100,
        placeholder="Context about your group (maturity level, time constraints, etc.)",
    )
    submitted = st.button("Generate Study Plan", type="primary", use_container_width=True)


def render_questions(questions: list[str]) -> None:
    for idx, question in enumerate(questions, start=1):
        with st.container(border=True):
            st.markdown(f"**Q{idx}. {question}**")


if submitted:
    payload = StudyPlanRequest(
        reference=reference.strip(),
        translation=translation,
        passage_text=passage_text.strip() or None,
        goals=goals.strip() or None,
        user_notes=user_notes.strip() or None,
    )

    with st.spinner("Generating study plan..."):
        service = StudyPlanService()
        try:
            result = service.generate_study_plan(payload)
            st.session_state.last_result = result
        except InvalidReferenceError as exc:
            st.error(f"Invalid reference: {exc}")
        except PassageNotFoundError as exc:
            st.error(f"Passage not found: {exc}")
        except (PassageProviderError, StudyPlanProviderError, StudyPlanValidationError) as exc:
            st.error(f"Generation failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

result = st.session_state.last_result
if result is not None:
    st.subheader(result.passage_title)
    st.caption(
        f"{result.normalized_reference} ({result.translation}) | "
        f"Model: {result.model}"
    )

    with st.expander("Passage", expanded=False):
        st.write(result.passage_text)

    st.markdown("### Context")
    for point in result.context_points:
        st.markdown(f"- {point}")

    st.markdown("### Questions")
    render_questions(result.discussion_questions)

    if result.usage is not None:
        st.caption(
            "Tokens | "
            f"prompt: {result.usage.prompt_tokens or 0}, "
            f"completion: {result.usage.completion_tokens or 0}, "
            f"total: {result.usage.total_tokens or 0}"
        )
