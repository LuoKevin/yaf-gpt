from dataclasses import dataclass
from enum import Enum
from langgraph.graph import END, StateGraph

class UserIntent(Enum):
    GENERAL_QUERY = "general_query"
    BIBLE_STUDY = "bible_study"
    SPIRIT_ADVICE = "spirit_advice"
    PASSAGE_RECS = "passage_recs"


class GraphState:
    intent: UserIntent
    passage_reference: str
    user_id: str

def build_graph() -> StateGraph:
    """Build the state graph for the application."""
    graph = StateGraph(GraphState)

    # Define states
    start_state = graph.add_state("start")
    processing_state = graph.add_state("processing")
    end_state = graph.add_state("end", is_end=True)

    # Define transitions
    graph.add_transition(start_state, processing_state, condition=lambda x: True)
    graph.add_transition(processing_state, end_state, condition=lambda x: True)

    return graph