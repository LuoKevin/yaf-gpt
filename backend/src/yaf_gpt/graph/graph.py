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
    builder = StateGraph(GraphState)
    builder.add_node("classify_intent")
    builder.add_node("handle_general_query")
    builder.add_node("handle_bible_study")
    builder.add_node("handle_spirit_advice")
    builder.add_node("handle_passage_recs")
    builder.add_edge("classify_intent", "handle_general_query", condition=lambda state: state.intent == UserIntent.GENERAL_QUERY)
    builder.add_edge("classify_intent", "handle_bible_study", condition=lambda state: state.intent == UserIntent.BIBLE_STUDY)
    builder.add_edge("classify_intent", "handle_spirit_advice", condition=lambda state: state.intent == UserIntent.SPIRIT_ADVICE)
    builder.add_edge("classify_intent", "handle_passage_recs", condition=lambda state: state.intent == UserIntent.PASSAGE_RECS)
    builder.add_edge("handle_general_query", END)
    builder.add_edge("handle_bible_study", END)
    builder.add_edge("handle_spirit_advice", END)
    builder.add_edge("handle_passage_recs", END)    
    return builder