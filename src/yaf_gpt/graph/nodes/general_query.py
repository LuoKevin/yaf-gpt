

from src.yaf_gpt.graph.graph import GraphState

def handle_general_query(state: GraphState) -> str:
    client = get_openai_client()  # Assume this function gets an OpenAI client
    return f"This is a general query response to: {state.query}"