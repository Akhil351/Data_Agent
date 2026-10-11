from typing import cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from utils import pick_llm
from models import RouterSchema, DataAgentSchema, AgentSchema, ETLAgentSchema
from .etl_analyst import etl_analyst
from .sql_analyst import sql_analyst

# ==================================================
# 1. Configure the router LLM
# ==================================================

llm = pick_llm("high")
llm_router = llm.with_structured_output(RouterSchema)


# ==================================================
# 2. Router node
# ==================================================


def router_node(state: DataAgentSchema) -> dict:
    message = state.messages[-1].content

    route_response = cast(
        RouterSchema,
        llm_router.invoke(str(message)),
    )

    route = route_response.answer.strip().lower()

    if route not in {"etl", "sql"}:
        raise ValueError(f"Invalid route response: {route}")

    return {"route_response": route}


# ==================================================
# 3. ETL node
# ==================================================


def etl_node(state: DataAgentSchema) -> dict:
    user_message = state.messages[-1].content

    response = etl_analyst.invoke(
        ETLAgentSchema(messages=[HumanMessage(content=str(user_message))])
    )

    # Get the messages returned by the ETL agent.
    messages = (
        response.messages
        if isinstance(response, ETLAgentSchema)
        else response.get("messages", [])
    )

    # Find the final AI response without tool calls.
    answer = "The ETL agent did not return a final answer."

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            answer = str(message.content)
            break

    return {"messages": [AIMessage(content=answer)]}


# ==================================================
# 4. SQL node
# ==================================================


def sql_node(state: DataAgentSchema) -> dict:
    user_message = state.messages[-1].content

    input_schema = AgentSchema(
        messages=[],
        user_question=str(user_message),
        curated_question="",
        prompt_query_context="",
        generated_sql_query="",
        is_safe="No",
        comments="",
        sql_query_execution_result="",
        final_answer="",
    )

    response = sql_analyst.invoke(input_schema)

    # Get the final answer from the SQL agent's state.
    if isinstance(response, AgentSchema):
        answer = response.final_answer
    else:
        answer = response.get("final_answer", "")

    if not answer:
        answer = "The SQL agent did not return a final answer."

    return {"messages": [AIMessage(content=str(answer))]}


# ==================================================
# 5. Build the data agent graph
# ==================================================

data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")


# ==================================================
# 6. Route to the appropriate agent
# ==================================================


def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"

    if state.route_response == "etl":
        return "etl_node"

    raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "sql_node": "sql_node",
        "etl_node": "etl_node",
    },
)

data_agent_graph.add_edge("sql_node", END)
data_agent_graph.add_edge("etl_node", END)

data_agent = data_agent_graph.compile()


# ==================================================
# 7. Run the data agent
# ==================================================

if __name__ == "__main__":
    response = data_agent.invoke(
        DataAgentSchema(
            messages=[
                HumanMessage(
                    content=(
                        "Transform the data stored in "
                        "'data/extract/extracted_data.csv'. "
                        "Save the transformed data in "
                        "'data/transform' in CSV format. "
                        "Filter the data to show Bulbasaur Pokémon only."
                    )
                )
            ],
            route_response="",
        )
    )

    print("\nData Agent Response:\n")

    # Print the final assistant response.
    messages = (
        response.messages
        if isinstance(response, DataAgentSchema)
        else response.get("messages", [])
    )

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            print(message.content)
            break
