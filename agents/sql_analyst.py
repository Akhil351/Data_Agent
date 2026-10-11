from utils import (
    pick_llm,
    schema_details,
    execute_sql as execute_sql_query,
)

from models import AgentSchema, JudgeSchema
from typing import cast
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END


def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question

    llm = pick_llm("low")
    response = llm.invoke(f"Curate the following question: {user_question}").content

    if not isinstance(response, str):
        raise TypeError("Expected the LLM response to be a string")

    state.curated_question = response
    state.messages = state.messages + [
        AIMessage(content=f"Curated question: {response}")
    ]

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_question
    schema_info = schema_details("public")

    prompt = f"""
You are an SQL analyst agent. Convert the user's natural-language
question into a PostgreSQL query.

Use the database schema and sample data below to understand the
tables, columns, and relationships.

Unless the user explicitly asks for a specific number of rows,
limit the output to 10 rows.

Generate only the SQL query without explanations or Markdown fences.

The query must be read-only and must not modify the database.

User's question:
{curated_question}

Database schema details:
{schema_info}
"""

    state.prompt_query_context = prompt
    return state


def generate_sql(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context

    llm = pick_llm("medium")
    generated_sql_query = llm.invoke(prompt).content

    if not isinstance(generated_sql_query, str):
        raise TypeError("Expected the LLM response to be a string")

    state.generated_sql_query = generated_sql_query.strip()
    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    llm = pick_llm("medium")
    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
You are an SQL security judge.

Determine whether the SQL query is safe for read-only data retrieval.

Answer "Yes" only if the query is a permitted read-only query.
Answer "No" if it modifies data or database structure, or performs
any operation that is not permitted for read-only retrieval.

SQL query to evaluate:
{sql_query}
"""

    response = cast(JudgeSchema, llm_judge.invoke(prompt))

    state.is_safe = response.answer
    state.comments = response.comments

    return state


def canceled_sql(state: AgentSchema) -> AgentSchema:
    comments = state.comments

    state.final_answer = (
        "The generated SQL query was deemed unsafe to execute. "
        f"Reason: {comments}. The query was not executed."
    )

    state.messages = state.messages + [AIMessage(content=state.final_answer)]

    return state


def execute_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    execution_result = execute_sql_query(sql_query)

    state.sql_query_execution_result = (
        execution_result
        if execution_result is not None
        else "SQL query execution failed."
    )

    return state


def represent_final_answer(state: AgentSchema) -> AgentSchema:
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_question

    llm = pick_llm("low")

    prompt = f"""
You are an SQL analyst agent. Answer the user's question using
the SQL execution result.

Keep the answer concise, clear, and user-friendly.
Do not include SQL code or unnecessary technical details.
If the result is empty or insufficient, explain that clearly.

Execution result:
{execution_result}

User's question:
{curated_question}
"""

    llm_response = llm.invoke(prompt).content

    if not isinstance(llm_response, str):
        raise TypeError("Expected the LLM response to be a string")

    state.final_answer = llm_response

    state.messages = state.messages + [AIMessage(content=llm_response)]

    return state


def is_safe_sql_edge(state: AgentSchema) -> str:
    if state.is_safe.strip().lower() == "yes":
        return "execute_sql"

    return "canceled_sql"


# Build the graph
sql_agent_graph = StateGraph(AgentSchema)

# Add nodes
sql_agent_graph.add_node("curate_question", curate_question)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

# Connect the main workflow
sql_agent_graph.add_edge(START, "curate_question")
sql_agent_graph.add_edge("curate_question", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# Route according to the safety judgment
sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_sql_edge,
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql",
    },
)

# Connect the ending nodes
sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)

# Compile the graph
sql_analyst = sql_agent_graph.compile()


if __name__ == "__main__":
    input_schema = AgentSchema(
        messages=[],
        user_question=(
            "What are the different types of payment methods "
            "we have in our database?"
        ),
        curated_question="",
        prompt_query_context="",
        generated_sql_query="",
        is_safe="No",
        comments="",
        sql_query_execution_result="",
        final_answer="",
    )

    # Execute the graph
    sql_analyst_response = sql_analyst.invoke(input_schema)

    print("*" * 50)
    print("Final Answer:")
    print(sql_analyst_response["final_answer"])
