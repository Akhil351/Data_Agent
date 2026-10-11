from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from utils import (
    pick_llm,
    extract_load,
    transform_load_context,
    execute_code,
)
from models import ETLAgentSchema

# ==================================================
# 1. Extraction and loading tool
# ==================================================


@tool
def extract_load_tool(
    url: str,
    output_folder: str,
    format: str,
) -> str:
    """Extract data from a URL and save it in the requested format."""

    try:
        result = extract_load(url, output_folder, format)
        return str(result)

    except Exception as error:
        return f"Extraction and loading failed: " f"{type(error).__name__}: {error}"


# ==================================================
# 2. Transformation and loading tool
# ==================================================


@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str,
    output_format: str,
    user_question: str,
) -> str:
    """Transform data according to the user's request and save it."""

    try:
        # Read a sample of the input data for context.
        sample_data = transform_load_context(input_file_path)

        # Select the LLM used to generate transformation code.
        llm = pick_llm("high")

        prompt = f"""
You are a Python data analyst who uses Pandas.

Generate executable Python code to transform the input data
according to the user's request.

Input file: {input_file_path}
Output folder: {output_folder}
Output format: {output_format}

Requirements:
1. Load the input file into a Pandas DataFrame.
2. Apply the requested transformation.
3. Create the output folder if it does not exist.
4. Save the transformed data in the specified output folder.
5. Use the requested output format: CSV, JSON, or Parquet.
6. Use an appropriate output filename.
7. Never overwrite the input file.
8. Return only Python code, without explanations or Markdown fences.

User request:
{user_question}

Sample input data:
{sample_data}
"""

        response = llm.invoke(prompt).content

        # Support string responses and text-block responses.
        if isinstance(response, str):
            pandas_code = response

        elif isinstance(response, list):
            text_parts = []

            for block in response:
                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    text = block.get("text")

                    if isinstance(text, str):
                        text_parts.append(text)

            pandas_code = "\n".join(text_parts)

        else:
            raise TypeError(
                f"Unexpected LLM response type: " f"{type(response).__name__}"
            )

        if not pandas_code.strip():
            raise ValueError("The LLM did not generate Python code.")

        # Remove Markdown code fences if present.
        if pandas_code.strip().startswith("```"):
            lines = pandas_code.strip().splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            pandas_code = "\n".join(lines).strip()

        # Execute the generated Python code.
        execution_result = execute_code(pandas_code)

        # Check the failure format used by execute_code().
        if isinstance(execution_result, str) and execution_result.startswith(
            "Failed to execute code:"
        ):
            return "Transformation failed.\n" f"Reason: {execution_result}"

        return (
            "Transformation tool finished.\n"
            f"Input file: {input_file_path}\n"
            f"Output folder: {output_folder}\n"
            f"Output format: {output_format}\n\n"
            f"Generated Python code:\n{pandas_code}\n\n"
            f"Execution result:\n{execution_result}"
        )

    except Exception as error:
        return f"Transformation failed: " f"{type(error).__name__}: {error}"


# ==================================================
# 3. Register tools with the LLM
# ==================================================

tools = [
    extract_load_tool,
    transform_load_tool,
]

llm = pick_llm("high")
llm_bind = llm.bind_tools(tools)


# ==================================================
# 4. LLM node
# ==================================================


def llm_node(state: ETLAgentSchema):
    system_message = SystemMessage(
        content=(
            "You are an ETL data analyst with tools for extracting, "
            "loading, and transforming data. Use the appropriate tool "
            "to complete the user's request. Supply all required tool "
            "arguments. After a tool returns, inspect its result. "
            "If the tool reports an error, explain the error honestly. "
            "Do not claim a task succeeded unless the tool result "
            "supports that conclusion. When the task is complete, "
            "respond clearly to the user without calling more tools."
        )
    )

    response = llm_bind.invoke([system_message] + state.messages)

    # The add_messages reducer merges these new messages into the state.
    return {"messages": [response]}


# ==================================================
# 5. Tool node
# ==================================================


def tool_node(state: ETLAgentSchema):
    tools_by_name = {current_tool.name: current_tool for current_tool in tools}

    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        raise TypeError("Expected an AIMessage containing tool calls.")

    tool_results = []

    for tool_call in last_message.tool_calls:
        selected_tool = tools_by_name.get(tool_call["name"])

        if selected_tool is None:
            result = f"Unknown tool: {tool_call['name']}"
        else:
            try:
                result = selected_tool.invoke(tool_call["args"])
            except Exception as error:
                result = f"Tool execution failed: {error}"

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_results}


# ==================================================
# 6. Conditional routing
# ==================================================


def is_tool_call(state: ETLAgentSchema):
    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    return END


# ==================================================
# 7. Build the LangGraph workflow
# ==================================================

etl_analyst_graph = StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        END: END,
    },
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()


# ==================================================
# 8. Run the ETL agent
# ==================================================

if __name__ == "__main__":

    response = etl_analyst.invoke(
        ETLAgentSchema(
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
            ]
        )
    )

    print(response)


# response = etl_analyst.invoke(
#         ETLAgentSchema(
#             messages=[
#                 HumanMessage(
#                     content=(
#                         "I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder"
#                     )
#                 )
#             ]
#         )
#     )