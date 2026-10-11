from models import DataAgentSchema
from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage, AIMessage

if __name__ == "__main__":
    response = data_agent.invoke(
        DataAgentSchema(
            messages=[
                HumanMessage(
                    content=(
                        "What are the different types of payment methods "
                        "we have in our database?"
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
