# 🤖 AI Data Agent

An LLM-powered, multi-agent data processing and analysis project built with **Python, LangChain, LangGraph, Pydantic, Pandas, and PostgreSQL**. It accepts natural-language requests and routes each request to either an ETL workflow or an SQL analysis workflow.

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Running the Project](#-running-the-project)
- [Example Requests](#-example-requests)
- [Agent Workflows](#-agent-workflows)
- [Security Notes](#-security-notes)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

The AI Data Agent lets a user describe a data task in natural language. A router LLM classifies the request and sends it to the appropriate specialist:

- **ETL Analyst** — retrieves data from an API or transforms an existing CSV, JSON, or Parquet file, then saves the result.
- **SQL Analyst** — refines a database question, gathers PostgreSQL schema context, generates a SQL query, checks it with an LLM-based safety judge, executes it if approved, and explains the result.

LangGraph coordinates the nodes and conditional edges. Pydantic `BaseModel` classes define the graph state and structured LLM responses.

## 🏗️ Architecture

```mermaid
flowchart TD
    A([User Request]) --> B[Data Agent Graph]
    B --> C[Router Node]
    C --> D{Route Decision}
    D -->|ETL| E[ETL Analyst Graph]
    D -->|SQL| F[SQL Analyst Graph]

    E --> G[Extract / Transform Tools]
    G --> H[Save Data and Report Result]

    F --> I[Curate Question]
    I --> J[Gather PostgreSQL Schema]
    J --> K[Generate SQL]
    K --> L[LLM Safety Judge]
    L -->|Approved| M[Execute SQL]
    L -->|Rejected| N[Return Safety Message]
    M --> O[Generate Final Answer]

    H --> P([Response])
    N --> P
    O --> P
```

### Request flow

1. The user submits a request through the configured Python entry point.
2. The router returns a structured classification: `etl` or `sql`.
3. LangGraph conditionally dispatches the request to the selected sub-agent.
4. The selected agent runs its workflow and tools.
5. The parent Data Agent returns the specialist's final response.

## ✨ Features

### Data Agent router

- Classifies natural-language requests as ETL or SQL.
- Uses a Pydantic `RouterSchema` for structured routing output.
- Uses LangGraph conditional edges to choose a specialist agent.

### ETL Analyst

- Extracts JSON data from an HTTP API endpoint.
- Normalizes JSON records into a Pandas DataFrame.
- Saves extracted data as CSV, JSON Lines, or Parquet.
- Reads CSV, JSON Lines, or Parquet inputs for transformation context.
- Uses an LLM to generate Pandas transformation code from the user's request.
- Executes the generated code and reports the result.

### SQL Analyst

- Refines the user's question using an LLM.
- Inspects tables, columns, data types, and sample rows in a PostgreSQL schema.
- Generates SQL using the database schema and question as context.
- Uses a structured LLM safety-judge response (`Yes` or `No`).
- Executes the query when the safety judge approves it.
- Produces a concise natural-language answer from the query result.

### State and orchestration

- LangGraph state graphs for the parent, ETL, and SQL workflows.
- Pydantic `BaseModel` schemas for graph state and structured outputs.
- Separate utility modules for LLM selection, ETL operations, and database access.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Application runtime |
| LangChain | LLM integration and tool calling |
| LangGraph | Graph-based workflow orchestration |
| Pydantic | State schemas and structured LLM output |
| Pandas | JSON normalization and data transformation |
| Requests | HTTP API requests |
| SQLAlchemy | PostgreSQL connection and query execution |
| PostgreSQL | Database used by the SQL Analyst |
| python-dotenv | Loading environment variables from `.env` |
| uv | Project and dependency management |

## 📁 Project Structure

```text
Data_Agent/
├── agents/
│   ├── data_agent.py       # Parent router graph
│   ├── etl_analyst.py      # ETL agent and ETL tool wrappers
│   └── sql_analyst.py      # SQL analysis graph
├── models/
│   ├── __init__.py         # Exports Pydantic schemas
│   └── schema.py           # Agent and router schemas
├── utils/
│   ├── __init__.py         # Exposes shared utilities
│   ├── llm_pick.py         # Selects the LLM by complexity level
│   ├── etl_tools.py        # API extraction, file context, code execution
│   └── database.py         # PostgreSQL connections, schema and SQL helpers
├── main.py                 # Main runnable example
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version (3.12)
├── .gitignore
├── sql_analyst_agent.png   # SQL agent diagram/image
└── README.md
```

## ✅ Prerequisites

- Python **3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An API key and access to the LLM models configured in `utils/llm_pick.py`
- A running PostgreSQL database and credentials for SQL analysis
- Network access to any API endpoints used for ETL extraction

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akhil351/Data_Agent.git
cd Data_Agent
```

### 2. Install dependencies

Install `uv` if it is not already installed, then run:

```bash
uv sync
```

This uses `pyproject.toml` and the committed `uv.lock` to create/synchronize the project environment.

If you prefer a manually managed virtual environment, create one with Python 3.12 and install the project dependencies declared in `pyproject.toml`.

## ⚙️ Configuration

Create a `.env` file in the project root. The application loads environment variables with `python-dotenv`.

At minimum, configure the LLM API key and PostgreSQL connection string:

```dotenv
OPENAI_API_KEY=your_llm_api_key
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/database_name
```

Replace the example values with your own credentials. URL-encode special characters in a database password when placing it inside a connection URL.

### PostgreSQL setup

1. Install and start PostgreSQL.
2. Create the database and schema/tables you want to query.
3. Set `DATABASE_URL` to a valid SQLAlchemy PostgreSQL connection URL.
4. Ensure the configured database user can read the schema and data needed by the agent.

The SQL agent currently requests schema details for the PostgreSQL `public` schema. Its usefulness therefore depends on the tables and sample data available there.

### LLM model configuration

`utils/llm_pick.py` defines three model-selection levels—`low`, `medium`, and `high`—and creates `ChatOpenAI` instances for them. The configured model identifiers must be available to the API endpoint/account you use. If you change providers or model names, update this utility and its environment configuration accordingly.

> **Important:** `utils/__init__.py` imports the database utilities, and `utils/database.py` raises an error if `DATABASE_URL` is missing. Configure `DATABASE_URL` before running the application, including when you only intend to test the ETL workflow with the current package imports.

## 💻 Running the Project

Run the main example from the repository root:

```bash
uv run python main.py
```

The current `main.py` example asks the SQL Analyst:

> What are the different types of payment methods we have in our database?

The database must contain the relevant data for this example to return a meaningful answer. To try a different request, edit the `HumanMessage` content in `main.py`.

You can also run an agent module's built-in example from the project root, for example:

```bash
uv run python -m agents.etl_analyst
uv run python -m agents.sql_analyst
```

Use the module invocation form so Python resolves the project packages consistently.

## 💬 Example Requests

### ETL: Extract API data

```text
Extract data from https://pokeapi.co/api/v2/pokemon and save it in data/extract in CSV format.
```

Expected behavior: the ETL Analyst selects the extraction tool, requests the endpoint, normalizes the response, and saves `extracted_data.csv` under the requested output folder. The API and network must be available.

### ETL: Transform a file

```text
Transform data/extract/extracted_data.csv, keep only Bulbasaur Pokémon, and save the result in data/transform in CSV format.
```

Expected behavior: the ETL Analyst reads a small sample for context, asks the LLM to generate Pandas code, executes that code, and reports the execution result. The input file must already exist.

### SQL: Ask a database question

```text
What are the different types of payment methods we have in our database?
```

Expected behavior: the SQL Analyst inspects the `public` schema, generates a query, sends it through the configured safety-judge step, and either executes the approved query or returns a message that it was rejected.

These are example prompts, not guarantees of a successful result: outputs depend on the API response, files, database schema/data, credentials, and configured LLMs.

## 🤖 Agent Workflows

### 1. Data Agent — `agents/data_agent.py`

- Receives the user's latest message.
- Calls the router LLM using `RouterSchema`.
- Validates the route against `etl` and `sql`.
- Invokes the matching sub-agent.
- Returns the sub-agent's final answer as a message.

### 2. ETL Analyst — `agents/etl_analyst.py`

The ETL graph uses an LLM node and a tool node. After each LLM response, a conditional edge checks whether tool calls are present. If so, the tools run and the result is returned to the LLM; otherwise, the graph ends.

Available tool wrappers:

- `extract_load_tool(url, output_folder, format)` — extracts API JSON and writes CSV, JSON Lines, or Parquet output.
- `transform_load_tool(input_file_path, output_folder, output_format, user_question)` — prepares input context, requests Pandas code from the LLM, executes the generated code, and returns a status message.

### 3. SQL Analyst — `agents/sql_analyst.py`

The SQL graph follows this sequence:

1. `curate_question` — rewrites the question for clarity.
2. `prompt_query_context` — gathers schema details and builds the SQL-generation prompt.
3. `generate_sql` — generates the SQL query.
4. `is_safe_sql` — asks an LLM judge whether the query is read-only/safe.
5. `execute_sql` or `canceled_sql` — executes the approved query or returns a rejection message.
6. `represent_final_answer` — turns the query result into a user-friendly answer.

## 🧱 Data Models

The schemas are defined in `models/schema.py`:

- **`DataAgentSchema`** — parent graph messages and routing response.
- **`ETLAgentSchema`** — messages processed by the ETL graph.
- **`AgentSchema`** — SQL question, curated question, prompt context, generated query, safety result, execution result, and final answer.
- **`RouterSchema`** — router classification (`sql` or `etl`) and comments.
- **`JudgeSchema`** — safety-judge result (`Yes` or `No`) and comments.

## 🔐 Security Notes and Limitations

- **Generated Python execution:** `utils/etl_tools.py` uses Python `exec()` to execute LLM-generated transformation code. This is arbitrary code execution, not a sandbox. Do not expose this application to untrusted users or run it with access to sensitive files/secrets until execution is isolated in a properly restricted environment.
- **SQL safety check:** the SQL workflow uses an LLM-based judge before execution. An LLM judgment is not a hard security boundary and can be wrong. Use a dedicated PostgreSQL account with read-only permissions, restrict network/database access, and apply database-side limits where possible.
- **Database helper behavior:** `execute_sql()` runs the supplied statement through SQLAlchemy and commits the session. Do not rely only on the LLM judge to prevent modifications; enforce permissions in PostgreSQL.
- **Secrets:** keep `.env` out of version control. Never commit API keys, passwords, or production database URLs.
- **External inputs:** validate API endpoints, file paths, output formats, and user instructions before production use.

## 🛠️ Troubleshooting

### `DATABASE_URL is missing from .env`

Create `.env` in the repository root and set a valid `DATABASE_URL`. Check that the URL uses the correct PostgreSQL driver and credentials.

### PostgreSQL connection errors

Confirm that PostgreSQL is running, the database exists, the credentials are correct, and the configured user can access the `public` schema.

### LLM authentication or model errors

Check `OPENAI_API_KEY`, confirm that the selected model identifiers are supported by your account/provider, and verify network access.

### ETL extraction errors

Check the endpoint URL, network connectivity, API response format, permissions, and output folder. The extraction helper expects a JSON response.

### Transformation errors

Check that the input file exists and uses CSV, JSON Lines, or Parquet. Inspect the tool's returned execution error. Generated code may fail if it assumes columns or paths that do not exist.

### Missing Python packages

From the repository root, run:

```bash
uv sync
```

## 👨‍💻 Author

**Akhil**

- GitHub: [@Akhil351](https://github.com/Akhil351)
- Repository: [Data_Agent](https://github.com/Akhil351/Data_Agent)
