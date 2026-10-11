import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL") or ""

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")

DB_CONFIG = {
    "POOL_SIZE": int(os.getenv("DB_POOL_SIZE", "5")),
    "MAX_OVERFLOW": int(os.getenv("DB_MAX_OVERFLOW", "10")),
    "POOL_TIMEOUT": int(os.getenv("DB_POOL_TIMEOUT", "30")),
    "POOL_RECYCLE": int(os.getenv("DB_POOL_RECYCLE", "1800")),
}

_engine_cache: dict[str, Engine] = {}
_sessionmaker_cache: dict[str, Any] = {}


def create_db_engine(
    db_conn_string: str,
    echo: bool = False,
    **kwargs: Any,
) -> Engine:
    """Create or return a cached SQLAlchemy engine."""

    if db_conn_string in _engine_cache:
        return _engine_cache[db_conn_string]

    default_settings = {
        "pool_pre_ping": True,
        "pool_size": DB_CONFIG["POOL_SIZE"],
        "max_overflow": DB_CONFIG["MAX_OVERFLOW"],
        "pool_timeout": DB_CONFIG["POOL_TIMEOUT"],
        "pool_recycle": DB_CONFIG["POOL_RECYCLE"],
    }

    engine_settings = {**default_settings, **kwargs}

    engine = create_engine(
        db_conn_string,
        echo=echo,
        **engine_settings,
    )

    _engine_cache[db_conn_string] = engine
    return engine


def create_db_session(
    engine: Engine,
    expire_on_commit: bool = True,
    autoflush: bool = False,
) -> Session:
    """Create a new session using a cached session factory."""

    cache_key = f"{engine.url}_{expire_on_commit}_{autoflush}"

    if cache_key not in _sessionmaker_cache:
        _sessionmaker_cache[cache_key] = sessionmaker(
            bind=engine,
            expire_on_commit=expire_on_commit,
            autoflush=autoflush,
        )

    session_factory = _sessionmaker_cache[cache_key]
    return session_factory()


def get_db_session() -> Session:
    """Return a new database session."""

    engine = create_db_engine(DATABASE_URL)
    return create_db_session(engine)


def schema_details(schema_name: str) -> str:
    """Return table names, columns, data types, and up to five sample rows."""

    session = get_db_session()

    try:
        engine = session.get_bind()
        inspector = inspect(engine)
        preparer = engine.dialect.identifier_preparer

        schema_info = f"Database Schema: {schema_name}\n"
        tables = inspector.get_table_names(schema=schema_name)

        for table_name in tables:
            schema_info += f"\nTable: {table_name}\n"

            columns = inspector.get_columns(
                table_name,
                schema=schema_name,
            )

            for column in columns:
                schema_info += (
                    f"  Column: {column['name']}, " f"Data Type: {column['type']}\n"
                )

            quoted_schema = preparer.quote_schema(schema_name)
            quoted_table = preparer.quote(table_name)

            sample_query = text(f"SELECT * FROM {quoted_schema}.{quoted_table} LIMIT 5")

            sample_rows = session.execute(sample_query).fetchall()

            schema_info += "  Sample Data:\n"

            for row in sample_rows:
                schema_info += f"    {tuple(row)}\n"

        if not tables:
            schema_info += "No tables found in this schema.\n"

        return schema_info

    except Exception as error:
        return f"Error fetching schema details: {error}"

    finally:
        session.close()


def execute_sql(query: str) -> str | None:
    """Execute SQL and return results as a string, or None on error."""

    session = get_db_session()

    try:
        result = session.execute(text(query))
        column_names = result.keys()

        if len(column_names) > 0:
            rows = result.fetchall()
            output = str([tuple(row) for row in rows])
        else:
            output = "Query executed successfully."

        session.commit()
        return output

    except Exception as error:
        session.rollback()
        print(f"Error executing query: {error}")
        return None

    finally:
        session.close()
