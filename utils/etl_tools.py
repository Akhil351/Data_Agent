import os
import requests
import pandas as pd


def extract_load(url: str, output_folder: str, output_format: str) -> str:
    """
    Extract data from an API and save it to a local file.

    Args:
        url: API endpoint.
        output_folder: Destination folder.
        output_format: csv, json, or parquet.

    Returns:
        Success or failure message.
    """

    supported_formats = {"csv", "json", "parquet"}

    if output_format.lower() not in supported_formats:
        return f"Unsupported format: {output_format}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Support APIs returning {"results": [...]} or a list
        if isinstance(data, dict):
            records = data.get("results", data)
        else:
            records = data

        if isinstance(records, dict):
            records = [records]

        df = pd.json_normalize(records)

        # Save directly to the specified folder
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.join(
            output_folder, f"extracted_data.{output_format.lower()}"
        )

        if output_format.lower() == "csv":
            df.to_csv(filename, index=False)

        elif output_format.lower() == "json":
            df.to_json(filename, orient="records", lines=True)

        elif output_format.lower() == "parquet":
            df.to_parquet(filename, index=False)

        return (
            f"Data successfully extracted and saved to {filename}. "
            f"Rows extracted: {len(df)}"
        )

    except (requests.exceptions.RequestException, ValueError, OSError) as e:
        return f"Failed to extract data: {e}"

    except Exception as e:
        return f"Failed to process API data: {e}"


def transform_load_context(file_path: str) -> str:
    """
    Read a CSV, JSON, or Parquet file and return its first three rows.
    """

    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(file_path)

        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)

        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)

        else:
            return f"Unsupported file format: {file_extension}"

        if df.empty:
            return "The input file contains no data."

        return df.head(3).to_string(index=False)

    except Exception as e:
        return f"Failed to read or process the file: {e}"


def execute_code(code: str) -> str:
    """
    Execute Python code and return its execution status.

    WARNING:
        Only use with trusted code. Arbitrary exec() can execute
        dangerous operations and is not a secure sandbox.
    """

    try:
        exec(code, {})
        return "Code executed successfully."

    except Exception as e:
        return f"Failed to execute code: {e}"


if __name__ == "__main__":

    api_url = "https://pokeapi.co/api/v2/pokemon"

    # Extract API data into the existing data/extract folder
    output_folder = "data/extract"

    print(extract_load(url=api_url, output_folder=output_folder, output_format="csv"))

    # Read the saved CSV file
    file_path = os.path.join(output_folder, "extracted_data.csv")

    print("\nFirst three rows:")
    print(transform_load_context(file_path))
