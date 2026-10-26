
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DATA_GOV_IN_API_KEY")
URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)

def main():
    if not API_KEY:
        print("API key missing. Check the root .env file.")
        return

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1,
    }

    try:
        print("Connecting to the dataset...")
        response = requests.get(
            URL,
            params=params,
            timeout=(10, 60),
        )

        print("HTTP status:", response.status_code)
        response.raise_for_status()

        data = response.json()
        print("Dataset title:", data.get("title", "Not provided"))
        print("Total records:", data.get("total", "Not provided"))

        records = data.get("records", [])
        print("Records received:", len(records))

        if records:
            print("Available fields:", list(records[0].keys()))
            print("Sample record:", records[0])
        else:
            print("No records returned.")

    except requests.Timeout:
        print("The API timed out again.")
        print("The key has not been confirmed invalid.")

    except requests.RequestException as exc:
        print("Request failed:", type(exc).__name__)
        if exc.response is not None:
            print("HTTP status:", exc.response.status_code)
            print("Response:", exc.response.text[:300])

    except ValueError:
        print("The server did not return valid JSON.")

if __name__ == "__main__":
    main()
