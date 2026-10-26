"""Diagnose AirSight's data.gov.in connection without printing the API key."""
import os
import sys
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("RESULT: python-dotenv is not installed.")
    print("Run: python -m pip install python-dotenv")
    sys.exit(1)

# Load the project-root .env regardless of the current working directory.
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")
key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()

if not key:
    print("RESULT: API key not found in project-root .env.")
    print("Check that .env contains DATA_GOV_IN_API_KEY=your_key")
    sys.exit(2)

url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
params = {"api-key": key, "format": "json", "limit": 5, "offset": 0}

print("Testing government API directly (5 records; key hidden)...")
try:
    response = requests.get(url, params=params, timeout=(5, 12))
    print("HTTP status:", response.status_code)
    if response.status_code != 200:
        print("RESULT: API returned an HTTP error.")
        print("Response excerpt:", response.text[:400].replace(key, "[REDACTED]"))
        sys.exit(3)
    try:
        data = response.json()
    except ValueError:
        print("RESULT: HTTP 200, but response was not valid JSON.")
        print("Response excerpt:", response.text[:300].replace(key, "[REDACTED]"))
        sys.exit(4)

    records = data.get("records", []) if isinstance(data, dict) else []
    print("Records returned:", len(records))
    print("Total reported by API:", data.get("total", "not provided"))
    if records:
        first = records[0]
        print("Record fields:", ", ".join(first.keys()) if isinstance(first, dict) else "unexpected format")
        print("RESULT: Direct API connection works. Problem is likely in backend code/configuration.")
    else:
        print("RESULT: API responded but returned no records. Check resource/key/API response.")
except requests.exceptions.ConnectTimeout:
    print("RESULT: Connection timed out before reaching the API.")
except requests.exceptions.ReadTimeout:
    print("RESULT: API connection opened but the API did not respond within 12 seconds.")
except requests.exceptions.SSLError:
    print("RESULT: TLS/SSL connection error.")
except requests.exceptions.ConnectionError as exc:
    print("RESULT: Network/DNS/connection error:", str(exc)[:300])
except requests.exceptions.RequestException as exc:
    print("RESULT: Request failed:", type(exc).__name__, str(exc)[:300])
