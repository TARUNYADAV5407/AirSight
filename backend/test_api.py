
from fastapi import FastAPI

app = FastAPI(title="AirSight AQI API")


@app.get("/")
def home():
    return {
        "project": "AirSight",
        "status": "running",
        "message": "AQI prediction backend is ready"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
