from datetime import date
from pathlib import Path
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.live_data import fetch_live_stations

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "aqi_model.joblib"

app = FastAPI(title="AirSight AQI Prediction API",
              description="Air-quality estimates and next-day forecasts from Open-Meteo/CAMS.",
              version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

saved = None
model = None
feature_columns = []
pollutant_columns = []
if MODEL_FILE.exists():
    saved = joblib.load(MODEL_FILE)
    model = saved.get("model")
    feature_columns = saved.get("feature_columns", [])
    pollutant_columns = saved.get("pollutant_columns", [])

class AQIRequest(BaseModel):
    city: str = Field(..., min_length=1)
    month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=0, le=6)
    pm25: float | None = Field(None, ge=0)
    pm10: float | None = Field(None, ge=0)
    no: float | None = Field(None, ge=0)
    no2: float | None = Field(None, ge=0)
    nox: float | None = Field(None, ge=0)
    nh3: float | None = Field(None, ge=0)
    co: float | None = Field(None, ge=0)
    so2: float | None = Field(None, ge=0)
    o3: float | None = Field(None, ge=0)
    benzene: float | None = Field(None, ge=0)
    toluene: float | None = Field(None, ge=0)
    xylene: float | None = Field(None, ge=0)

POLLUTANT_INPUT_MAP = {
    "PM2.5":"pm25","PM10":"pm10","NO":"no","NO2":"no2","NOx":"nox",
    "NH3":"nh3","CO":"co","SO2":"so2","O3":"o3","Benzene":"benzene",
    "Toluene":"toluene","Xylene":"xylene"
}

def _predict_from_values(values):
    if model is None:
        raise ValueError("Trained model file is missing.")
    row = pd.DataFrame([{column: values.get(column) for column in feature_columns}])
    prediction = float(model.predict(row)[0])
    if not math.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction.")
    return round(max(0.0, prediction), 2)

@app.get("/")
def home():
    return {"project":"AirSight","status":"API running",
            "data_source":"Open-Meteo/CAMS model grid, not direct monitoring stations",
            "endpoints":{"health":"/health","live_stations":"/live-stations",
                         "live_prediction":"/predict-live?city=Delhi","documentation":"/docs"}}

@app.get("/health")
def health():
    return {"status":"healthy","model_loaded":model is not None,
            "live_data_source":"Open-Meteo/CAMS model grid"}

@app.get("/live-stations")
def live_stations(city: str | None = Query(default=None, min_length=1),
                  limit: int = Query(default=300, ge=1, le=1000)):
    try:
        return {"status":"success", **fetch_live_stations(city.strip() if city else None, limit),
                "note":"Model-grid estimates/forecasts, not direct CPCB station measurements."}
    except Exception as exc:
        print("Open-Meteo request failed:", type(exc).__name__, str(exc)[:200])
        raise HTTPException(status_code=502,
            detail=f"Open-Meteo air-quality request failed: {type(exc).__name__}. Check network/API availability.") from exc

@app.get("/predict-live")
def predict_live(city: str = Query(..., min_length=1),
                 limit: int = Query(default=300, ge=1, le=1000)):
    try:
        result = fetch_live_stations(city=city.strip(), limit=limit)
    except Exception as exc:
        print("Open-Meteo prediction request failed:", type(exc).__name__, str(exc)[:200])
        raise HTTPException(status_code=502, detail=f"Open-Meteo request failed: {type(exc).__name__}.") from exc
    stations = result.get("stations", [])
    if not stations:
        raise HTTPException(status_code=404, detail=f"No configured city found for '{city}'.")
    station = stations[0]
    forecast = station.get("forecast_aqi")
    if forecast is None:
        raise HTTPException(status_code=502, detail="Open-Meteo did not return a next-day AQI value for this location.")
    pollutants = station.get("pollutants", {})
    return {"city":station["city"],"station":station["station"],
            "last_update":station.get("last_update"),
            "predicted_next_day_aqi":round(float(forecast), 1),
            "unit":"US AQI points","model":"Open-Meteo/CAMS forecast (not AirSight Random Forest)",
            "pollutants_used":[k for k in pollutants if k != "US AQI"],
            "pollutant_count":sum(1 for k in pollutants if k != "US AQI"),
            "current_aqi":pollutants.get("US AQI"),
            "current_aqi_status":"Model-estimated US AQI from Open-Meteo; not a CPCB station reading.",
            "note":"Forecast is from Open-Meteo/CAMS. This is not an official CPCB AQI warning."}

@app.post("/predict")
def predict(request: AQIRequest):
    values = {"City":request.city.strip(),"Month":request.month,"DayOfWeek":request.day_of_week}
    for ds_name, req_name in POLLUTANT_INPUT_MAP.items():
        values[ds_name] = getattr(request, req_name)
    try:
        prediction = _predict_from_values(values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return {"city":request.city.strip(),"predicted_next_day_aqi":prediction,
            "unit":"AQI points","model":saved.get("selected_model","Saved model") if saved else "Saved model",
            "note":"This is the trained AirSight model output; ensure inputs match its training units."}
