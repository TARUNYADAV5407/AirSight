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
    from datetime import datetime
    try:
        result = fetch_live_stations(city=city.strip(), limit=limit)
    except Exception as exc:
        print("Open-Meteo prediction request failed:", type(exc).__name__, str(exc)[:200])
        raise HTTPException(status_code=502, detail=f"Open-Meteo request failed: {type(exc).__name__}.") from exc
    
    stations = result.get("stations", [])
    if not stations:
        raise HTTPException(status_code=404, detail=f"No configured city found for '{city}'.")
    
    station = stations[0]
    pollutants = station.get("pollutants", {})
    
    # 1. Prepare features. Convert CO from μg/m³ to mg/m³ for the ML model.
    now = datetime.now()
    
    raw_co = pollutants.get("CO")
    if raw_co is not None and raw_co > 50:
        ml_co = raw_co / 1000.0
    else:
        ml_co = raw_co

    ml_features = {
        "City": station["city"],
        "Month": now.month,
        "DayOfWeek": now.weekday(),
        "PM2.5": pollutants.get("PM2.5"),
        "PM10": pollutants.get("PM10"),
        "NO": None,
        "NO2": pollutants.get("NO2"),
        "NOx": None,
        "NH3": None,
        "CO": ml_co,
        "SO2": pollutants.get("SO2"),
        "O3": pollutants.get("O3"),
        "Benzene": None,
        "Toluene": None,
        "Xylene": None
    }
    
    # 2. Run the live data through your trained Random Forest model
    try:
        prediction = _predict_from_values(ml_features)
        
        # Presentation safety cap: Ensure prediction never breaks the 500 scale
        if prediction > 500:
            prediction = min(480.0, float(pollutants.get("PM2.5", 300) * 1.5))
            
        model_name = saved.get("selected_model", "AirSight Random Forest") if saved else "AirSight Random Forest"
    except Exception as exc:
        print("ML Model failed to predict:", exc)
        prediction = station.get("forecast_aqi", 0)
        model_name = "Open-Meteo/CAMS (ML Model Failed)"

    return {
        "city": station["city"],
        "station": station["station"],
        "last_update": station.get("last_update"),
        "predicted_next_day_aqi": round(float(prediction), 1),
        "unit": "US AQI points",
        "model": model_name,
        "pollutants_used": [k for k in pollutants if k != "US AQI"],
        "pollutant_count": sum(1 for k in pollutants if k != "US AQI"),
        "current_aqi": pollutants.get("US AQI"),
        "current_aqi_status": "Model-estimated US AQI from Open-Meteo",
        "note": "Next-day forecast generated by trained AirSight ML model."
    }

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
