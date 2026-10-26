
"""AirSight live air-quality data using Open-Meteo."""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import requests

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# City, state, latitude, longitude
CITIES = [
    ("Delhi", "Delhi", 28.6139, 77.2090),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Faridabad", "Haryana", 28.4089, 77.3178),
    ("Gurugram", "Haryana", 28.4595, 77.0266),
    ("Chandigarh", "Chandigarh", 30.7333, 76.7794),
    ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319),
    ("Agra", "Uttar Pradesh", 27.1767, 78.0081),
    ("Varanasi", "Uttar Pradesh", 25.3176, 82.9739),
    ("Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Patna", "Bihar", 25.5941, 85.1376),
    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Indore", "Madhya Pradesh", 22.7196, 75.8573),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Surat", "Gujarat", 21.1702, 72.8311),
    ("Pune", "Maharashtra", 18.5204, 73.8567),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Bengaluru", "Karnataka", 12.9716, 77.5946),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185),
    ("Bhubaneswar", "Odisha", 20.2961, 85.8245),
    ("Guwahati", "Assam", 26.1445, 91.7362),
    ("Ranchi", "Jharkhand", 23.3441, 85.3096),
    ("Raipur", "Chhattisgarh", 21.2514, 81.6296),
    ("Amritsar", "Punjab", 31.6340, 74.8723),
    ("Ludhiana", "Punjab", 30.9000, 75.8573),
    ("Jodhpur", "Rajasthan", 26.2389, 73.0243),
    ("Nagpur", "Maharashtra", 21.1458, 79.0882),
    ("Noida", "Uttar Pradesh", 28.5355, 77.3910),
    ("Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538),
    ("Meerut", "Uttar Pradesh", 28.9845, 77.7064),
    ("Prayagraj", "Uttar Pradesh", 25.4358, 81.8463),
    ("Gorakhpur", "Uttar Pradesh", 26.7606, 83.3732),
    ("Bareilly", "Uttar Pradesh", 28.3670, 79.4304),
    ("Moradabad", "Uttar Pradesh", 28.8386, 78.7733),
    ("Mathura", "Uttar Pradesh", 27.4924, 77.6737),
    ("Panipat", "Haryana", 29.3909, 76.9635),
    ("Karnal", "Haryana", 29.6857, 76.9905),
    ("Hisar", "Haryana", 29.1492, 75.7217),
    ("Rohtak", "Haryana", 28.8955, 76.6066),
    ("Sonipat", "Haryana", 28.9931, 77.0151),
    ("Ambala", "Haryana", 30.3782, 76.7767),
    ("Rewari", "Haryana", 28.1990, 76.6190),
    ("Kota", "Rajasthan", 25.2138, 75.8648),
    ("Udaipur", "Rajasthan", 24.5854, 73.7125),
    ("Ajmer", "Rajasthan", 26.4499, 74.6399),
    ("Bikaner", "Rajasthan", 28.0229, 73.3119),
    ("Alwar", "Rajasthan", 27.5530, 76.6346),
    ("Vadodara", "Gujarat", 22.3072, 73.1812),
    ("Rajkot", "Gujarat", 22.3039, 70.8022),
    ("Jamnagar", "Gujarat", 22.4707, 70.0577),
    ("Nashik", "Maharashtra", 19.9975, 73.7898),
    ("Kolhapur", "Maharashtra", 16.7050, 74.2433),
    ("Solapur", "Maharashtra", 17.6599, 75.9064),
    ("Thane", "Maharashtra", 19.2183, 72.9781),
    ("Mysuru", "Karnataka", 12.2958, 76.6394),
    ("Mangaluru", "Karnataka", 12.9141, 74.8560),
    ("Hubballi", "Karnataka", 15.3647, 75.1240),
    ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558),
    ("Madurai", "Tamil Nadu", 9.9252, 78.1198),
    ("Salem", "Tamil Nadu", 11.6643, 78.1460),
    ("Kochi", "Kerala", 9.9312, 76.2673),
    ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    ("Kozhikode", "Kerala", 11.2588, 75.7804),
    ("Vijayawada", "Andhra Pradesh", 16.5062, 80.6480),
    ("Guntur", "Andhra Pradesh", 16.3067, 80.4365),
    ("Tirupati", "Andhra Pradesh", 13.6288, 79.4192),
    ("Warangal", "Telangana", 17.9689, 79.5941),
    ("Dhanbad", "Jharkhand", 23.7957, 86.4304),
    ("Jamshedpur", "Jharkhand", 22.8046, 86.2029),
    ("Gaya", "Bihar", 24.7914, 85.0002),
    ("Muzaffarpur", "Bihar", 26.1209, 85.3647),
    ("Siliguri", "West Bengal", 26.7271, 88.3953),
    ("Durgapur", "West Bengal", 23.5204, 87.3119),
    ("Cuttack", "Odisha", 20.4625, 85.8828),
    ("Rourkela", "Odisha", 22.2604, 84.8536),
    ("Jabalpur", "Madhya Pradesh", 23.1815, 79.9864),
    ("Gwalior", "Madhya Pradesh", 26.2183, 78.1828),
    ("Ujjain", "Madhya Pradesh", 23.1765, 75.7885),
    ("Shimla", "Himachal Pradesh", 31.1048, 77.1734),
    ("Jammu", "Jammu and Kashmir", 32.7266, 74.8570),
    ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
    ("Imphal", "Manipur", 24.8170, 93.9368),
    ("Shillong", "Meghalaya", 25.5788, 91.8933),
    ("Aizawl", "Mizoram", 23.7271, 92.7176),
    ("Kohima", "Nagaland", 25.6751, 94.1086),
    ("Agartala", "Tripura", 23.8315, 91.2868),
    ("Itanagar", "Arunachal Pradesh", 27.0844, 93.6053),
    ("Gangtok", "Sikkim", 27.3389, 88.6065),
    ("Port Blair", "Andaman and Nicobar Islands", 11.6234, 92.7265),
    ("Puducherry", "Puducherry", 11.9416, 79.8083),
    ("Panaji", "Goa", 15.4909, 73.8278),
]

POLLUTANTS = {
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "CO": "carbon_monoxide",
    "NO2": "nitrogen_dioxide",
    "SO2": "sulphur_dioxide",
    "O3": "ozone",
    "US AQI": "us_aqi",
}

_cache = {}
_cache_lock = threading.Lock()
CACHE_SECONDS = 300


def _fetch_batch(cities):
    """Fetch up to 40 locations in one request."""
    params = {
        "latitude": ",".join(str(c[2]) for c in cities),
        "longitude": ",".join(str(c[3]) for c in cities),
        "current": ",".join(POLLUTANTS.values()),
        "hourly": ",".join(POLLUTANTS.values()),
        "forecast_days": 2,
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(AIR_URL, params=params, timeout=(5, 20))
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else [data]


def _num(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _at(values, index):
    try:
        return _num(values[index])
    except (IndexError, TypeError):
        return None


def fetch_live_stations(city=None, limit=300):
    """Compatible with the existing AirSight backend function."""
    cache_key = (city.strip().casefold() if city else "*", int(limit))
    now = time.time()

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]

    selected = CITIES
    if city:
        query = city.strip().casefold()
        selected = [
            item for item in CITIES
            if item[0].casefold() == query
            or query in item[0].casefold()
            or query == item[1].casefold()
        ]

    selected = selected[:max(1, min(int(limit), len(CITIES)))]

    if not selected:
        return {
            "dataset": "Open-Meteo/CAMS model-grid estimates",
            "total_records": 0,
            "records_received": 0,
            "stations_found": 0,
            "stations": [],
        }

    # Keep requests manageable; batches are fetched concurrently.
    batches = [selected[i:i + 40] for i in range(0, len(selected), 40)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results_by_batch = list(pool.map(_fetch_batch, batches))

    stations = []
    for batch, results in zip(batches, results_by_batch):
        for index, item in enumerate(batch):
            data = results[index] if index < len(results) else {}
            current = data.get("current") or {}
            hourly = data.get("hourly") or {}
            pollutants = {}

            for label, key in POLLUTANTS.items():
                value = _num(current.get(key))
                if value is None:
                    value = _at(hourly.get(key, []), 0)
                if value is not None:
                    pollutants[label] = value

            # First available US AQI forecast value dated after today's date.
            times = hourly.get("time", [])
            aqi_values = hourly.get("us_aqi", [])
            tomorrow_aqi = None
            today = datetime.now().date().isoformat()
            for timestamp, value in zip(times, aqi_values):
                if timestamp[:10] > today and value is not None:
                    tomorrow_aqi = float(value)
                    break

            name, state, lat, lon = item
            stations.append({
                "city": name,
                "state": state,
                "country": "India",
                "station": f"{name} (Open-Meteo model grid)",
                "last_update": current.get(
                    "time", datetime.now().strftime("%Y-%m-%d %H:%M")
                ),
                "latitude": lat,
                "longitude": lon,
                "pollutants": pollutants,
                "forecast_aqi": tomorrow_aqi,
                "data_source": "Open-Meteo/CAMS model grid",
            })

    result = {
        "dataset": "Open-Meteo/CAMS model-grid estimates; not direct CPCB readings",
        "total_records": len(stations),
        "records_received": len(stations),
        "stations_found": len(stations),
        "stations": stations,
    }

    with _cache_lock:
        _cache[cache_key] = (time.time(), result)

    return result
