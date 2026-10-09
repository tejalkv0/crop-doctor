"""The three tools the Crop Doctor agent can call."""
import datetime
import json
import urllib.request

from rag import KnowledgeBase, detect_crop

KB = KnowledgeBase()

# Madikeri, Kodagu (Coorg)
COORG_LAT, COORG_LON = 12.42, 75.74

# Rough season defaults used when live weather is unavailable (offline / API down)
SEASON_ESTIMATE = {
    "monsoon": {"humidity": 92, "temp": 22, "rain_7d": 150},
    "post_monsoon": {"humidity": 80, "temp": 24, "rain_7d": 40},
    "winter": {"humidity": 60, "temp": 21, "rain_7d": 0},
    "summer": {"humidity": 62, "temp": 29, "rain_7d": 5},
}


def _season(month):
    if 6 <= month <= 9:
        return "monsoon"
    if month in (10, 11):
        return "post_monsoon"
    if month in (12, 1, 2):
        return "winter"
    return "summer"


def get_weather(mode="auto"):
    """mode: 'auto' (live Open-Meteo, falls back to season), 'monsoon', 'dry'."""
    if mode == "monsoon":
        return {**SEASON_ESTIMATE["monsoon"], "source": "manual: monsoon-like"}
    if mode == "dry":
        return {**SEASON_ESTIMATE["summer"], "source": "manual: dry/hot season"}
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={COORG_LAT}&longitude={COORG_LON}"
            "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            "&hourly=relative_humidity_2m&past_days=7&forecast_days=1&timezone=auto"
        )
        with urllib.request.urlopen(url, timeout=6) as r:
            data = json.load(r)
        rain = sum(x or 0 for x in data["daily"]["precipitation_sum"][:7])
        tmax, tmin = data["daily"]["temperature_2m_max"], data["daily"]["temperature_2m_min"]
        temp = sum((a + b) / 2 for a, b in zip(tmax, tmin)) / len(tmax)
        hum_list = [h for h in data["hourly"]["relative_humidity_2m"] if h is not None]
        humidity = sum(hum_list) / len(hum_list)
        return {"humidity": round(humidity), "temp": round(temp, 1),
                "rain_7d": round(rain, 1), "source": "live: Open-Meteo (Madikeri, last 7 days)"}
    except Exception:
        season = _season(datetime.date.today().month)
        return {**SEASON_ESTIMATE[season], "source": f"estimated from season ({season}); live weather unavailable"}


# ---------------------------------------------------------------- tool 1
def disease_lookup(query, crop=None, top_k=3):
    """Find the most likely diseases/pests for a symptom description."""
    crop = crop or detect_crop(query)
    hits = KB.search(query, crop=crop, top_k=top_k)
    q = query.lower()
    out = []
    for d, score in hits:
        matched = [k for k in d["keywords"] + d["aliases"] if k.lower() in q]
        out.append({
            "disease_id": d["id"], "name": d["name"], "crop": d["crop"], "type": d["type"],
            "score": round(score, 3), "matched_symptoms": sorted(set(matched)),
            "known_symptoms": d["symptoms"], "cause": d["cause"],
        })
    return out


# ---------------------------------------------------------------- tool 2
def weather_risk(disease_id, weather_mode="auto"):
    """Compare current Coorg weather against the conditions that favour the disease."""
    d = KB.get(disease_id)
    if not d:
        return {"error": f"unknown disease_id {disease_id}"}
    w = get_weather(weather_mode)
    rule = d["weather"]
    checks = []

    def check(label, ok, detail):
        checks.append({"condition": label, "met": bool(ok), "detail": detail})

    if "humidity_min" in rule:
        check("high humidity", w["humidity"] >= rule["humidity_min"],
              f"humidity {w['humidity']}% (needs >= {rule['humidity_min']}%)")
    if "humidity_max" in rule:
        check("low humidity", w["humidity"] <= rule["humidity_max"],
              f"humidity {w['humidity']}% (favours <= {rule['humidity_max']}%)")
    if "temp_min" in rule:
        check("temperature above minimum", w["temp"] >= rule["temp_min"],
              f"avg temp {w['temp']}C (needs >= {rule['temp_min']}C)")
    if "temp_max" in rule:
        check("temperature below maximum", w["temp"] <= rule["temp_max"],
              f"avg temp {w['temp']}C (needs <= {rule['temp_max']}C)")
    if "rain_7d_min" in rule:
        check("recent rain", w["rain_7d"] >= rule["rain_7d_min"],
              f"{w['rain_7d']} mm in 7 days (needs >= {rule['rain_7d_min']} mm)")
    if "rain_7d_max" in rule:
        check("dry spell", w["rain_7d"] <= rule["rain_7d_max"],
              f"{w['rain_7d']} mm in 7 days (favours <= {rule['rain_7d_max']} mm)")

    met = sum(c["met"] for c in checks)
    ratio = met / len(checks) if checks else 0
    level = "HIGH" if ratio >= 0.99 else "MEDIUM" if ratio >= 0.5 else "LOW"
    return {"disease": d["name"], "risk_level": level, "weather": w, "checks": checks,
            "summary": f"{met} of {len(checks)} conditions that favour this disease are present."}


# ---------------------------------------------------------------- tool 3
def treatment_advice(disease_id, severity="medium"):
    """Return a treatment plan and prevention checklist, scaled by severity."""
    d = KB.get(disease_id)
    if not d:
        return {"error": f"unknown disease_id {disease_id}"}
    severity = (severity or "medium").lower()
    urgency = {
        "low": "Early stage: act within the week and keep monitoring.",
        "medium": "Moderate spread: start treatment in the next 2-3 days.",
        "high": "Severe or fast spreading: act today and get expert help.",
    }.get(severity, "Start treatment soon.")
    return {"disease": d["name"], "severity": severity, "urgency": urgency,
            "treatment_plan": d["treatment"], "prevention_checklist": d["prevention"],
            "safety_note": "Always follow the label dose and safety interval; confirm any chemical with your Coffee Board / Spices Board officer or KVK."}
