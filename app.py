import json
import os
import pickle
from datetime import date, datetime

import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phone_price_model_final.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")

app = Flask(__name__)

with open(COLUMNS_PATH, "rb") as f:
    MODEL_COLUMNS = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    with xgb.config_context(verbosity=0):
        model = pickle.load(f)

BOOSTER = model.get_booster()

# Integrity check: the inference vector must be built in the exact column order
# the model was trained on. model_columns.pkl was serialized alongside the model
# and must match the booster's stored feature names.
_TRAINED_FEATURES = BOOSTER.feature_names
if list(MODEL_COLUMNS) != list(_TRAINED_FEATURES):
    raise RuntimeError(
        "Feature schema mismatch: model_columns.pkl does not match the trained "
        "model's feature order. Refusing to start with a broken inference schema."
    )

BRANDS = {
    "Apple": "apple", "Samsung": "samsung", "Xiaomi": "xiaomi", "OnePlus": "oneplus",
    "Google": "google", "Motorola": "motorola", "Realme": "realme", "Poco": "poco",
    "Vivo": "vivo", "Oppo": "oppo", "Honor": "honor", "Infinix": "infinix",
    "IQOO": "iqoo", "Nokia": "nokia", "Nothing": "nothing", "Redmi": "red",
    "Tecno": "tecno", "Asus": "asus", "Sony": "sony", "Huawei": "huawei",
    "Lava": "lava", "Itel": "itel", "CMF": "cmf", "HMD": "hmd",
    "Nubia": "nubia", "Realme (Other)": "realme", "Other": None,
}

PROCESSOR_BRANDS = [
    ("Qualcomm Snapdragon", "qualcomm"), ("MediaTek Dimensity", "dimensity"),
    ("MediaTek Helio", "mediatek"), ("Google Tensor", "google"),
    ("Samsung Exynos", "samsung"), ("HiSilicon Kirin", "hisilicon"),
    ("Unisoc", "unisoc"), ("Unknown / Other", "unknown"),
]

OS_OPTIONS = [
    "Android v17", "Android v16", "Android v15", "Android v14", "Android v14.0",
    "Android v13", "Android v12", "Android v11", "Android v10", "Android v10.0",
    "Android v9.0 (Pie)", "HarmonyOS v6.1", "HarmonyOS v6.0", "HarmonyOS v5.1",
    "HarmonyOS v4.2", "iOS v27", "iOS v26.3", "iOS v26", "iOS v18", "iOS v17",
    "iOS v16", "iOS v15", "iOS v14.0", "iOS v13.0",
]

SALE_WINDOWS = [
    {"name": "Independence Day", "month": 8, "day": 15, "discount": 0.05},
    {"name": "Great Indian Festival / Big Billion Days", "month": 9, "day": 25, "discount": 0.09},
    {"name": "Diwali / Festive Week", "month": 10, "day": 20, "discount": 0.11},
    {"name": "Black Friday", "month": 11, "day": 27, "discount": 0.07},
    {"name": "Year-End Clearance", "month": 12, "day": 31, "discount": 0.08},
    {"name": "Republic Day", "month": 1, "day": 26, "discount": 0.08},
]

# Heuristic range applied around the point estimate. This is an approximate
# model range, NOT a statistically computed confidence/prediction interval.
ESTIMATE_RANGE_PCT = 0.08

# Input validation bounds, mirroring the constraints the UI enforces.
# Out-of-range or unparsable values are rejected with a JSON 400 instead of
# silently falling back to a default.
NUM_RANGES = {
    "ram": (1, 32),
    "processor_speed": (0.5, 4.5),
    "memory": (64, 1024),
    "battery": (1000, 10000),
    "fast_charging": (0, 240),
    "rear_camera": (1, 300),
    "front_camera": (1, 100),
    "rear_camera_count": (1, 5),
    "screen_size": (3.5, 8),
    "refresh_rate": (60, 144),
}
VALID_CORES = {4, 6, 8, 10}
VALID_PROCESSOR_BRANDS = {v for _, v in PROCESSOR_BRANDS}
VALID_CHARGING_TYPES = {"Super Fast", "Standard"}


def validate_payload(payload):
    """Reject malformed requests before they reach the model. Raises ValueError
    with a human-readable message; defaults are applied only for missing keys,
    never for invalid ones."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    problems = []

    brand = str(payload.get("brand", "")).strip()
    if brand and brand not in BRANDS:
        problems.append(f"Unsupported brand: {brand}")

    proc = str(payload.get("processor_brand", "")).strip()
    if proc and proc not in VALID_PROCESSOR_BRANDS:
        problems.append(f"Unsupported processor brand: {proc}")

    os_name = str(payload.get("os", "")).strip()
    if os_name and os_name not in OS_OPTIONS:
        problems.append(f"Unsupported OS: {os_name}")

    ctype = str(payload.get("charging_speed_type", "")).strip()
    if ctype and ctype not in VALID_CHARGING_TYPES:
        problems.append(f"Unsupported charging type: {ctype}")

    for key, (lo, hi) in NUM_RANGES.items():
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{key} must be a number (got {raw!r}).")
            continue
        if not (lo <= val <= hi):
            problems.append(f"{key} must be between {lo:g} and {hi:g} (got {val:g}).")

    cores = payload.get("cores")
    if cores not in (None, ""):
        try:
            if int(cores) not in VALID_CORES:
                problems.append("cores must be one of 4, 6, 8 or 10.")
        except (TypeError, ValueError):
            problems.append("cores must be an integer.")

    if problems:
        raise ValueError(" ".join(problems))


def _num(d, key, default):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _bool(d, key):
    v = d.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)


def parse_json_body():
    """Strict JSON body parser: rejects malformed bodies with a 400 instead of
    silently substituting defaults."""
    data = request.get_json(silent=True)
    if data is None and request.data:
        raise ValueError("Request body must be valid JSON.")
    return data or {}


def _col_idx(name):
    return MODEL_COLUMNS.index(name)


def build_feature_vector(payload):
    row = np.zeros(len(MODEL_COLUMNS), dtype=float)

    brand = str(payload.get("brand", "Samsung")).strip()
    brand_col = BRANDS.get(brand)
    if brand_col is not None:
        row[_col_idx(f"brand_name_{brand_col}")] = 1.0
        if brand_col == "samsung":
            row[_col_idx("brand_name_samaung")] = 1.0

    proc = str(payload.get("processor_brand", "qualcomm")).strip()
    row[_col_idx(f"processor_brand_{proc}")] = 1.0

    os_name = str(payload.get("os", "Android v15")).strip()
    if f"os_{os_name}" in MODEL_COLUMNS:
        row[_col_idx(f"os_{os_name}")] = 1.0

    charge_type = str(payload.get("charging_speed_type", "Super Fast")).strip()
    row[_col_idx(f"charging_speed_type_{charge_type}")] = 1.0

    battery = _num(payload, "battery", 5000)
    fast_charge = _num(payload, "fast_charging", 45)

    row[_col_idx("has_5G")] = 1.0 if _bool(payload, "has_5g") else 0.0
    row[_col_idx("has_NFC")] = 1.0 if _bool(payload, "has_nfc") else 0.0
    row[_col_idx("has_IR")] = 1.0 if _bool(payload, "has_ir") else 0.0
    row[_col_idx("num_core")] = _num(payload, "cores", 8)
    row[_col_idx("processor_speed")] = _num(payload, "processor_speed", 3.2)
    row[_col_idx("ram")] = _num(payload, "ram", 12)
    row[_col_idx("memory")] = _num(payload, "memory", 256)
    row[_col_idx("battery_capacity(mAh)")] = battery
    row[_col_idx("fast_charging(W)")] = fast_charge
    row[_col_idx("charging_ratio")] = fast_charge / battery if battery else 0.0
    row[_col_idx("screen_size")] = _num(payload, "screen_size", 6.7)
    row[_col_idx("refresh_rate")] = _num(payload, "refresh_rate", 120)
    row[_col_idx("rear_camera")] = _num(payload, "rear_camera", 50)
    row[_col_idx("front_camera")] = _num(payload, "front_camera", 12)
    row[_col_idx("rear_camera_count")] = _num(payload, "rear_camera_count", 3)
    row[_col_idx("processor_name_freq")] = _num(payload, "processor_name_freq", 1.0)

    return row


def predict_row(row):
    log_price = float(model.predict([row])[0])
    price = float(np.exp(log_price))
    return price


# Logical groupings of model columns, used both for per-prediction SHAP-style
# contribution summaries and for global gain aggregation on the insights page.
GROUPS = {
    "Processor": [f"processor_{p}" for p in ("speed", "name", "brand", "core")]
    + ["num_core", "processor_name_freq", "processor_speed"],
    "RAM": ["ram"],
    "Brand": [c for c in MODEL_COLUMNS if c.startswith("brand_name_")],
    "Storage": ["memory"],
    "Camera": ["rear_camera", "front_camera", "rear_camera_count"],
    "Battery & Charging": ["battery_capacity(mAh)", "fast_charging(W)", "charging_ratio"]
    + [c for c in MODEL_COLUMNS if c.startswith("charging_speed_type_")],
    "Display": ["screen_size", "refresh_rate"],
    "Connectivity": ["has_5G", "has_NFC", "has_IR"],
    "OS": [c for c in MODEL_COLUMNS if c.startswith("os_")],
}


def feature_contributions(row):
    dmat = xgb.DMatrix(row.reshape(1, -1), feature_names=MODEL_COLUMNS)
    contribs = BOOSTER.predict(dmat, pred_contribs=True)[0][:-1]

    weights = []
    for label, cols in GROUPS.items():
        idx = [MODEL_COLUMNS.index(c) for c in cols if c in MODEL_COLUMNS]
        weights.append((label, float(np.abs(contribs[idx]).sum())))
    total = sum(w for _, w in weights) or 1.0
    return [{"label": label, "weight": w, "pct": round(100 * w / total)} for label, w in weights]


def inr(n):
    return "₹{:,.0f}".format(n)


def format_indian(n):
    s = "{:,.0f}".format(n)
    parts = s.split(".")
    before, after = parts[0], ("." + parts[1]) if len(parts) > 1 else ""
    sign = ""
    if before.startswith("-"):
        sign, before = "-", before[1:]
    if len(before) > 3:
        head, tail = before[:-3], before[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        before = ",".join(groups) + "," + tail
    return sign + before + after


def major_sale_windows(today=None):
    today = today or date.today()
    windows = []
    for w in SALE_WINDOWS:
        year = today.year
        wd = date(year, w["month"], w["day"])
        if wd < today:
            wd = date(year + 1, w["month"], w["day"])
        windows.append({**w, "date": wd})
    windows.sort(key=lambda x: x["date"])
    return windows


def advise(payload):
    row = build_feature_vector(payload)
    price = predict_row(row)

    market = _num(payload, "market_price", 0)
    if not market:
        raise ValueError("market_price is required: the advisor compares the current market price against the specification-based estimate.")

    variance = (market - price) / price * 100.0

    if variance <= -2.0:
        verdict = {"label": "BUY NOW", "tone": "green", "emoji": "shopping_bag"}
        reason = (f"The current market price is {abs(variance):.1f}% below the specification-based "
                  f"estimate of ₹{price:,.0f}. The hardware is priced below its typical specification "
                  "value — waiting for a future sale is unlikely to save significantly more.")
    elif variance <= 5.0:
        verdict = {"label": "GOOD TIME", "tone": "amber", "emoji": "thumb_up"}
        reason = (f"The current price is {abs(variance):.1f}% away from the specification-based estimate "
                  f"of ₹{price:,.0f}. That is a fair specification-to-price ratio; buy when you find this price or better.")
    else:
        verdict = {"label": "WAIT", "tone": "red", "emoji": "schedule"}
        reason = (f"The current market price is {variance:.1f}% above the specification-based estimate of "
                  f"₹{price:,.0f}. The hardware is priced above its typical specification value — "
                  "a seasonal sale period may bring it closer.")

    # Specification-only valuation may diverge a lot from retail positioning.
    gap_warning = None
    if abs(variance) > 25.0:
        gap_warning = ("Large model/market gap — specification-only estimate may not fully "
                       "reflect this model's current retail positioning (brand pricing, launch age, discounts).")

    windows = major_sale_windows()
    timeline = []
    now_node = {"key": "now", "name": "NOW", "date": date.today().isoformat(),
                "price": market, "target": False, "is_now": True}
    for w in windows:
        est = price * (1 - w["discount"])
        timeline.append({
            "key": w["name"].lower().replace(" ", "-"),
            "name": w["name"],
            "date": w["date"].isoformat(),
            "price": est,
            "target": False,
            "is_now": False,
        })
    best = min(timeline, key=lambda t: t["price"])
    best["target"] = True
    timeline.insert(0, now_node)

    weeks_until = max(0, (date.fromisoformat(best["date"]) - date.today()).days // 7)

    tier = "Budget" if price <= 25000 else ("Mid-Range" if price <= 60000 else "Premium")
    return {
        "price": price,
        "market_price": market,
        "variance_pct": round(variance, 1),
        "verdict": verdict,
        "reason": reason,
        "gap_warning": gap_warning,
        "tier": tier,
        "timeline": timeline,
        "target_price": best["price"],
        "best_window": best["name"],
        "weeks_until": weeks_until,
        "contributions": feature_contributions(row),
        "brand": str(payload.get("brand", "Phone")),
        "ram": _num(payload, "ram", 0),
        "memory": _num(payload, "memory", 0),
        "processor_speed": _num(payload, "processor_speed", 0),
        "cores": _num(payload, "cores", 0),
        "rear_camera": _num(payload, "rear_camera", 0),
        "battery": _num(payload, "battery", 0),
        "screen_size": _num(payload, "screen_size", 0),
        "refresh_rate": _num(payload, "refresh_rate", 0),
    }


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict")


@app.route("/advisor")
def advisor_page():
    return render_template("advisor.html", active="advisor")


@app.route("/compare")
def compare_page():
    return render_template("compare.html", active="compare")


@app.route("/insights")
def insights_page():
    return render_template("insights.html", active="insights")


@app.route("/methodology")
def methodology_page():
    return render_template("methodology.html", active="methodology")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = parse_json_body()
        validate_payload(payload)
        row = build_feature_vector(payload)
        price = predict_row(row)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed: " + str(e)}), 500
    lo = price * (1 - ESTIMATE_RANGE_PCT)
    hi = price * (1 + ESTIMATE_RANGE_PCT)
    return jsonify({
        "price": price,
        "price_low": lo,
        "price_high": hi,
        "range_pct": ESTIMATE_RANGE_PCT * 100,
        "contributions": feature_contributions(row),
    })


@app.route("/api/advise", methods=["POST"])
def api_advise():
    try:
        payload = parse_json_body()
        validate_payload(payload)
        result = advise(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Advice failed")
        return jsonify({"error": "Advice failed: " + str(e)}), 500
    return jsonify(result)


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "model": "XGBoost Regressor",
        "num_features": len(MODEL_COLUMNS),
    })


@app.route("/api/insights")
def api_insights():
    score = BOOSTER.get_score(importance_type="gain")
    # Include every trained column: features that never split any tree have
    # zero gain and are listed with 0% rather than being silently dropped.
    ranked = sorted(
        ((name, score.get(name, 0.0)) for name in MODEL_COLUMNS),
        key=lambda kv: -kv[1],
    )
    total = sum(v for _, v in ranked)

    groups = []
    for label, cols in GROUPS.items():
        g = sum(v for k, v in ranked if k in cols)
        groups.append({"label": label, "gain": round(g, 1), "pct": round(100 * g / total)})
    groups.sort(key=lambda g: -g["gain"])

    return jsonify({
        "model": "XGBoost Regressor",
        "trees": int(model.n_estimators),
        "max_depth": int(model.max_depth),
        "learning_rate": float(model.learning_rate),
        "num_features": len(MODEL_COLUMNS),
        "target": "log(price) — exp-transformed to INR",
        "features": [
            {"name": k, "gain": round(v, 1), "pct": round(100 * v / total)}
            for k, v in ranked[:12]
        ],
        "all_features": [
            {"name": k, "gain": round(v, 1), "pct": round(100 * v / total)}
            for k, v in ranked
        ],
        "groups": groups,
    })


COMPARE_PRESETS = [
    {"name": "Samsung Galaxy S25 Ultra", "brand": "Samsung", "ram": 12, "memory": 256,
     "processor_speed": 3.4, "cores": 8, "battery": 5000, "fast_charging": 45,
     "rear_camera": 200, "front_camera": 12, "screen_size": 6.7, "refresh_rate": 120,
     "processor_brand": "qualcomm", "os": "Android v15", "charging_speed_type": "Super Fast",
     "has_5g": True, "has_nfc": True, "market_flipkart": 109999, "market_amazon": 110500, "tag": "5G"},
    {"name": "Apple iPhone 16 Pro", "brand": "Apple", "ram": 8, "memory": 256,
     "processor_speed": 3.8, "cores": 6, "battery": 3582, "fast_charging": 27,
     "rear_camera": 48, "front_camera": 12, "screen_size": 6.3, "refresh_rate": 120,
     "processor_brand": "unknown", "os": "iOS v18", "charging_speed_type": "Super Fast",
     "has_5g": True, "has_nfc": True, "market_flipkart": 124900, "market_amazon": 121900, "tag": "A18 Pro"},
    {"name": "Google Pixel 10", "brand": "Google", "ram": 12, "memory": 128,
     "processor_speed": 3.1, "cores": 8, "battery": 4050, "fast_charging": 30,
     "rear_camera": 48, "front_camera": 10, "screen_size": 6.1, "refresh_rate": 120,
     "processor_brand": "google", "os": "Android v16", "charging_speed_type": "Standard",
     "has_5g": True, "has_nfc": True, "market_flipkart": 71499, "market_amazon": 71999, "tag": "AI"},
    {"name": "OnePlus 13", "brand": "OnePlus", "ram": 12, "memory": 256,
     "processor_speed": 3.3, "cores": 8, "battery": 6000, "fast_charging": 100,
     "rear_camera": 50, "front_camera": 32, "screen_size": 6.82, "refresh_rate": 120,
     "processor_brand": "qualcomm", "os": "Android v15", "charging_speed_type": "Super Fast",
     "has_5g": True, "has_nfc": True, "market_flipkart": 59999, "market_amazon": 59999, "tag": "FastCharge"},
    {"name": "Xiaomi Redmi Note 14 Pro", "brand": "Xiaomi", "ram": 8, "memory": 128,
     "processor_speed": 2.5, "cores": 8, "battery": 5500, "fast_charging": 45,
     "rear_camera": 200, "front_camera": 20, "screen_size": 6.67, "refresh_rate": 120,
     "processor_brand": "mediatek", "os": "Android v14", "charging_speed_type": "Super Fast",
     "has_5g": True, "has_nfc": True, "market_flipkart": 24999, "market_amazon": 25499, "tag": "5G"},
    {"name": "Poco X6 Pro", "brand": "Poco", "ram": 8, "memory": 256,
     "processor_speed": 2.8, "cores": 8, "battery": 5000, "fast_charging": 67,
     "rear_camera": 64, "front_camera": 16, "screen_size": 6.67, "refresh_rate": 120,
     "processor_brand": "mediatek", "os": "Android v14", "charging_speed_type": "Super Fast",
     "has_5g": True, "has_nfc": True, "market_flipkart": 27999, "market_amazon": 27999, "tag": "5G"},
]


@app.route("/api/compare")
def api_compare():
    rows = []
    for p in COMPARE_PRESETS:
        price = predict_row(build_feature_vector(p))
        amazon = p["market_amazon"]
        flipkart = p["market_flipkart"]
        if amazon <= flipkart:
            best = "Amazon"
            best_price = amazon
        else:
            best = "Flipkart"
            best_price = flipkart
        diff = (best_price - price) / price * 100
        rows.append({**p, "predicted": price, "amazon": amazon, "flipkart": flipkart,
                     "best": best, "best_price": best_price, "diff_pct": round(diff, 1)})
    return jsonify({"devices": rows, "reference_prices": True})


@app.after_request
def apply_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        code=404,
        title="Page not found",
        message="The page you are looking for does not exist or has been moved.",
    ), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Unhandled error")
    return render_template(
        "error.html",
        code=500,
        title="Something went wrong",
        message="An unexpected error occurred while processing your request. Please try again.",
    ), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)