import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mpogidi Price Monitor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stSidebar"] { background: #f8f7f4; }
h1 { font-weight: 600; letter-spacing: -0.5px; }
.stAlert { border-radius: 8px; }
.pred-box {
    border-radius: 10px; padding: 1.2rem 1.5rem;
    margin-top: 0.5rem; font-size: 1rem;
    color: #111111 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Commodity labels ──────────────────────────────────────────────────────────
LABELS_RAW = [
    ("average_white_maize", 1, "Maize (White)"),
    ("average_Maize_Yellow", 2, "Maize (Yellow)"),
    ("average_Millet", 3, "Millet"),
    ("average_Sorghum", 4, "Sorghum"),
    ("average_Rice_Local_perfumed", 5, "Rice Local (perfumed)"),
    ("average_Rice_Local_non_perfumed", 6, "Rice Local (non perfumed)"),
    ("average_Rice_Imported_perfumed", 7, "Rice Imported (perfumed)"),
    ("average_Rice_Imported_non_perfumed", 8, "Rice Imported (non perfumed)"),
    ("average_Yam_White", 9, "Yam (White)"),
    ("average_Yam_Puna", 10, "Yam (Puna)"),
    ("average_Cocoyam", 11, "Cocoyam"),
    ("average_Cassava", 12, "Cassava"),
    ("average_Gari", 13, "Gari"),
    ("average_Dried_Cassava_Powder_Kokonte", 14, "Dried Cassava Powder (Kokonte)"),
    ("average_Dried_Cassava_Chips_Kokonte", 15, "Dried Cassava Chips (Kokonte)"),
    ("average_Plantain_Apentu", 16, "Plantain (Apentu)"),
    ("average_Plantain_Apem", 17, "Plantain (Apem)"),
    ("average_Banana_Exotic", 18, "Banana (Exotic)"),
    ("average_Banana_Local", 19, "Banana (Local)"),
    ("average_Orange", 20, "Orange"),
    ("average_Watermelon", 21, "Watermelon"),
    ("average_Pineapple", 22, "Pineapple"),
    ("average_Mango_Exotic", 23, "Mango (Exotic)"),
    ("average_Mango_Local", 24, "Mango (Local)"),
    ("average_Coconut_Fresh", 25, "Coconut (Fresh)"),
    ("average_Tomato_Local", 26, "Tomato (Local)"),
    ("average_Tomato_Navrongo", 27, "Tomato (Navrongo)"),
    ("average_Garden_Egg", 28, "Garden Egg"),
    ("average_Okro_Fresh", 29, "Okro (Fresh)"),
    ("average_Okro_Dried", 30, "Okro (Dried)"),
    ("average_Onion", 31, "Onion"),
    ("average_Ginger", 32, "Ginger"),
    ("average_Dried_Pepper_Legon_18", 33, "Dried Pepper (Legon 18)"),
    ("average_Fresh_Pepper_Legon_18", 34, "Fresh Pepper (Legon 18)"),
    ("average_Fresh_Pepper_Bonnet", 35, "Fresh Pepper (Bonnet)"),
    ("average_Palm_Fruit", 36, "Palm Fruit"),
    ("average_Tiger_Nut", 37, "Tiger Nut"),
    ("average_Unshelled_Groundnut", 38, "Unshelled Groundnut"),
    ("average_Groundnut_Red", 39, "Groundnut (Red)"),
    ("average_Cowpea_White", 40, "Cowpea (White)"),
    ("average_Soya_Bean", 41, "Soya Bean"),
    ("average_Melon_seeds_Agushi", 42, "Melon seeds (Agushi)"),
    ("average_Melon_seeds_Agushi_Powder", 43, "Melon seeds (Agushi) Powder"),
    ("average_Melon_seeds_Neri", 44, "Melon seeds (Neri)"),
    ("average_Melon_seeds_Neri_Powder", 45, "Melon seeds (Neri) Powder"),
    ("average_Groundnut_Oil", 46, "Groundnut Oil"),
    ("average_Palm_Oil", 47, "Palm Oil"),
    ("average_Coconut_Oil", 48, "Coconut Oil"),
    ("average_Beef", 49, "Beef"),
    ("average_Pork", 50, "Pork"),
    ("average_Smoked_Herring", 51, "Smoked Herring"),
    ("average_Salted_Dried_Tilapia_Koobi", 52, "Salted Dried Tilapia (Koobi)"),
    ("average_Anchovy", 53, "Anchovy"),
    ("average_Kako", 54, "Kako"),
    ("average_Egg_Commercial", 55, "Egg (Commercial)"),
    ("average_Live_Bird", 56, "Live Bird"),
    ("average_Chicken", 57, "Chicken"),
    ("average_Nkontomire", 58, "Nkontomire"),
    ("average_Ademe_Ayoyo_jute_mallow", 59, "Ademe/Ayoyo (jute mallow)"),
    ("average_Alefu_Amaranthus", 60, "Alefu (Amaranthus)"),
    ("average_Cabbage", 61, "Cabbage"),
    ("average_Lettuce", 62, "Lettuce"),
    ("average_Carrot", 63, "Carrot"),
    ("average_Pawpaw", 64, "Pawpaw"),
    ("average_Avocado_Pear", 65, "Avocado Pear"),
    ("average_Bambara_Bean", 66, "Bambara Bean"),
    ("average_Mutton_Sheep_meat", 67, "Mutton (Sheep meat)"),
    ("average_Chevon_Goat_meat", 68, "Chevon (Goat meat)"),
    ("average_Snail", 69, "Snail"),
    ("average_Sweet_Potato_general_white_pinkish", 70, "Sweet Potato (white/pinkish)"),
    ("average_Sweet_Potato_ORANGE", 71, "Sweet Potato (Orange)"),
    ("average_Cassava_Dough", 72, "Cassava Dough"),
    ("average_Fresh_Cow_Milk", 73, "Fresh Cow Milk"),
    ("average_Fresh_Red_Fish", 74, "Fresh Red Fish"),
    ("average_Fresh_Salmon_Mackerel_Fish", 75, "Fresh Salmon (Mackerel) Fish"),
    ("average_Fresh_Kpanla_Fish", 76, "Fresh Kpanla Fish"),
    ("average_Plantain_Riped", 77, "Plantain (Riped)"),
]
LABELS = {code: name for code, _n, name in LABELS_RAW}
LABELS_NUM = {code: num for code, num, _n in LABELS_RAW}

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets"
DEFAULT_ASSETS = {
    "retail": "aZSU9sEwFXNtHn2AKdQoW5",
    "wholesale": "adriu7CRPWtWqKniR5tMDf",
}

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

# ── Ghana GeoJSON (embedded — 16 regions, simplified polygons) ────────────────
GHANA_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"region": "Greater Accra"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-0.55, 5.55],
                        [-0.55, 5.90],
                        [0.10, 5.90],
                        [0.10, 5.55],
                        [-0.55, 5.55],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Ashanti"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-2.70, 6.00],
                        [-2.70, 7.40],
                        [-1.00, 7.40],
                        [-1.00, 6.00],
                        [-2.70, 6.00],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Western"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-3.20, 4.75],
                        [-3.20, 6.50],
                        [-1.80, 6.50],
                        [-1.80, 4.75],
                        [-3.20, 4.75],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Central"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-1.80, 5.00],
                        [-1.80, 6.20],
                        [-0.55, 6.20],
                        [-0.55, 5.00],
                        [-1.80, 5.00],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Eastern"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-1.00, 5.80],
                        [-1.00, 7.00],
                        [0.15, 7.00],
                        [0.15, 5.80],
                        [-1.00, 5.80],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Volta"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [0.10, 5.55],
                        [0.10, 8.60],
                        [1.18, 8.60],
                        [1.18, 5.55],
                        [0.10, 5.55],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Oti"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-0.10, 7.80],
                        [-0.10, 9.00],
                        [1.18, 9.00],
                        [1.18, 7.80],
                        [-0.10, 7.80],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Bono"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-3.20, 7.20],
                        [-3.20, 8.40],
                        [-1.50, 8.40],
                        [-1.50, 7.20],
                        [-3.20, 7.20],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Bono East"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-1.50, 7.40],
                        [-1.50, 8.60],
                        [-0.10, 8.60],
                        [-0.10, 7.40],
                        [-1.50, 7.40],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Ahafo"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-3.00, 6.80],
                        [-3.00, 7.60],
                        [-2.00, 7.60],
                        [-2.00, 6.80],
                        [-3.00, 6.80],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Northern"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-2.80, 9.00],
                        [-2.80, 10.70],
                        [-0.15, 10.70],
                        [-0.15, 9.00],
                        [-2.80, 9.00],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Savannah"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-2.80, 8.40],
                        [-2.80, 9.50],
                        [-1.20, 9.50],
                        [-1.20, 8.40],
                        [-2.80, 8.40],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "North East"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-0.15, 9.50],
                        [-0.15, 10.70],
                        [1.00, 10.70],
                        [1.00, 9.50],
                        [-0.15, 9.50],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Upper East"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-1.20, 10.50],
                        [-1.20, 11.15],
                        [1.00, 11.15],
                        [1.00, 10.50],
                        [-1.20, 10.50],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Upper West"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-3.00, 9.80],
                        [-3.00, 11.15],
                        [-1.20, 11.15],
                        [-1.20, 9.80],
                        [-3.00, 9.80],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"region": "Western North"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-3.20, 6.40],
                        [-3.20, 7.40],
                        [-2.20, 7.40],
                        [-2.20, 6.40],
                        [-3.20, 6.40],
                    ]
                ],
            },
        },
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_secret(key, fallback=""):
    try:
        return st.secrets["kobo"][key]
    except Exception:
        return fallback


def fetch_kobo_asset(asset_id, label, username, password):
    url = f"{BASE_URL}/{asset_id}/data.json?page_size=5000"
    pages = []
    while url:
        resp = requests.get(url, auth=(username, password), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        df = pd.json_normalize(payload.get("results", []))
        df["source"] = label
        pages.append(df)
        url = payload.get("next")
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def strip_group_prefixes(df):
    df.columns = [c.rsplit("/", 1)[-1] for c in df.columns]
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def tidy_data(df):
    id_cols = [
        c
        for c in [
            "region",
            "district",
            "X4",
            "Market",
            "location",
            "name.officer",
            "source",
            "_geolocation",
        ]
        if c in df.columns
    ]
    avg_cols = [c for c in df.columns if c.startswith("average_")]
    df = df[[c for c in id_cols + avg_cols if c in df.columns]].copy()
    df = df.melt(
        id_vars=id_cols, value_vars=avg_cols, var_name="commodity", value_name="Price"
    )
    df["commodity_name"] = df["commodity"].map(LABELS).fillna(df["commodity"])
    df["commodity_number"] = df["commodity"].map(LABELS_NUM)
    if "X4" in df.columns:
        df["market_day"] = pd.to_datetime(df["X4"], dayfirst=True, errors="coerce")
        df["Year"] = df["market_day"].dt.year
        df["Month"] = df["market_day"].dt.month
        df["rWeek"] = df["market_day"].dt.isocalendar().week.astype("Int64")
        df["week"] = ((df["market_day"].dt.day - 1) // 7 + 1).astype("Int64")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Type"] = df["source"]
    return df


# ── Prediction engine ─────────────────────────────────────────────────────────
def predict_next_month(df, commodity_name, price_type="retail"):
    sub = df[
        (df["commodity_name"] == commodity_name) & (df["Type"] == price_type)
    ].copy()
    if sub.empty or "market_day" not in sub.columns:
        return None, None, None, "insufficient data"

    monthly = (
        sub.groupby(["Year", "Month"])["Price"]
        .mean()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    if len(monthly) < 3:
        return None, None, None, "need at least 3 months of data"

    monthly["t"] = np.arange(len(monthly))
    monthly["sin12"] = np.sin(2 * np.pi * monthly["Month"] / 12)
    monthly["cos12"] = np.cos(2 * np.pi * monthly["Month"] / 12)
    monthly["lag1"] = monthly["Price"].shift(1)
    monthly["lag2"] = monthly["Price"].shift(2)
    monthly = monthly.dropna()

    if len(monthly) < 2:
        # fallback: simple trend extrapolation
        last = monthly["Price"].iloc[-1]
        slope = (monthly["Price"].iloc[-1] - monthly["Price"].iloc[0]) / max(
            len(monthly) - 1, 1
        )
        pred = last + slope
        return round(last, 2), round(pred, 2), None, "trend only"

    features = ["t", "sin12", "cos12", "lag1", "lag2"]
    X = monthly[features].values
    y = monthly["Price"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(Xs, y)

    # Build next-month row
    last_row = monthly.iloc[-1]
    next_t = last_row["t"] + 1
    next_month = (int(last_row["Month"]) % 12) + 1
    next_row = np.array(
        [
            [
                next_t,
                np.sin(2 * np.pi * next_month / 12),
                np.cos(2 * np.pi * next_month / 12),
                last_row["Price"],
                monthly["Price"].iloc[-2] if len(monthly) >= 2 else last_row["Price"],
            ]
        ]
    )
    pred_price = model.predict(scaler.transform(next_row))[0]

    current_price = monthly["Price"].iloc[-1]
    return round(current_price, 2), round(pred_price, 2), next_month, "ok"


# ── Map builder ───────────────────────────────────────────────────────────────
def normalise_region(name):
    """Lowercase + strip for fuzzy matching between KoBo values and GeoJSON."""
    if not isinstance(name, str):
        return ""
    return name.lower().strip().replace("-", " ").replace("_", " ")


# Pre-build normalised → canonical GeoJSON region name lookup
GEOJSON_REGION_NORM = {
    normalise_region(f["properties"]["region"]): f["properties"]["region"]
    for f in GHANA_GEOJSON["features"]
}


def build_map(df, commodity_name, month_num, price_type):
    import json

    # 1. Filter and Aggregate Data
    sub = df[
        (df["commodity_name"] == commodity_name)
        & (df["Month"] == month_num)
        & (df["Type"] == price_type)
        & (df["region"].notna())
    ].copy()

    # Apply the matching logic
    sub["region_canon"] = sub["region"].apply(
        lambda r: GEOJSON_REGION_NORM.get(normalise_region(r), None)
    )
    matched = sub[sub["region_canon"].notna()]
    region_prices = matched.groupby("region_canon")["Price"].mean().to_dict()

    # 2. Setup Color Scale variables for JavaScript
    prices = list(region_prices.values())
    min_p = min(prices) if prices else 0
    max_p = max(prices) if prices else 1

    # 3. Prepare GeoJSON for JS (Inject prices into properties)
    geojson_data = json.loads(json.dumps(GHANA_GEOJSON))
    for feat in geojson_data["features"]:
        name = feat["properties"]["region"]
        p = region_prices.get(name)
        feat["properties"]["price"] = p
        feat["properties"]["price_text"] = f"GH₵ {p:,.2f}" if p else "No Data"

    # 4. Generate the HTML string (Injecting Python variables into JS)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 500px; width: 100%; border-radius: 8px; background: #f8f9fa; }}
            .info {{ padding: 6px 8px; font: 14px/16px Arial; background: rgba(255,255,255,0.8); 
                     box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const geojsonData = {json.dumps(geojson_data)};
            const map = L.map('map').setView([7.9465, -1.0232], 6);

            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png').addTo(map);

            function getColor(p) {{
                if (p === null || p === undefined) return '#d1d5db';
                const minP = {min_p};
                const maxP = {max_p};
                if (maxP === minP) return '#ffdd57';
                const ratio = (p - minP) / (maxP - minP);
                const r = Math.floor(34 + ratio * (220 - 34));
                const g = Math.floor(197 - ratio * (197 - 38));
                const b = Math.floor(94 - ratio * (94 - 38));
                return `rgb(${{r}},${{g}},${{b}})`;
            }}

            const geojson = L.geoJson(geojsonData, {{
                style: function(feature) {{
                    return {{ fillColor: getColor(feature.properties.price), weight: 1.5, color: 'white', fillOpacity: 0.8 }};
                }},
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(`<b>${{feature.properties.region}}</b><br>${{feature.properties.price_text}}`);
                }}
            }}).addTo(map);
            
            map.fitBounds(geojson.getBounds());
        </script>
    </body>
    </html>
    """
    return html_content


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — clean: just download button + credentials (hidden if secrets set)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌽 Mpogidi")

    has_secrets = bool(get_secret("username"))
    if has_secrets:
        username = get_secret("username")
        password = get_secret("password")
        st.success("🔒 Credentials secured")
    else:
        username = st.text_input("KoBo Username", placeholder="username")
        password = st.text_input(
            "KoBo Password", type="password", placeholder="password"
        )

    st.markdown("---")
    download_btn = st.button(
        "⬇️  Download & Refresh Data", use_container_width=True, type="primary"
    )
    if "fetched_at" in st.session_state:
        st.caption(f"Last refreshed: {st.session_state['fetched_at']}")

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("🌽 Mpogidi Price Monitor")
st.caption("Agricultural market price data — KoBo Collect · Ghana")

# ── Download ──────────────────────────────────────────────────────────────────
if download_btn:
    if not username or not password:
        st.error("Enter KoBo credentials in the sidebar.")
    else:
        frames, progress = [], st.progress(0, text="Starting…")
        for i, (label, aid) in enumerate(DEFAULT_ASSETS.items()):
            progress.progress(i / len(DEFAULT_ASSETS), text=f"Fetching {label}…")
            try:
                raw = fetch_kobo_asset(aid, label, username, password)
                raw = strip_group_prefixes(raw)
                frames.append(raw)
            except requests.exceptions.HTTPError as e:
                st.error(
                    f"❌ HTTP {e.response.status_code} fetching {label}. Check credentials."
                )
            except Exception as e:
                st.error(f"❌ {label}: {e}")
        progress.progress(1.0, text="Tidying…")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            st.session_state["df"] = tidy_data(combined)
            st.session_state["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success(f"✅ {len(st.session_state['df']):,} records loaded.")
        progress.empty()

if "df" not in st.session_state:
    st.info("👈 Click **Download & Refresh Data** to load data.")
    st.stop()

df = st.session_state["df"]

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Data", "🗺️ Regional Map & Prediction", "💾 Export"])

# ── TAB 1: Data ───────────────────────────────────────────────────────────────
with tab1:
    fetched_at = st.session_state.get("fetched_at", "—")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Markets", df["Market"].nunique() if "Market" in df.columns else "—")
    c3.metric("Commodities", df["commodity_name"].nunique())
    c4.metric("Last refreshed", fetched_at)

    st.markdown("### 🔍 Filter")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        price_type = st.multiselect(
            "Price type",
            df["Type"].dropna().unique().tolist(),
            default=df["Type"].dropna().unique().tolist(),
        )
    with fc2:
        commodities = st.multiselect(
            "Commodity", sorted(df["commodity_name"].dropna().unique().tolist())
        )
    with fc3:
        regions = st.multiselect(
            "Region",
            sorted(df["region"].dropna().unique().tolist())
            if "region" in df.columns
            else [],
        )

    filtered = df.copy()
    if price_type:
        filtered = filtered[filtered["Type"].isin(price_type)]
    if commodities:
        filtered = filtered[filtered["commodity_name"].isin(commodities)]
    if regions and "region" in filtered.columns:
        filtered = filtered[filtered["region"].isin(regions)]

    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} records")

    display_cols = [
        c
        for c in [
            "market_day",
            "Year",
            "Month",
            "rWeek",
            "week",
            "region",
            "district",
            "Market",
            "commodity_number",
            "commodity_name",
            "Price",
            "Type",
            "name.officer",
        ]
        if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols]
        .sort_values("market_day", ascending=False)
        .head(500)
        .reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

    if commodities and "market_day" in filtered.columns:
        import altair as alt

        chart_data = (
            filtered.groupby(["market_day", "commodity_name", "Type"])["Price"]
            .mean()
            .reset_index()
        )
        chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("market_day:T", title="Market day"),
                y=alt.Y("Price:Q", title="Avg price (GH₵)"),
                color="commodity_name:N",
                strokeDash="Type:N",
                tooltip=[
                    "market_day:T",
                    "commodity_name:N",
                    "Type:N",
                    alt.Tooltip("Price:Q", format=".0f"),
                ],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

# ── TAB 2: Map & Prediction ───────────────────────────────────────────────────
with tab2:
    import streamlit.components.v1 as components

    st.markdown("### 🗺️ Regional Price Map & 📈 Prediction")

    all_commodities = sorted(df["commodity_name"].dropna().unique().tolist())
    available_months = sorted(df["Month"].dropna().unique().astype(int).tolist())
    available_years = sorted(df["Year"].dropna().unique().astype(int).tolist())

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        map_commodity = st.selectbox("Commodity", all_commodities, key="map_comm")
    with mc2:
        current_month = datetime.now().month
        month_options = [f"{MONTH_NAMES[m]} ({m})" for m in available_months]
        # default to current month if present, else latest
        default_month_idx = next(
            (i for i, m in enumerate(available_months) if m == current_month),
            len(available_months) - 1,
        )
        map_month = st.selectbox(
            "Month", month_options, index=default_month_idx, key="map_month"
        )
        map_month_num = int(map_month.split("(")[1].rstrip(")"))
    with mc3:
        map_type = st.selectbox("Price type", ["retail", "wholesale"], key="map_type")
    with mc4:
        map_year = st.selectbox(
            "Year", available_years, index=len(available_years) - 1, key="map_year"
        )

    # ── Map Rendering ─────────────────────────────────────────────────────────
    map_df = df[df["Year"] == map_year].copy()

    # Check for matches before attempting to render
    temp_sub = map_df[
        (map_df["commodity_name"] == map_commodity)
        & (map_df["Month"] == map_month_num)
        & (map_df["Type"] == map_type)
        & (map_df["region"].notna())
    ].copy()
    temp_sub["region_canon"] = temp_sub["region"].apply(
        lambda r: GEOJSON_REGION_NORM.get(normalise_region(r), None)
    )
    matched = temp_sub[temp_sub["region_canon"].notna()]

    with st.expander("🔎 Region name debug", expanded=False):
        kobo_regions = temp_sub["region"].dropna().unique().tolist()
        st.write("**KoBo region values in data:**", sorted(kobo_regions))
        st.write(
            "**GeoJSON canonical names:**",
            sorted([f["properties"]["region"] for f in GHANA_GEOJSON["features"]]),
        )
        if not matched.empty:
            st.success(
                "**Matched:** " + str(sorted(matched["region_canon"].unique().tolist()))
            )
        else:
            st.error(
                "No regions matched — Names in KoBo don't align with the map boundaries."
            )

    if matched.empty:
        st.warning(
            f"No regional data for **{map_commodity}** in "
            f"**{MONTH_NAMES[map_month_num]} {map_year}** ({map_type})."
        )
    else:
        # Generate the Leaflet HTML
        map_html = build_map(map_df, map_commodity, map_month_num, map_type)
        # Render using Streamlit Components
        components.html(map_html, height=520)

    st.markdown("---")

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("### 🔮 National Price Prediction — Next Month")
    st.caption(
        "Uses a Ridge regression model with lag prices, trend, and seasonal patterns "
        "fitted on all available monthly data. Fuel price data is not publicly accessible "
        "via API in this region, so the prediction relies entirely on your KoBo price history."
    )

    current_price, pred_price, next_month, status = predict_next_month(
        df, map_commodity, map_type
    )

    if status == "ok" and pred_price is not None:
        pct_change = ((pred_price - current_price) / current_price) * 100
        next_month_name = MONTH_NAMES.get(next_month, "Next month")

        if pct_change > 2:
            direction = "📈 Increase"
            color = "#fee2e2"
            border = "#ef4444"
            arrow = "▲"
        elif pct_change < -2:
            direction = "📉 Decrease"
            color = "#dcfce7"
            border = "#22c55e"
            arrow = "▼"
        else:
            direction = "➡️ Stable"
            color = "#fef9c3"
            border = "#eab308"
            arrow = "●"

        p1, p2, p3 = st.columns(3)
        p1.metric("Current national avg (GH₵)", f"{current_price:,.2f}")
        p2.metric(
            f"Predicted — {next_month_name} (GH₵)",
            f"{pred_price:,.2f}",
            delta=f"{pct_change:+.1f}%",
        )
        p3.metric("Direction", direction)

        st.markdown(
            f'<div class="pred-box" style="background:{color};border:1.5px solid {border};color:#111111;">'
            f"<b>{arrow} {direction}</b> &nbsp;·&nbsp; "
            f"Predicted national average for <b>{map_commodity}</b> ({map_type}) "
            f"in <b>{next_month_name}</b>: "
            f"<b>GH₵ {pred_price:,.2f}</b> "
            f"({pct_change:+.1f}% vs current GH₵ {current_price:,.2f})<br>"
            f'<small style="color:#333;">Model: Ridge regression · '
            f"Features: trend, seasonality, 2-month price lag · "
            f"Fuel price data excluded (not available via open API)</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show historical trend used for prediction
        sub = df[(df["commodity_name"] == map_commodity) & (df["Type"] == map_type)]
        monthly_hist = (
            sub.groupby(["Year", "Month"])["Price"]
            .mean()
            .reset_index()
            .sort_values(["Year", "Month"])
        )
        if not monthly_hist.empty:
            import altair as alt

            monthly_hist["date_label"] = monthly_hist.apply(
                lambda r: f"{MONTH_NAMES[int(r.Month)]} {int(r.Year)}", axis=1
            )
            monthly_hist["t"] = range(len(monthly_hist))

            # Add prediction point
            pred_row = pd.DataFrame(
                [
                    {
                        "Year": monthly_hist["Year"].iloc[-1]
                        + (1 if next_month == 1 else 0),
                        "Month": next_month,
                        "Price": pred_price,
                        "date_label": f"{next_month_name} (pred)",
                        "t": len(monthly_hist),
                    }
                ]
            )
            hist_chart = pd.concat([monthly_hist, pred_row], ignore_index=True)
            hist_chart["is_pred"] = hist_chart["date_label"].str.contains("pred")

            base = alt.Chart(hist_chart)
            line = base.mark_line().encode(
                x=alt.X("t:Q", axis=alt.Axis(labels=False), title=""),
                y=alt.Y("Price:Q", title="Avg price (GH₵)"),
                color=alt.condition(
                    alt.datum.is_pred, alt.value("#ef4444"), alt.value("#3b82f6")
                ),
                tooltip=["date_label:N", alt.Tooltip("Price:Q", format=".0f")],
            )
            points = base.mark_point(size=60).encode(
                x="t:Q",
                y="Price:Q",
                color=alt.condition(
                    alt.datum.is_pred, alt.value("#ef4444"), alt.value("#3b82f6")
                ),
                tooltip=["date_label:N", alt.Tooltip("Price:Q", format=".0f")],
            )
            st.altair_chart(
                (line + points)
                .properties(
                    height=220,
                    title=f"{map_commodity} ({map_type}) — historical & prediction",
                )
                .interactive(),
                use_container_width=True,
            )

    elif status == "insufficient data":
        st.info(f"Not enough data to predict for **{map_commodity}**.")
    elif status == "need at least 3 months of data":
        st.info("Need at least 3 months of data for a reliable prediction.")
    else:
        st.info(f"Prediction unavailable: {status}")

# ── TAB 3: Export ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 💾 Export data")

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        exp_type = st.multiselect(
            "Price type",
            df["Type"].dropna().unique().tolist(),
            default=df["Type"].dropna().unique().tolist(),
            key="exp_type",
        )
    with ec2:
        exp_comm = st.multiselect(
            "Commodity (blank = all)",
            sorted(df["commodity_name"].dropna().unique().tolist()),
            key="exp_comm",
        )
    with ec3:
        exp_reg = st.multiselect(
            "Region (blank = all)",
            sorted(df["region"].dropna().unique().tolist())
            if "region" in df.columns
            else [],
            key="exp_reg",
        )

    exp_df = df.copy()
    if exp_type:
        exp_df = exp_df[exp_df["Type"].isin(exp_type)]
    if exp_comm:
        exp_df = exp_df[exp_df["commodity_name"].isin(exp_comm)]
    if exp_reg and "region" in exp_df.columns:
        exp_df = exp_df[exp_df["region"].isin(exp_reg)]

    st.caption(f"{len(exp_df):,} records ready to export")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download CSV",
            data=exp_df.to_csv(index=False).encode(),
            file_name=f"mpogidi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        xl = io.BytesIO()
        with pd.ExcelWriter(xl, engine="openpyxl") as w:
            exp_df.to_excel(w, index=False, sheet_name="All")
            exp_df[exp_df["Type"] == "retail"].to_excel(
                w, index=False, sheet_name="Retail"
            )
            exp_df[exp_df["Type"] == "wholesale"].to_excel(
                w, index=False, sheet_name="Wholesale"
            )
        st.download_button(
            "⬇️ Download Excel",
            data=xl.getvalue(),
            file_name=f"mpogidi_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.markdown("---")
st.caption("Built by ICT Unit · Data source: KoBo Collect (eu.kobotoolbox.org)")
