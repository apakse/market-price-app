import streamlit as st
import requests
import pandas as pd
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mpogidi Price Monitor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f8f7f4; }
h1 { font-weight: 600; letter-spacing: -0.5px; }
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Commodity labels (from labels_df.xlsx — all 77 commodities) ───────────────
LABELS = {
    "average_white_maize": "Maize (White)",
    "average_Maize_Yellow": "Maize (Yellow)",
    "average_Millet": "Millet",
    "average_Sorghum": "Sorghum",
    "average_Rice_Local_perfumed": "Rice Local (perfumed)",
    "average_Rice_Local_non_perfumed": "Rice Local (non perfumed)",
    "average_Rice_Imported_perfumed": "Rice Imported (perfumed)",
    "average_Rice_Imported_non_perfumed": "Rice Imported (non perfumed)",
    "average_Yam_White": "Yam (White)",
    "average_Yam_Puna": "Yam (Puna)",
    "average_Cocoyam": "Cocoyam",
    "average_Cassava": "Cassava",
    "average_Gari": "Gari",
    "average_Dried_Cassava_Powder_Kokonte": "Dried Cassava Powder (Kokonte)",
    "average_Dried_Cassava_Chips_Kokonte": "Dried Cassava Chips (Kokonte)",
    "average_Plantain_Apentu": "Plantain (Apentu)",
    "average_Plantain_Apem": "Plantain (Apem)",
    "average_Banana_Exotic": "Banana (Exotic)",
    "average_Banana_Local": "Banana (Local)",
    "average_Orange": "Orange",
    "average_Watermelon": "Watermelon",
    "average_Pineapple": "Pineapple",
    "average_Mango_Exotic": "Mango (Exotic)",
    "average_Mango_Local": "Mango (Local)",
    "average_Coconut_Fresh": "Coconut (Fresh)",
    "average_Tomato_Local": "Tomato (Local)",
    "average_Tomato_Navrongo": "Tomato (Navrongo)",
    "average_Garden_Egg": "Garden Egg",
    "average_Okro_Fresh": "Okro (Fresh)",
    "average_Okro_Dried": "Okro (Dried)",
    "average_Onion": "Onion",
    "average_Ginger": "Ginger",
    "average_Dried_Pepper_Legon_18": "Dried Pepper (Legon 18)",
    "average_Fresh_Pepper_Legon_18": "Fresh Pepper (Legon 18)",
    "average_Fresh_Pepper_Bonnet": "Fresh Pepper (Bonnet)",
    "average_Palm_Fruit": "Palm Fruit",
    "average_Tiger_Nut": "Tiger Nut",
    "average_Unshelled_Groundnut": "Unshelled Groundnut",
    "average_Groundnut_Red": "Groundnut (Red)",
    "average_Cowpea_White": "Cowpea (White)",
    "average_Soya_Bean": "Soya Bean",
    "average_Melon_seeds_Agushi": "Melon seeds (Agushi)",
    "average_Melon_seeds_Agushi_Powder": "Melon seeds (Agushi) Powder",
    "average_Melon_seeds_Neri": "Melon seeds (Neri)",
    "average_Melon_seeds_Neri_Powder": "Melon seeds (Neri) Powder",
    "average_Groundnut_Oil": "Groundnut Oil",
    "average_Palm_Oil": "Palm Oil",
    "average_Coconut_Oil": "Coconut Oil",
    "average_Beef": "Beef",
    "average_Pork": "Pork",
    "average_Smoked_Herring": "Smoked Herring",
    "average_Salted_Dried_Tilapia_Koobi": "Salted Dried Tilapia (Koobi)",
    "average_Anchovy": "Anchovy",
    "average_Kako": "Kako",
    "average_Egg_Commercial": "Egg (Commercial)",
    "average_Live_Bird": "Live Bird",
    "average_Chicken": "Chicken",
    "average_Nkontomire": "Nkontomire",
    "average_Ademe_Ayoyo_jute_mallow": "Ademe/Ayoyo (jute mallow)",
    "average_Alefu_Amaranthus": "Alefu (Amaranthus)",
    "average_Cabbage": "Cabbage",
    "average_Lettuce": "Lettuce",
    "average_Carrot": "Carrot",
    "average_Pawpaw": "Pawpaw",
    "average_Avocado_Pear": "Avocado Pear",
    "average_Bambara_Bean": "Bambara Bean",
    "average_Mutton_Sheep_meat": "Mutton (Sheep meat)",
    "average_Chevon_Goat_meat": "Chevon (Goat meat)",
    "average_Snail": "Snail",
    "average_Sweet_Potato_general_white_pinkish": "Sweet Potato (white/pinkish)",
    "average_Sweet_Potato_ORANGE": "Sweet Potato (Orange)",
    "average_Cassava_Dough": "Cassava Dough",
    "average_Fresh_Cow_Milk": "Fresh Cow Milk",
    "average_Fresh_Red_Fish": "Fresh Red Fish",
    "average_Fresh_Salmon_Mackerel_Fish": "Fresh Salmon (Mackerel) Fish",
    "average_Fresh_Kpanla_Fish": "Fresh Kpanla Fish",
    "average_Plantain_Riped": "Plantain (Riped)",
}

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets"
DEFAULT_ASSETS = {
    "retail":    "aZSU9sEwFXNtHn2AKdQoW5",
    "wholesale": "adriu7CRPWtWqKniR5tMDf",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    id_cols  = [c for c in ["region", "district", "X4", "Market", "location",
                              "name.officer", "source", "_geolocation"] if c in df.columns]
    avg_cols = [c for c in df.columns if c.startswith("average_")]

    df = df[[c for c in id_cols + avg_cols if c in df.columns]].copy()
    df = df.melt(id_vars=id_cols, value_vars=avg_cols,
                 var_name="commodity", value_name="Price")

    # Human-readable names from labels_df
    df["commodity_name"] = df["commodity"].map(LABELS).fillna(df["commodity"])

    if "X4" in df.columns:
        df["market_day"] = pd.to_datetime(df["X4"], dayfirst=True, errors="coerce")
        df["Year"]  = df["market_day"].dt.year
        df["Month"] = df["market_day"].dt.month
        df["Week"]  = df["market_day"].dt.isocalendar().week.astype("Int64")

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Type"] = df["source"]
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("**KoBo Credentials**")
    username = st.text_input("Username", value="sridhiswap")
    password = st.text_input("Password", type="password", value="Fl*#.Dc@sR1d")

    st.markdown("---")
    st.markdown("**Asset IDs** *(edit if forms change)*")
    retail_id    = st.text_input("Retail asset ID",    value=DEFAULT_ASSETS["retail"])
    wholesale_id = st.text_input("Wholesale asset ID", value=DEFAULT_ASSETS["wholesale"])

    st.markdown("---")
    download_btn = st.button("⬇️  Download & Refresh Data",
                             use_container_width=True, type="primary")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌽 Mpogidi Price Monitor")
st.caption("Agricultural market price data — KoBo Collect")

if "df" not in st.session_state:
    st.info("Configure credentials in the sidebar, then click **Download & Refresh Data**.")
    st.stop()

# ── Download ──────────────────────────────────────────────────────────────────
if download_btn:
    assets = {"retail": retail_id, "wholesale": wholesale_id}
    frames = []
    progress = st.progress(0, text="Starting download…")
    for i, (label, aid) in enumerate(assets.items()):
        progress.progress(i / len(assets), text=f"Fetching {label} data…")
        try:
            raw = fetch_kobo_asset(aid, label, username, password)
            raw = strip_group_prefixes(raw)
            frames.append(raw)
        except Exception as e:
            st.error(f"Failed to fetch **{label}**: {e}")
    progress.progress(1.0, text="Tidying data…")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        st.session_state["df"] = tidy_data(combined)
        st.session_state["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.success(f"✅ {len(st.session_state['df']):,} price records loaded.")
    progress.empty()

# ── Data ──────────────────────────────────────────────────────────────────────
df = st.session_state["df"]
fetched_at = st.session_state.get("fetched_at", "—")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records",  f"{len(df):,}")
col2.metric("Markets",        df["Market"].nunique() if "Market" in df.columns else "—")
col3.metric("Commodities",    df["commodity_name"].nunique())
col4.metric("Last refreshed", fetched_at)

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Filter")
fc1, fc2, fc3 = st.columns(3)

with fc1:
    price_type = st.multiselect("Price type",
                                options=df["Type"].dropna().unique().tolist(),
                                default=df["Type"].dropna().unique().tolist())
with fc2:
    all_commodities = sorted(df["commodity_name"].dropna().unique().tolist())
    commodities = st.multiselect("Commodity", options=all_commodities, default=[])

with fc3:
    if "region" in df.columns:
        all_regions = sorted(df["region"].dropna().unique().tolist())
        regions = st.multiselect("Region", options=all_regions, default=[])
    else:
        regions = []

filtered = df.copy()
if price_type:
    filtered = filtered[filtered["Type"].isin(price_type)]
if commodities:
    filtered = filtered[filtered["commodity_name"].isin(commodities)]
if regions and "region" in filtered.columns:
    filtered = filtered[filtered["region"].isin(regions)]

st.caption(f"Showing **{len(filtered):,}** of {len(df):,} records")

# ── Table ─────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Data preview")
display_cols = [c for c in ["market_day", "region", "district", "Market",
                             "commodity_name", "Price", "Type", "name.officer"]
                if c in filtered.columns]
st.dataframe(
    filtered[display_cols]
    .sort_values("market_day", ascending=False)
    .head(500)
    .reset_index(drop=True),
    use_container_width=True,
    height=380,
)

# ── Chart ─────────────────────────────────────────────────────────────────────
if commodities and "market_day" in filtered.columns:
    st.markdown("### 📈 Price trend")
    import altair as alt
    chart_data = (
        filtered
        .groupby(["market_day", "commodity_name", "Type"])["Price"]
        .mean()
        .reset_index()
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("market_day:T", title="Market day"),
            y=alt.Y("Price:Q", title="Average price"),
            color="commodity_name:N",
            strokeDash="Type:N",
            tooltip=["market_day:T", "commodity_name:N", "Type:N",
                     alt.Tooltip("Price:Q", format=".0f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("### 💾 Export")
dl1, dl2 = st.columns(2)

with dl1:
    csv_buf = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered (CSV)", data=csv_buf,
        file_name=f"mpogidi_prices_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True,
    )

with dl2:
    xl_buf = io.BytesIO()
    with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="Prices")
        df[df["Type"] == "retail"].to_excel(writer, index=False, sheet_name="Retail")
        df[df["Type"] == "wholesale"].to_excel(writer, index=False, sheet_name="Wholesale")
    st.download_button(
        "⬇️ Download filtered (Excel)", data=xl_buf.getvalue(),
        file_name=f"mpogidi_prices_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Built by ICT Unit · Data source: KoBo Collect (eu.kobotoolbox.org)")
