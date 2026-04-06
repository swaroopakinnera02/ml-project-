import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #e3f2fd; }
.stButton>button { background-color: #0e6ba8; color: white; border-radius: 10px; }
.stButton>button:hover { background-color: #144552; color: white; }
</style>
""", unsafe_allow_html=True)

# ── Data loading & cleaning ──────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv")

    # RAM: "8GB" → 8
    df["RAM"] = (
        df["Ram"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .astype(int)
    )

    # Storage: "512GB" → 512, "1TB" → 1024
    def parse_rom(val):
        val = str(val).strip().upper()
        if "TB" in val:
            return int(val.replace("TB", "").strip()) * 1000
        return int(val.replace("GB", "").strip())

    df["Storage_GB"] = df["ROM"].apply(parse_rom)

    # Screen size (already float)
    df["Screen_Size"] = pd.to_numeric(df["display_size"], errors="coerce")

    # Price (already int, in ₹)
    df["Price"] = pd.to_numeric(df["price"], errors="coerce")

    # Keep useful columns and drop bad rows
    df = df[["brand", "name", "RAM", "Storage_GB", "Screen_Size",
             "ROM_type", "processor", "OS", "Price"]].dropna()

    df = df[
        (df["RAM"] > 0) &
        (df["Storage_GB"] > 0) &
        (df["Screen_Size"] > 0) &
        (df["Price"] > 0)
    ].reset_index(drop=True)

    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[["RAM", "Storage_GB", "Screen_Size"]]
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))
    return model, accuracy


df = load_data()
model, accuracy = train_model(df)

# ── Sidebar inputs ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Laptop Specifications")

    ram_opts     = sorted(df["RAM"].unique().tolist())
    storage_opts = sorted(df["Storage_GB"].unique().tolist())

    ram     = st.selectbox("RAM (GB)", ram_opts,
                           index=ram_opts.index(8) if 8 in ram_opts else 0)
    storage = st.selectbox("Storage (GB)", storage_opts,
                           index=storage_opts.index(512) if 512 in storage_opts else 0)
    screen  = st.slider("Screen Size (inch)",
                        float(df["Screen_Size"].min()),
                        float(df["Screen_Size"].max()),
                        15.6, 0.1)

    st.markdown("---")
    st.caption("📂 **Data source**\n\nUploaded dataset — Indian laptop market (~900 real listings)")

input_df = pd.DataFrame({
    "RAM":         [ram],
    "Storage_GB":  [storage],
    "Screen_Size": [screen],
})

# ── Main layout ──────────────────────────────────────────────────────────────

st.title("💻 Laptop Price Predictor")
st.markdown("Predict laptop price using Machine Learning — powered by a **real Indian laptop dataset**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Preview")
    st.caption(f"Showing first 10 of **{len(df):,}** real laptop listings")
    display_cols = ["brand", "name", "RAM", "Storage_GB", "Screen_Size", "Price"]
    st.dataframe(df[display_cols].head(10), use_container_width=True)

with col2:
    st.subheader("📈 Model Info")
    st.metric("R² Score",       f"{accuracy:.2f}")
    st.metric("Total Records",  f"{len(df):,}")
    st.metric("Avg. Price (₹)", f"₹ {df['Price'].mean():,.0f}")
    st.metric("Price Range",    f"₹ {df['Price'].min():,} – ₹ {df['Price'].max():,}")

# ── Charts ────────────────────────────────────────────────────────────────────

col3, col4 = st.columns(2)

with col3:
    st.subheader("📉 Price vs RAM")
    fig, ax = plt.subplots(figsize=(6, 3))
    for brand, grp in df.groupby("brand"):
        ax.scatter(grp["RAM"], grp["Price"] / 1000,
                   alpha=0.5, s=30, label=brand)
    ax.set_xlabel("RAM (GB)")
    ax.set_ylabel("Price (₹ thousands)")
    ax.set_title("RAM vs Price by Brand")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.subheader("📊 Avg Price by Brand")
    brand_avg = df.groupby("brand")["Price"].mean().sort_values(ascending=True)
    n_brands = len(brand_avg)
    fig2, ax2 = plt.subplots(figsize=(6, max(4, n_brands * 0.35)))
    bars = ax2.barh(brand_avg.index, brand_avg.values / 1000,
                    color="#0e6ba8", alpha=0.85, height=0.6)
    # Add value labels at end of each bar
    for bar, val in zip(bars, brand_avg.values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"₹{val/1000:.0f}k", va="center", ha="left", fontsize=7)
    ax2.set_xlabel("Avg Price (₹ thousands)")
    ax2.set_title("Average Price by Brand")
    ax2.tick_params(axis="y", labelsize=8)
    ax2.set_xlim(0, brand_avg.max() / 1000 * 1.2)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Prediction ───────────────────────────────────────────────────────────────

st.subheader("💰 Predict Laptop Price")

if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"Estimated Laptop Price: ₹ {price:,.0f}")
    st.caption(
        f"Based on: RAM = {ram} GB | Storage = {storage} GB | "
        f"Screen = {screen}\""
    )

    # Show similar real laptops from the dataset
    st.subheader("🔍 Similar Laptops in Dataset")
    similar = df[
        (df["RAM"] == ram) &
        (df["Storage_GB"] == storage) &
        (df["Screen_Size"].between(screen - 0.5, screen + 0.5))
    ][["brand", "name", "RAM", "Storage_GB", "Screen_Size", "Price"]].head(5)

    if not similar.empty:
        st.dataframe(similar, use_container_width=True)
    else:
        st.info("No exact matches found — try adjusting the specs.")