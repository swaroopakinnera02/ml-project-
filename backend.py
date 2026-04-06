from contextlib import asynccontextmanager
import pickle, os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLS = ["RAM", "Storage_GB", "Screen_Size"]
MODEL_PATH = "laptop_price_model.pkl"

# ── Data ──────────────────────────────────────────────────────────────────────

def parse_rom(val):
    val = str(val).strip().upper()
    if "TB" in val:
        return int(val.replace("TB", "").strip()) * 1000
    return int(val.replace("GB", "").strip())

def load_and_clean():
    df = pd.read_csv("data.csv")
    df["RAM"]         = df["Ram"].astype(str).str.replace(r"[^\d]", "", regex=True).astype(int)
    df["Storage_GB"]  = df["ROM"].apply(parse_rom)
    df["Screen_Size"] = pd.to_numeric(df["display_size"], errors="coerce")
    df["Price"]       = pd.to_numeric(df["price"], errors="coerce")
    df = df[["brand","name","RAM","Storage_GB","Screen_Size","ROM_type","processor","OS","Price"]].dropna()
    df = df[(df["RAM"]>0)&(df["Storage_GB"]>0)&(df["Screen_Size"]>0)&(df["Price"]>0)]
    return df.reset_index(drop=True)

# ── ML ────────────────────────────────────────────────────────────────────────

def train(df):
    X = df[FEATURE_COLS]
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    score = round(float(r2_score(y_test, model.predict(X_test))), 4)
    return model, score

# ── App state ─────────────────────────────────────────────────────────────────

class State:
    df = None
    model = None
    r2 = None

state = State()

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.df = load_and_clean()
    state.model, state.r2 = train(state.df)
    print(f"✅ Model ready — R²={state.r2}")
    yield

app = FastAPI(title="Laptop Price Predictor API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ram: int         = Field(..., gt=0)
    storage_gb: int  = Field(..., gt=0)
    screen_size: float = Field(..., gt=0)

class SimilarRequest(BaseModel):
    ram: int
    storage_gb: int
    screen_size: float
    tolerance: float = 0.5
    limit: int = 5

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/dataset/stats")
def stats():
    df = state.df
    return {
        "total_records":   int(len(df)),
        "avg_price":       round(float(df["Price"].mean()), 2),
        "min_price":       int(df["Price"].min()),
        "max_price":       int(df["Price"].max()),
        "ram_options":     sorted(df["RAM"].unique().tolist()),
        "storage_options": sorted(df["Storage_GB"].unique().tolist()),
        "screen_min":      float(df["Screen_Size"].min()),
        "screen_max":      float(df["Screen_Size"].max()),
        "brands":          sorted(df["brand"].unique().tolist()),
    }

@app.get("/dataset/preview")
def preview(n: int = Query(10, ge=1, le=100)):
    cols = ["brand","name","RAM","Storage_GB","Screen_Size","Price"]
    return {"total": len(state.df), "records": state.df[cols].head(n).to_dict(orient="records")}

@app.get("/dataset/brand-prices")
def brand_prices():
    return (
        state.df.groupby("brand")["Price"].mean().round(2)
        .sort_values().reset_index()
        .rename(columns={"Price":"avg_price"})
        .to_dict(orient="records")
    )

@app.post("/predict")
def predict(body: PredictRequest):
    import pandas as pd
    X = pd.DataFrame([[body.ram, body.storage_gb, body.screen_size]], columns=FEATURE_COLS)
    price = round(float(state.model.predict(X)[0]), 2)
    return {"predicted_price": max(price, 0), "currency": "INR", "inputs": body.model_dump()}

@app.post("/similar")
def similar(body: SimilarRequest):
    df = state.df
    cols = ["brand","name","RAM","Storage_GB","Screen_Size","Price"]
    mask = (
        (df["RAM"] == body.ram) &
        (df["Storage_GB"] == body.storage_gb) &
        (df["Screen_Size"].between(body.screen_size - body.tolerance, body.screen_size + body.tolerance))
    )
    return df.loc[mask, cols].head(body.limit).to_dict(orient="records")

@app.get("/model/info")
def model_info():
    m = state.model
    return {
        "r2_score":     state.r2,
        "feature_cols": FEATURE_COLS,
        "intercept":    round(float(m.intercept_), 2),
        "coefficients": {col: round(float(c), 2) for col, c in zip(FEATURE_COLS, m.coef_)},
    }