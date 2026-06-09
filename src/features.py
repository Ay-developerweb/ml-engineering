""" features.py 
    - Reusable preprocessing for the Ames Housing dataset.
    Usage: from src.features import build_feature_matrix
            X, y = build_feature_matrix("data/raw/house_prices.csv") 
""" 
import pandas as pd 
import numpy as np 

# ── Column definitions ───────────────────────────────────────────────────────── 
STRUCTURAL_CAT = [ "Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Mas Vnr Type", ] 
STRUCTURAL_NUM = [ "Garage Yr Blt", "Garage Area", "Garage Cars", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath", "Mas Vnr Area", ]
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0} 
QUALITY_COLS = [ "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC", ]
OHE_COLS = ["Neighborhood", "House Style", "Sale Type", "Sale Condition"]
DROP_COLS = ["Order", "PID"]

# ── Pipeline steps ───────────────────────────────────────────────────────────── 
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using domain-aware strategy."""
    df = df.copy()
    for col in STRUCTURAL_CAT:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in STRUCTURAL_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # These two are OUTSIDE the loops
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    return df

 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from domain reasoning."""
    df = df.copy()
    df["house_age"] = df["Yr Sold"] - df["Year Built"]
    df["remodel_age"] = df["Yr Sold"] - df["Year Remod/Add"] 
    df["total_sf"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"] 
    df["total_bathrooms"] = ( df["Full Bath"] + df["Half Bath"] * 0.5 + df["Bsmt Full Bath"] + df["Bsmt Half Bath"] * 0.5 ) 
    df["has_garage"] = (df["Garage Area"] > 0).astype(int) 
    df["has_pool"] = (df["Pool Area"] > 0).astype(int) 
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode quality columns and one-hot encode categoricals."""
    df = df.copy()
    for col in QUALITY_COLS:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)
    return df
    
def build_feature_matrix( filepath: str, ) -> tuple[pd.DataFrame, pd.Series]:
    """ Full pipeline: load → clean → engineer → encode → return (X, y). 
        Parameters ---------- filepath : str Path to the raw Ames Housing CSV. 
        Returns ------- X : pd.DataFrame — feature matrix (samples × features) 
                        y : pd.Series — SalePrice target 
    """
    df = pd.read_csv(filepath) 
    df = fill_missing(df)
    df = engineer_features(df) 
    df = encode_features(df) 
    
    # Drop raw string columns and ID cols 
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist() 
    enc_cols = [col + "_enc" for col in QUALITY_COLS]
    drop = DROP_COLS + QUALITY_COLS + OHE_COLS + string_cols + enc_cols
    drop = [c for c in drop if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop + ["SalePrice"]] 
    X = df[feature_cols].astype(float) 
    y = df["SalePrice"] 
    return X, y