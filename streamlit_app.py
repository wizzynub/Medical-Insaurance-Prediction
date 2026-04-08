import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


@st.cache_data(show_spinner=False)
def load_dataset(default_path: str) -> pd.DataFrame:
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    raise FileNotFoundError(f"Could not find dataset at {default_path}")


def preprocess_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = raw_df.copy()
    # Basic sanity: required columns
    required = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    # Map binary categoricals consistent with final.py
    df["sex"] = df["sex"].map({"female": 0, "male": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

    # One-hot encode region
    dummies_region = pd.get_dummies(df["region"], dtype=int)
    df = pd.concat([df.drop(columns=["region"]), dummies_region], axis=1)

    X = df.drop("charges", axis=1)
    y = df["charges"]
    return X, y, dummies_region


def scale_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, pd.Index]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X.columns,
    )


def train_random_forest(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    random_state: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)
    return model


def build_single_input_dataframe(
    age: int,
    sex_label: str,
    bmi: float,
    children: int,
    smoker_label: str,
    region_label: str,
    full_feature_columns: pd.Index,
) -> pd.DataFrame:
    # Start with base numerical/encoded fields
    row: Dict[str, Any] = {
        "age": age,
        "sex": {"female": 0, "male": 1}[sex_label],
        "bmi": bmi,
        "children": children,
        "smoker": {"no": 0, "yes": 1}[smoker_label],
    }

    # Derive all region dummy columns from training
    region_dummy_cols = [c for c in full_feature_columns if c not in ["age", "sex", "bmi", "children", "smoker"]]
    for col in region_dummy_cols:
        row[col] = 1 if col == region_label else 0

    # Create DataFrame with the exact same column order
    df = pd.DataFrame([row], columns=full_feature_columns)
    df.fillna(0, inplace=True)
    return df


def main() -> None:
    st.set_page_config(page_title="Medical Insurance Charges - Regression", layout="wide")
    st.title("Medical Insurance Charges Prediction")
    st.caption("Interactive Streamlit app based on your RandomForest workflow")

    default_csv_path = "medicalInsaurance.csv"

    with st.sidebar:
        st.header("Data & Model Settings")
        source_choice = st.radio("Dataset source", ["Use bundled CSV", "Upload CSV"], index=0)

        uploaded_df: Optional[pd.DataFrame] = None
        if source_choice == "Upload CSV":
            uploaded = st.file_uploader("Upload a CSV with the same schema", type=["csv"])
            if uploaded is not None:
                uploaded_df = pd.read_csv(uploaded)

        test_size = st.slider("Test size", 0.1, 0.4, 0.33, 0.01)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

        st.subheader("Random Forest Hyperparameters")
        n_estimators = st.slider("n_estimators", 50, 500, 200, 10)
        max_depth_choice = st.selectbox("max_depth", ["None", 5, 10, 15, 20], index=2)
        max_depth_value = 0 if max_depth_choice == "None" else int(max_depth_choice)
        min_samples_split = st.selectbox("min_samples_split", [2, 5, 10], index=2)

        train_clicked = st.button("Train / Retrain Model", type="primary")

    # Load data
    try:
        if uploaded_df is not None:
            raw_df = uploaded_df
        else:
            raw_df = load_dataset(default_csv_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Preview of Data")
    st.dataframe(raw_df.head(), use_container_width=True)

    # Preprocess
    try:
        X, y, dummies_region = preprocess_dataframe(raw_df)
    except Exception as exc:
        st.error(f"Preprocessing error: {exc}")
        st.stop()

    with st.expander("Show derived region dummy columns"):
        st.write(list(dummies_region.columns))

    # Train (with caching keyed by parameters and data digest)
    if train_clicked or "trained_model" not in st.session_state:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns = scale_train_test(
            X, y, test_size=float(test_size), random_state=int(random_state)
        )
        model = train_random_forest(
            X_train_scaled,
            y_train,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth_value),
            min_samples_split=int(min_samples_split),
            random_state=int(random_state),
        )
        st.session_state["trained_model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["feature_columns"] = feature_columns
        st.session_state["X_test_scaled"] = X_test_scaled
        st.session_state["y_test"] = y_test

    if "trained_model" not in st.session_state:
        st.info("Click 'Train / Retrain Model' to train the model")
        st.stop()

    model: RandomForestRegressor = st.session_state["trained_model"]
    scaler: StandardScaler = st.session_state["scaler"]
    feature_columns: pd.Index = st.session_state["feature_columns"]
    X_test_scaled: np.ndarray = st.session_state["X_test_scaled"]
    y_test: pd.Series = st.session_state["y_test"]

    # Evaluate
    r2_test = model.score(X_test_scaled, y_test)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² on test set", f"{r2_test:.4f}")
        with col2:
            try:
                cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring="r2")
                st.metric("Cross-validated R² (on test split)", f"{cv_scores.mean():.4f}")
            except Exception:
                st.write("Cross-validation unavailable for current split size.")

    # Feature importance (align names)
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({"feature": list(feature_columns), "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False)
            st.subheader("Feature Importances")
            st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        pass

    st.subheader("Make a Prediction")
    with st.form("predict_form", clear_on_submit=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            in_age = st.number_input("age", min_value=0, max_value=120, value=30, step=1)
            in_children = st.number_input("children", min_value=0, max_value=10, value=0, step=1)
        with col_b:
            in_bmi = st.number_input("bmi", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            in_sex = st.selectbox("sex", ["female", "male"], index=0)
        with col_c:
            in_smoker = st.selectbox("smoker", ["no", "yes"], index=0)
            # Region options are from training-time dummy columns
            region_cols = [c for c in feature_columns if c not in ["age", "sex", "bmi", "children", "smoker"]]
            default_region = region_cols[0] if region_cols else "region_unknown"
            in_region = st.selectbox("region", options=region_cols or [default_region])

        submitted = st.form_submit_button("Predict charges")

    if submitted:
        try:
            single_df = build_single_input_dataframe(
                age=int(in_age),
                sex_label=str(in_sex),
                bmi=float(in_bmi),
                children=int(in_children),
                smoker_label=str(in_smoker),
                region_label=str(in_region),
                full_feature_columns=feature_columns,
            )
            single_scaled = scaler.transform(single_df)
            pred = model.predict(single_scaled)[0]
            st.success(f"Predicted charges: ${pred:,.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()


