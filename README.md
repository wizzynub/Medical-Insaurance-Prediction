# Medical Insurance Regression (Streamlit App)

Predict medical insurance charges using a RandomForest model, with an interactive Streamlit UI for training, evaluation, and single-record prediction.

## Features
- Load bundled `medicalInsaurance.csv` or upload your own CSV with the same schema
- Preprocessing consistent with `final.py` (binary mappings + one-hot for `region`)
- Train/test split, feature scaling, RandomForest training
- Test R², optional cross-validated R² (on test split), and feature importances
- Interactive form to predict charges for a single individual

## Project Structure
- `final.py`: Original experimentation/training script
- `streamlit_app.py`: Streamlit app (training, evaluation, prediction)
- `medicalInsaurance.csv`: Dataset used by default (ensure present in the project root)
- `requirements.txt`: Python dependencies

## Setup
1. Create and activate a virtual environment (recommended)
   - Windows (PowerShell):
     ```bash
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux (bash):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Streamlit App
From the project root:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser (default: `http://localhost:8501`).

## Using the App
1. In the sidebar:
   - Choose dataset source (bundled CSV or upload your own with same columns)
   - Adjust test size and random seed
   - Set RandomForest hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`)
   - Click "Train / Retrain Model"
2. Review metrics:
   - Test R²
   - Cross-validated R² (if the split size allows)
   - Feature importances
3. Use the form to enter a single individual's details and click Predict to see the estimated charges.

## Data Schema
Your CSV should include these columns:
- `age` (int)
- `sex` (categorical: `female`, `male`)
- `bmi` (float)
- `children` (int)
- `smoker` (categorical: `no`, `yes`)
- `region` (categorical: e.g., `southwest`, `southeast`, `northwest`, `northeast`)
- `charges` (float, target)

Preprocessing rules mirror `final.py`:
- `sex`: `female` → 0, `male` → 1
- `smoker`: `no` → 0, `yes` → 1
- `region`: one-hot encoded into separate dummy columns at train time

## Reproducing the Script Flow
You can still run the non-UI script for quick experiments:
```bash
python final.py
```

## Troubleshooting
- If the app cannot find the CSV, ensure `medicalInsaurance.csv` is in the project root or upload a file via the sidebar.
- Cross-validation on the test split may fail for small splits; the app will display a note instead.
- If ports are busy, run `streamlit run streamlit_app.py --server.port 8502`.

## License
This project is provided as-is for educational purposes.
