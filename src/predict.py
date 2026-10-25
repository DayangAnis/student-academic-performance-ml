"""Generate exam-score predictions from the saved fitted pipeline."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of new student rows (predictor columns only)")
    parser.add_argument("--output", default="artifacts/predictions.csv", help="Output CSV path")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "best_model.joblib"))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}. Run python -m src.train_model first.")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    features = pd.read_csv(input_path)
    model = joblib.load(model_path)
    predictions = model.predict(features)

    result = features.copy()
    result["Predicted_Exam_Score"] = predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
