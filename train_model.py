import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_pipeline(categorical_features, numeric_features):
    # Preprocessing: OneHotEncode categoricals, scale numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Full pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    return model

def main(data_path, out_dir):
    # Load dataset
    df = pd.read_csv(data_path)

    # Target column
    target = "HeartDisease"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify feature types
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    model = build_pipeline(categorical_features, numeric_features)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained with accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    print(f"✅ Model saved to {out_dir}/model.joblib")

    # Save metadata (with feature names!)
    metadata = {
        "feature_names": X.columns.tolist(),
        "target": target
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print(f"✅ Metadata saved to {out_dir}/metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for model + metadata")
    args = parser.parse_args()

    main(args.data, args.out_dir)


