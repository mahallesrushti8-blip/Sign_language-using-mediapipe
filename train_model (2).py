"""
STEP 2: TRAIN THE MODEL
-------------------------------------------------
Reads landmarks.csv (produced by collect_data.py) and trains a
Random Forest classifier to recognize your custom signs from
hand-landmark coordinates. Saves the trained model to model.pkl.

Run: python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

CSV_PATH = "landmarks.csv"
MODEL_PATH = "model.pkl"

# Load the data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} gestures:")
print(df['label'].value_counts(), "\n")

X = df.drop("label", axis=1)   # the 63 landmark coordinates
y = df["label"]                # the gesture name

# Every gesture needs enough samples to be split into train/test,
# and enough total data for the model to actually learn patterns from.
MIN_PER_CLASS = 30  # bare minimum to run; 150-300 is recommended for good accuracy
counts = y.value_counts()
too_few = counts[counts < MIN_PER_CLASS]
if len(too_few) > 0:
    print(f"ERROR: these gestures have fewer than {MIN_PER_CLASS} samples:")
    print(too_few)
    print("\nCurrent counts for all gestures:")
    print(counts)
    print(f"\nGo back to collect_data.py and record more samples "
          f"(aim for 150-300 each), then rerun this script.")
    raise SystemExit(1)

# Split into training and test sets so we can check accuracy honestly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest works well here: fast, needs little tuning, handles
# this kind of small tabular numeric data nicely.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy on held-out test data:")
print(classification_report(y_test, y_pred))

# Save the trained model to disk
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
