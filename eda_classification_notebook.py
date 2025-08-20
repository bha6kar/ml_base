import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# -------------------------
# 1. Generate synthetic dataset
# -------------------------
np.random.seed(42)
n = 300
df = pd.DataFrame({
    "id": range(1, n+1),
    "age": np.random.randint(18, 70, n),
    "income": np.random.randint(20000, 120000, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "purchases": np.random.poisson(5, n),
    "churn": np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# Save CSV version too
df.to_csv("sample_classification_data.csv", index=False)
print("Dataset saved as sample_classification_data.csv")

# -------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isna().sum())

print("\n--- Target Balance (churn) ---")
print(df["churn"].value_counts(normalize=True))

print("\n--- Numeric Summary ---")
print(df.describe())

# Histograms for numeric features
numeric_cols = ["age", "income", "purchases"]
for col in numeric_cols:
    plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Bar plot for a categorical feature
cat_col = "region"
counts = df[cat_col].value_counts()
plt.bar(counts.index, counts.values)
plt.title(f"Category counts: {cat_col}")
plt.show()

# -------------------------
# 3. Preprocessing + Model Training
# -------------------------
X = df.drop(columns=["churn", "id"])
y = df["churn"]

numeric_features = ["age", "income", "purchases"]
categorical_features = ["gender", "region"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Choose model (try Logistic Regression or Random Forest)
model = RandomForestClassifier(n_estimators=200, random_state=42)
pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
print("\n--- CV Scores (accuracy) ---")
print(cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Fit and evaluate
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("\n--- Test Set Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha="center", va="center", color="red")
plt.colorbar()
plt.show()
