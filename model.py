import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import argparse

# --- Argument Parser for Dataset Path ---
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/salary_data.csv', help='Path to dataset')
args = parser.parse_args()

# --- Step 1: Load Dataset ---
df = pd.read_csv(args.data)
print("📄 Dataset Preview:")
print(df.head())

X = df.drop(columns=['salary'])
y = df['salary']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- Step 2: Define Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# --- Step 3: Define Models and Param Grids ---
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 10],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }
    }
}

# --- Step 4: Train, Evaluate, and Select Best Model ---
best_model = None
best_score = -np.inf
best_name = ''

for name, config in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', config['model'])
    ])

    grid = GridSearchCV(pipe, config['params'], cv=5, scoring='r2')
    grid.fit(X, y)

    cv_score = np.mean(cross_val_score(grid.best_estimator_, X, y, cv=5, scoring='r2'))
    print(f"✅ {name} CV R² Score: {cv_score:.2f}")
    print(f"🔹 Best Hyperparameters for {name}: {grid.best_params_}")

    if cv_score > best_score:
        best_score = cv_score
        best_model = grid.best_estimator_
        best_name = name

# --- Step 5: Save the Best Model ---
print(f"\n✅ Best Model Selected: {best_name} with Cross-Validated R² Score: {best_score:.2f}")
joblib.dump(best_model, 'model.pkl')
print("✅ Best Model saved as model.pkl — Ready for app integration!")
# --- Step 6: Save Preprocessor ---
joblib.dump(preprocessor, 'preprocessor.pkl')   