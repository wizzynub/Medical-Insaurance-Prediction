import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('medicalInsaurance.csv')
df.head()
for i in df.columns:
    if i in ['age', 'bmi', 'children', 'charges']:
        continue
    print(df[i].unique())
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
dummies_region = pd.get_dummies(df['region'], dtype=int)
print(dummies_region.head(5))

df = pd.concat([df, dummies_region], axis=1)
df.drop('region', axis=1, inplace=True)
df.head()
X = df.drop('charges', axis = 1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# Initialize the scaler only for features
scaler_X = StandardScaler()

# Scale features (X) - fit only on training data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# For Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R²: {grid_search.best_score_:.4f}")


# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_split=10, max_depth=10)
rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Random Forest R²: {rf_scores.mean():.4f}")
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate
score = rf.score(X_test_scaled, y_test)
print(f"R² score: {score}")
rf.score(X_test_scaled, y_test)