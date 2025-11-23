from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. Prepare X (Features) and y (Target)
target_col = "Visual_Outcome"  # 0 or 1
id_col = "Case_ID"

# Drop non-feature columns
X = df_master.drop(columns=[target_col, id_col])

# Handle categorical data (like Sex) if present
X = pd.get_dummies(X, drop_first=True)

y = df_master[target_col]

# 2. Standardization (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Keep column names for interpretation later
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42
)

# 4. Run LASSO with Cross-Validation (LassoCV)
# LassoCV automatically finds the best alpha (regularization strength)
print("Fitting LASSO...")
lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_train, y_train)

print(f"Best Alpha found: {lasso.alpha_}")

# 5. Interpret Results (Feature Selection)
# Get coefficients
coefs = pd.Series(lasso.coef_, index=X.columns)

# Filter out features that became Zero (Selected Features)
selected_features = coefs[coefs != 0].sort_values(ascending=False)

print(f"\nLASSO selected {len(selected_features)} features out of {X.shape[1]}:")
print(selected_features)

# 6. Calculate "Rad-Score" for Test Set
# The score is the linear combination: w1*x1 + w2*x2 ...
rad_scores = lasso.predict(X_test)

# Evaluate
auc = roc_auc_score(y_test, rad_scores)
print(f"Model AUC on Test Set: {auc:.3f}")
