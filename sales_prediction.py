# SALES PREDICTION USING PYTHON
# Project: Sales Prediction Using Advertising Data
# Dataset: Advertising.csv
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.feature_selection import f_regression

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

print("=" * 60)
print("       SALES PREDICTION USING PYTHON")
print("=" * 60)

file_path = "Advertising.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Advertising.csv was not found!\n"
        "Place Advertising.csv in the same folder as "
        "sales_prediction.py."
    )

df = pd.read_csv(file_path)

print("\n1. DATASET LOADED SUCCESSFULLY")
print("-" * 40)

# Remove the unnecessary index column, if present.
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

print("\n2. DATA CLEANING")
print("-" * 40)

print("Missing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Convert all required columns to numeric values.
required_columns = ["TV", "Radio", "Newspaper", "Sales"]

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing or invalid values.
df.dropna(subset=required_columns, inplace=True)

# Remove duplicate records.
df.drop_duplicates(inplace=True)

# Check for impossible negative values in this dataset.
df = df[(df[required_columns] >= 0).all(axis=1)]

df.reset_index(drop=True, inplace=True)

print("\nDataset shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save the cleaned dataset.
df.to_csv("cleaned_advertising_data.csv", index=False)

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

print("\nStatistical summary:")
print(df.describe().round(2))

print("\nCorrelation with Sales:")
print(df.corr(numeric_only=True)["Sales"].sort_values(ascending=False))

# Create output folder for graphs.
os.makedirs("output", exist_ok=True)

# Sales distribution.
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/sales_distribution.png", dpi=300)
plt.close()

# Advertising expenses compared with sales.
for feature in ["TV", "Radio", "Newspaper"]:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=feature,
        y="Sales",
        scatter_kws={"alpha": 0.7}
    )
    plt.title(f"{feature} Advertising vs Sales")
    plt.xlabel(f"{feature} Advertising Spend")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(f"output/{feature.lower()}_vs_sales.png", dpi=300)
    plt.close()

# Correlation heatmap.
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Advertising and Sales")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=300)
plt.close()

print("Exploratory graphs saved in the output folder.")

# ------------------------------------------------------------
# 4. FEATURE SELECTION
# ------------------------------------------------------------

print("\n4. FEATURE SELECTION")
print("-" * 40)

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Calculate F-statistics to assess individual feature relationships.
f_scores, p_values = f_regression(X, y)

feature_results = pd.DataFrame({
    "Feature": X.columns,
    "F_Score": f_scores,
    "P_Value": p_values
})

feature_results.sort_values(
    by="F_Score",
    ascending=False,
    inplace=True
)

print("\nFeature importance based on individual F-tests:")
print(feature_results.round(6).to_string(index=False))

feature_results.to_csv(
    "output/feature_selection_results.csv",
    index=False
)

# ------------------------------------------------------------
# 5. SPLIT THE DATA
# ------------------------------------------------------------

print("\n5. SPLITTING DATA INTO TRAINING AND TESTING SETS")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 6. TRAIN PREDICTION MODELS
# ------------------------------------------------------------

print("\n6. TRAINING MODELS")
print("-" * 40)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42
    )
}

results = []
trained_models = {}

for model_name, model in models.items():

    # Train the model.
    model.fit(X_train, y_train)

    # Predict sales for the test dataset.
    predictions = model.predict(X_test)

    # Evaluate the model.
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })

    trained_models[model_name] = model

    print(f"\nModel: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

# ------------------------------------------------------------
# 7. COMPARE MODELS
# ------------------------------------------------------------

print("\n7. MODEL COMPARISON")
print("-" * 40)

results_df = pd.DataFrame(results)
results_df.sort_values(by="RMSE", inplace=True)

print(results_df.round(4).to_string(index=False))

results_df.to_csv("output/model_comparison.csv", index=False)

# Select the model with the lowest test RMSE.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model based on test RMSE:", best_model_name)

# Plot model performance.
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("Comparison of Models Using RMSE")
plt.ylabel("RMSE (lower is better)")
plt.tight_layout()
plt.savefig("output/model_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 8. CROSS-VALIDATION
# ------------------------------------------------------------

print("\n8. CROSS-VALIDATION")
print("-" * 40)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

cv_rmse = -cv_scores

print("RMSE for each fold:", np.round(cv_rmse, 4))
print(f"Average cross-validation RMSE: {cv_rmse.mean():.4f}")
print(f"Standard deviation: {cv_rmse.std():.4f}")

# ------------------------------------------------------------
# 9. ACTUAL VS PREDICTED SALES
# ------------------------------------------------------------

print("\n9. ACTUAL VS PREDICTED SALES")
print("-" * 40)

test_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual_Sales": y_test.to_numpy(),
    "Predicted_Sales": test_predictions
})

prediction_results["Prediction_Error"] = (
    prediction_results["Actual_Sales"]
    - prediction_results["Predicted_Sales"]
)

print(prediction_results.head(10).round(3).to_string(index=False))

prediction_results.to_csv(
    "output/actual_vs_predicted.csv",
    index=False
)

# Plot actual and predicted values.
plt.figure(figsize=(7, 6))
plt.scatter(y_test, test_predictions, alpha=0.7)

minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales - {best_model_name}")
plt.tight_layout()
plt.savefig("output/actual_vs_predicted.png", dpi=300)
plt.close()

# Residual analysis.
residuals = y_test.to_numpy() - test_predictions

plt.figure(figsize=(8, 5))
plt.scatter(test_predictions, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual Error")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig("output/residual_analysis.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. ANALYZE ADVERTISING IMPACT
# ------------------------------------------------------------

print("\n10. ADVERTISING IMPACT ANALYSIS")
print("-" * 40)

linear_model = trained_models["Linear Regression"]

coefficients = pd.DataFrame({
    "Advertising_Channel": X.columns,
    "Coefficient": linear_model.coef_
})

coefficients.sort_values(
    by="Coefficient",
    ascending=False,
    inplace=True
)

print("\nLinear Regression Coefficients:")
print(coefficients.round(4).to_string(index=False))

print("\nInterpretation:")
print(
    "Each coefficient estimates the change in predicted sales "
    "for a one-unit increase in that advertising channel, "
    "holding the other channels constant."
)
print(
    "These are associations in this dataset and do not by "
    "themselves prove that advertising caused the sales change."
)

coefficients.to_csv(
    "output/advertising_impact.csv",
    index=False
)

# Visualize coefficients.
plt.figure(figsize=(8, 5))
sns.barplot(
    data=coefficients,
    x="Advertising_Channel",
    y="Coefficient"
)
plt.title("Estimated Advertising Impact on Sales")
plt.xlabel("Advertising Channel")
plt.ylabel("Linear Regression Coefficient")
plt.tight_layout()
plt.savefig("output/advertising_impact.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. TEST DIFFERENT ADVERTISING BUDGETS
# ------------------------------------------------------------

print("\n11. ADVERTISING BUDGET SCENARIOS")
print("-" * 40)

average_radio = df["Radio"].mean()
average_newspaper = df["Newspaper"].mean()

tv_values = [
    df["TV"].min(),
    df["TV"].quantile(0.25),
    df["TV"].median(),
    df["TV"].quantile(0.75),
    df["TV"].max()
]

scenarios = pd.DataFrame({
    "TV": tv_values,
    "Radio": [average_radio] * len(tv_values),
    "Newspaper": [average_newspaper] * len(tv_values)
})

scenarios["Predicted_Sales"] = best_model.predict(
    scenarios[["TV", "Radio", "Newspaper"]]
)

print(
    scenarios.round(3).to_string(index=False)
)

scenarios.to_csv(
    "output/advertising_budget_scenarios.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(
    scenarios["TV"],
    scenarios["Predicted_Sales"],
    marker="o"
)
plt.title("Predicted Sales at Different TV Advertising Levels")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Predicted Sales")
plt.tight_layout()
plt.savefig("output/tv_budget_scenarios.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 12. PREDICT SALES FOR A SAMPLE CAMPAIGN
# ------------------------------------------------------------

print("\n12. SAMPLE SALES PREDICTION")
print("-" * 40)

sample_campaign = pd.DataFrame({
    "TV": [200],
    "Radio": [35],
    "Newspaper": [20]
})

sample_prediction = best_model.predict(sample_campaign)[0]

print("Sample advertising budget:")
print(sample_campaign.to_string(index=False))

print(f"\nPredicted sales: {sample_prediction:.2f}")
print(f"Model used: {best_model_name}")

print(
    "\nNote: The prediction uses the same units and scale "
    "as the Sales column in the dataset."
)

# ------------------------------------------------------------
# 13. BUSINESS INSIGHTS
# ------------------------------------------------------------

print("\n13. BUSINESS INSIGHTS")
print("-" * 40)

correlations = df.corr(numeric_only=True)["Sales"].drop("Sales")
strongest_channel = correlations.abs().idxmax()

print(
    f"1. The strongest absolute correlation with sales is "
    f"{strongest_channel} "
    f"(correlation = {correlations[strongest_channel]:.3f})."
)

print(
    f"2. The best-performing model by test RMSE is "
    f"{best_model_name}."
)

print(
    f"3. Its test RMSE is "
    f"{results_df.iloc[0]['RMSE']:.3f}."
)

print(
    "4. Use the advertising impact analysis and budget scenarios "
    "to compare possible budget allocations."
)

print(
    "5. Validate these findings on newer campaign data before "
    "making major marketing budget decisions."
)

print(
    "\nDataset limitation: Advertising.csv contains TV, Radio, "
    "Newspaper, and Sales columns. It does not contain dates, "
    "platform names beyond these channels, or target segments."
)

print(
    "Therefore, this project performs regression-based sales "
    "prediction, not time-series forecasting or segment analysis."
)

# ------------------------------------------------------------
# 14. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("- cleaned_advertising_data.csv")
print("- output/model_comparison.csv")
print("- output/feature_selection_results.csv")
print("- output/advertising_impact.csv")
print("- output/advertising_budget_scenarios.csv")
print("- output/actual_vs_predicted.csv")
print("- Graphs saved inside the output folder")

print("\nCheck the output folder for your results and graphs.")