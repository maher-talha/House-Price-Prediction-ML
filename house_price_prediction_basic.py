# Import libraries (tools we need)
import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("All libraries loaded successfully!")
print("=" * 50)

# STEP 1: CREATE SAMPLE DATA AND SAVE AS CSV

print("\nSTEP 1: Creating sample housing data...")

# We are creating fake but realistic house data
# Think of this like filling an Excel sheet manually

np.random.seed(42)  # this makes sure we get same data every time
n = 100             # we will create 100 houses

# Create each column one by one
area      = np.random.randint(500, 3000, n)      # area in sq ft
bedrooms  = np.random.randint(1, 6, n)           # number of bedrooms
bathrooms = np.random.randint(1, 4, n)           # number of bathrooms
age       = np.random.randint(0, 40, n)          # age of house in years

# Price formula: area and rooms affect price
# We add some random noise to make it realistic
price = (
    150 * area
    + 10000 * bedrooms
    + 8000  * bathrooms
    - 500   * age
    + np.random.randint(-20000, 20000, n)  # random noise
)

# Make sure no price goes below 50,000
price = np.clip(price, 50000, 9999999)

# Put everything into a table (DataFrame = like Excel table)
data = pd.DataFrame({
    "Area_sqft" : area,
    "Bedrooms"  : bedrooms,
    "Bathrooms" : bathrooms,
    "House_Age" : age,
    "Price"     : price
})

# Add some missing values on purpose (to practice cleaning)
data.loc[5,  "Area_sqft"] = None
data.loc[15, "Bedrooms"]  = None
data.loc[30, "Area_sqft"] = None

# Save to CSV file
data.to_csv("house_data.csv", index=False)
print("Data saved to house_data.csv")
print(f"Total rows: {len(data)}, Total columns: {len(data.columns)}")
print("\nFirst 5 rows of data:")
print(data.head())



# STEP 2: LOAD AND CLEAN DATA

print("\n" + "=" * 50)
print("STEP 2: Loading and Cleaning Data...")

# Load the CSV file we just saved
df = pd.read_csv("house_data.csv")

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fix missing values by filling with the average (median)
df["Area_sqft"] = df["Area_sqft"].fillna(df["Area_sqft"].median())
df["Bedrooms"]  = df["Bedrooms"].fillna(df["Bedrooms"].median())

print("\nAfter cleaning - Missing values:")
print(df.isnull().sum())
print("Data is clean now!")

# STEP 3: CHOOSE FEATURES AND TARGET

print("\n" + "=" * 50)
print("STEP 3: Selecting Features...")

# Features = inputs we give to the model (X)
# Target   = what we want to predict (y)

X = df[["Area_sqft", "Bedrooms", "Bathrooms", "House_Age"]]  # inputs
y = df["Price"]                                               # output

print("Input features (X):")
print(X.head())
print("\nTarget (y) - first 5 prices:")
print(y.head().values)

# Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")



# STEP 4: EXPLORATORY DATA ANALYSIS (EDA) - CHARTS

print("\n" + "=" * 50)
print("STEP 4: Drawing Charts (EDA)...")

# --- Chart 1: Histogram - How is Price distributed? ---
plt.figure(figsize=(6, 4))
plt.hist(df["Price"] / 1000, bins=20, color="steelblue", edgecolor="black")
plt.title("How Many Houses in Each Price Range?")
plt.xlabel("Price (in $1000s)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("chart1_price_histogram.png")
plt.close()
print("Saved: chart1_price_histogram.png")

# --- Chart 2: Scatter - Does bigger area = higher price? ---
plt.figure(figsize=(6, 4))
plt.scatter(df["Area_sqft"], df["Price"] / 1000, color="coral", alpha=0.6)
plt.title("Does Bigger Area Mean Higher Price?")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in $1000s)")
plt.tight_layout()
plt.savefig("chart2_area_vs_price.png")
plt.close()
print("Saved: chart2_area_vs_price.png")

# --- Chart 3: Scatter - Does more bedrooms = higher price? ---
plt.figure(figsize=(6, 4))
plt.scatter(df["Bedrooms"], df["Price"] / 1000, color="green", alpha=0.6)
plt.title("Do More Bedrooms Mean Higher Price?")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (in $1000s)")
plt.tight_layout()
plt.savefig("chart3_bedrooms_vs_price.png")
plt.close()
print("Saved: chart3_bedrooms_vs_price.png")

# --- Chart 4: Scatter - Does old house = lower price? ---
plt.figure(figsize=(6, 4))
plt.scatter(df["House_Age"], df["Price"] / 1000, color="purple", alpha=0.6)
plt.title("Does Older House Mean Lower Price?")
plt.xlabel("House Age (years)")
plt.ylabel("Price (in $1000s)")
plt.tight_layout()
plt.savefig("chart4_age_vs_price.png")
plt.close()
print("Saved: chart4_age_vs_price.png")

# --- Chart 5: Correlation - Which features are related to price? ---
correlation = df.corr()
print("\nCorrelation with Price:")
print(correlation["Price"].sort_values(ascending=False))

plt.figure(figsize=(6, 4))
corr_values = correlation["Price"].drop("Price")
colors = ["green" if v > 0 else "red" for v in corr_values]
plt.bar(corr_values.index, corr_values.values, color=colors, edgecolor="black")
plt.title("Which Features Are Most Related to Price?")
plt.xlabel("Feature")
plt.ylabel("Correlation Score (-1 to +1)")
plt.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("chart5_correlation.png")
plt.close()
print("Saved: chart5_correlation.png")

# STEP 5: TRAIN THE MODEL

print("\n" + "=" * 50)
print("STEP 5: Training Linear Regression Model...")

# Create the model
model = LinearRegression()

# Train it - model learns from training data
model.fit(X_train, y_train)

print("Model trained successfully!")


# STEP 6: EVALUATE THE MODEL

print("\n" + "=" * 50)
print("STEP 6: Checking How Good Our Model Is...")

# Use the model to predict prices on test data
y_pred = model.predict(X_test)

# Calculate error metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"""
  MAE  (Mean Absolute Error)  = ${mae:,.0f}
       → On average, our prediction is off by ${mae:,.0f}

  MSE  (Mean Squared Error)   = ${mse:,.0f}
       → Big errors are punished more heavily here

  RMSE (Root Mean Sq. Error)  = ${rmse:,.0f}
       → Similar to MAE but penalizes big mistakes

  R²   (R-Squared Score)      = {r2:.2f}
       → Our model explains {r2*100:.1f}% of price variation
       → Closer to 1.0 = better model
""")


print("STEP 7: Drawing Prediction Charts...")

#  Chart 6: Actual vs Predicted prices 
plt.figure(figsize=(6, 4))
plt.scatter(y_test / 1000, y_pred / 1000, color="steelblue", alpha=0.7)
# Draw a perfect line (if prediction = actual, dots should be on this line)
min_val = min(y_test.min(), y_pred.min()) / 1000
max_val = max(y_test.max(), y_pred.max()) / 1000
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.legend()
plt.tight_layout()
plt.savefig("chart6_actual_vs_predicted.png")
plt.close()
print("Saved: chart6_actual_vs_predicted.png")

#  Chart 7: Errors (Residuals) 
errors = y_test.values - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors / 1000, bins=15, color="orange", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", label="Zero Error")
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error Amount ($1000s)")
plt.ylabel("Number of Houses")
plt.legend()
plt.tight_layout()
plt.savefig("chart7_errors.png")
plt.close()
print("Saved: chart7_errors.png")


print("\n" + "=" * 50)
print("STEP 8: What Did the Model Learn? (Coefficients)")

print(f"\n  Base Price (Intercept) = ${model.intercept_:,.0f}")
print("  (This is the starting price before adding features)\n")

feature_names = ["Area_sqft", "Bedrooms", "Bathrooms", "House_Age"]
print(f"  {'Feature':<15} {'Effect on Price':>18}  Meaning")
print("  " + "-" * 60)
for name, coef in zip(feature_names, model.coef_):
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {name:<15} ${coef:>15,.0f}  → price {direction} by this much per unit")

# --- Chart 8: Coefficients Bar Chart ---
plt.figure(figsize=(6, 4))
coefs = model.coef_
bar_colors = ["green" if c > 0 else "red" for c in coefs]
plt.bar(feature_names, coefs, color=bar_colors, edgecolor="black")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("How Much Each Feature Affects Price")
plt.xlabel("Feature")
plt.ylabel("Effect on Price ($)")
plt.tight_layout()
plt.savefig("chart8_coefficients.png")
plt.close()
print("\nSaved: chart8_coefficients.png")


print("\n" + "=" * 50)
print("PROJECT COMPLETE! Summary:")
print("=" * 50)
print(f"""
  Dataset      : 100 houses with 4 features
  Train/Test   : 80 houses to train, 20 to test

  Model Score:
  - MAE  = ${mae:,.0f}   (average error per house)
  - RMSE = ${rmse:,.0f}   (error with big mistake penalty)
  - R²   = {r2:.2f}        (model accuracy - higher is better)

  What the model learned:
  - Bigger area  → Higher price  (makes sense!)
  - More rooms   → Higher price  (makes sense!)
  - Older house  → Lower price   (makes sense!)

  Charts saved:
  chart1 - Price histogram
  chart2 - Area vs Price
  chart3 - Bedrooms vs Price
  chart4 - House Age vs Price
  chart5 - Correlation bar chart
  chart6 - Actual vs Predicted
  chart7 - Prediction Errors
  chart8 - Feature Coefficients
""")
