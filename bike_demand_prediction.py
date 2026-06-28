import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
def load_and_process_data(filepath='train.csv'):
    print(f"Attempting to load '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        print(f" -> Success! Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: '{filepath}' not found.")
        print("Make sure 'train.csv' is in the same folder as this script.")
        return None, None

    # Feature Engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Prevent Leakage
    cols_to_drop = ['casual', 'registered', 'datetime']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    X = df_clean.drop(columns=['count'])
    y = df_clean['count']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# ==========================================
# 2. Model Evaluation Helper (UPDATED)
# ==========================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2 : {r2:.4f}")
    print("-" * 40)
    
    # Return predictions so we can plot them later
    return y_test, y_pred, r2

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    X, y = load_and_process_data()
    if X is None: return

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n=== Question 1 Results: Bike Sharing Demand ===\n")
    
    # Store results for plotting
    plot_data = []

    # --- 1. Linear Regression ---
    y_true, y_pred, r2 = evaluate_model("Linear Regression", 
                                        LinearRegression(), X_train, X_test, y_train, y_test)
    plot_data.append(("Linear Regression", y_true, y_pred, r2))

    # --- 2. Polynomial (d=2, 3, 4) ---
    for d in [2, 3, 4]:
        X_train_poly = np.hstack([X_train**(p) for p in range(1, d+1)])
        X_test_poly  = np.hstack([X_test**(p) for p in range(1, d+1)])
        
        y_true, y_pred, r2 = evaluate_model(f"Polynomial (d={d})", 
                                            LinearRegression(), X_train_poly, X_test_poly, y_train, y_test)
        plot_data.append((f"Polynomial (d={d})", y_true, y_pred, r2))

    # --- 3. Quadratic Interaction ---
    poly_inter = PolynomialFeatures(degree=2, include_bias=False)
    X_train_quad = poly_inter.fit_transform(X_train)
    X_test_quad = poly_inter.transform(X_test)
    
    y_true, y_pred, r2 = evaluate_model("Quadratic (Interaction)", 
                                        LinearRegression(), X_train_quad, X_test_quad, y_train, y_test)
    plot_data.append(("Quadratic (Interaction)", y_true, y_pred, r2))

    # ==========================================
    # 4. PLOTTING THE 5 GRAPHS     # ==========================================
    print("Generating graphs...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # 2 rows, 3 columns
    axes = axes.flatten() # Flatten to easy list of 6 axes

    for i, (name, y_t, y_p, r2_score_val) in enumerate(plot_data):
        ax = axes[i]
        
        # Scatter Plot of Actual vs Predicted
        ax.scatter(y_t, y_p, alpha=0.3, s=10, color='blue', label='Predictions')
        
        # Red dashed line for "Perfect Prediction" (y = x)
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
        
        ax.set_title(f"{name}\nR2: {r2_score_val:.4f}")
        ax.set_xlabel("Actual Demand")
        ax.set_ylabel("Predicted Demand")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide the 6th empty graph (since we only have 5 models)
    fig.delaxes(axes[5]) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()