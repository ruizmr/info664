import sys
import joblib
import pandas as pd
import numpy as np

def calculate_reimbursement(days, miles, receipts, model_state):
    """
    Calculates the reimbursement amount using a pre-trained model state.
    """
    scaler = model_state['scaler']
    kmeans = model_state['kmeans']
    surrogate_tree = model_state['surrogate_tree']
    formulas = model_state['formulas']
    model_features = model_state['model_features']
    final_formula_features = model_state['final_formula_features']

    data = {'trip_duration_days': [days], 'miles_traveled': [miles], 'total_receipts_amount': [receipts]}
    df = pd.DataFrame(data)

    # Feature Engineering
    df['trip_duration_days_safe'] = df['trip_duration_days'].replace(0, 1)
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days_safe']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days_safe']
    df['days_x_miles'] = df['trip_duration_days'] * df['miles_traveled']
    df['receipts_x_days'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['days_sq'] = df['trip_duration_days']**2
    df['miles_sq'] = df['miles_traveled']**2
    df['receipts_sq'] = df['total_receipts_amount']**2
    df['is_short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['is_long_trip'] = (df['trip_duration_days'] > 7).astype(int)
    df['is_high_mileage'] = (df['miles_traveled'] > 600).astype(int)
    df['is_low_mileage'] = (df['miles_traveled'] < 100).astype(int)
    df['is_4_to_6_day_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['has_magic_cents'] = (((df['total_receipts_amount'] * 100).astype(int) % 100 == 49) | ((df['total_receipts_amount'] * 100).astype(int) % 100 == 99)).astype(int)
    df['miles_tier1'] = df['miles_traveled'].clip(upper=100)
    df['miles_tier2'] = df['miles_traveled'].clip(lower=100, upper=600) - 100
    df['miles_tier3'] = df['miles_traveled'].clip(lower=600) - 600
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_sweet_spot_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['is_in_efficiency_sweet_spot'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['short_trip_overspending_amount'] = np.maximum(0, df['receipts_per_day'] - 75)
    df['medium_trip_overspending_amount'] = np.maximum(0, df['receipts_per_day'] - 120)
    df['long_trip_overspending_amount'] = np.maximum(0, df['receipts_per_day'] - 90)
    df['is_extreme_travel_day'] = (df['miles_per_day'] > 800).astype(int)
    df['is_miles_per_day_outlier'] = 0
    df['is_receipts_per_day_outlier'] = 0
    
    # Clustering
    clustering_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipts_per_day']
    X_cluster = df[clustering_cols]
    X_scaled = scaler.transform(X_cluster)
    distances = kmeans.transform(X_scaled)
    for i in range(kmeans.n_clusters):
        df[f'dist_to_cluster_{i}'] = distances[:, i]

    # Predict leaf
    # Ensure all model features are present, adding dummy columns if needed
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0
    X_tree = df[model_features]
    leaf_id = surrogate_tree.apply(X_tree)[0]
    
    # Apply formula
    formula = formulas.get(leaf_id)
    if isinstance(formula, float) or isinstance(formula, np.float64):
        return formula
    else: # It's a model pipeline
        # Ensure all formula features are present
        for feature in final_formula_features:
            if feature not in df.columns:
                df[feature] = 0
        prediction = formula.predict(df[final_formula_features])
        return prediction[0]

if __name__ == "__main__":
    try:
        model = joblib.load('model_state.pkl')
    except FileNotFoundError:
        print("Error: model_state.pkl not found. Run discover_rules.py first.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)

    try:
        duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
    except ValueError:
        print("Error: Invalid input types. Please provide numeric values.", file=sys.stderr)
        sys.exit(1)

    result = calculate_reimbursement(duration, miles, receipts, model)
    print(f"{result:.2f}") 