import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import numpy as np
import optuna
import sys
import joblib

# --- Data Loading and Preparation ---
def load_and_prepare_data(path='public_cases.json'):
    try:
        with open(path, 'r') as f:
            raw_data = pd.read_json(f)
    except FileNotFoundError:
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.json_normalize(raw_data.to_dict(orient='records'))
    df.rename(columns={
        'input.trip_duration_days': 'trip_duration_days',
        'input.miles_traveled': 'miles_traveled',
        'input.total_receipts_amount': 'total_receipts_amount',
        'expected_output': 'reimbursement_amount'
    }, inplace=True)
    return df

def feature_engineer(df):
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
    
    for col in ['miles_per_day', 'receipts_per_day']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[f'is_{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    return df

def objective(trial, df):
    n_clusters = trial.suggest_int('n_clusters', 4, 12)
    oracle_n_estimators = trial.suggest_int('oracle_n_estimators', 100, 300, step=50)
    oracle_max_depth = trial.suggest_int('oracle_max_depth', 4, 8)
    oracle_learning_rate = trial.suggest_float('oracle_learning_rate', 0.01, 0.2)
    surrogate_max_depth = trial.suggest_int('surrogate_max_depth', 4, 10)
    surrogate_min_samples_leaf = trial.suggest_int('surrogate_min_samples_leaf', 5, 50)
    
    df_trial = df.copy()
    clustering_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipts_per_day']
    X_for_clustering = df_trial[clustering_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_for_clustering)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distances = kmeans.transform(X_scaled)
    for i in range(n_clusters):
        df_trial[f'dist_to_cluster_{i}'] = distances[:, i]

    features = [col for col in df_trial.columns if col not in ['reimbursement_amount']]
    X = df_trial[features]
    y_true = df_trial['reimbursement_amount']
    
    oracle_model = GradientBoostingRegressor(
        n_estimators=oracle_n_estimators, max_depth=oracle_max_depth,
        learning_rate=oracle_learning_rate, loss='huber', random_state=42
    )
    oracle_model.fit(X, y_true)
    y_oracle = oracle_model.predict(X)

    surrogate_tree = DecisionTreeRegressor(
        max_depth=surrogate_max_depth, min_samples_leaf=surrogate_min_samples_leaf, random_state=42
    )
    surrogate_tree.fit(X, y_oracle)

    final_formula_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_tier1', 'miles_tier2', 'miles_tier3']
    
    leaf_ids = surrogate_tree.apply(X)
    df_trial['leaf_id'] = leaf_ids
    df_trial['prediction'] = 0.0

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = df_trial['leaf_id'] == leaf_id
        leaf_df = df_trial[leaf_mask]
        
        if len(leaf_df) > len(final_formula_features) + 1:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            model = make_pipeline(poly, Ridge(alpha=0.5))
            model.fit(leaf_df[final_formula_features], leaf_df['reimbursement_amount'])
            prediction = model.predict(leaf_df[final_formula_features])
        else:
            prediction = leaf_df['reimbursement_amount'].mean()
        
        df_trial.loc[leaf_mask, 'prediction'] = prediction
        
    return mean_absolute_error(y_true, df_trial['prediction'])

def run_discovery():
    df_base = load_and_prepare_data()
    df_engineered = feature_engineer(df_base)

    print("--- Starting Hyperparameter Optimization ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, df_engineered), n_trials=25, show_progress_bar=True)
    
    best_params = study.best_params
    print("\n--- Best Hyperparameters Found ---")
    print(best_params)

    print("\n--- Generating Final Model with Best Hyperparameters ---")
    
    n_clusters = best_params['n_clusters']
    
    clustering_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipts_per_day']
    X_for_clustering = df_engineered[clustering_features]
    scaler = StandardScaler().fit(X_for_clustering)
    X_scaled = scaler.transform(X_for_clustering)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
    distances = kmeans.transform(X_scaled)
    for i in range(n_clusters):
        df_engineered[f'dist_to_cluster_{i}'] = distances[:, i]

    model_features = [col for col in df_engineered.columns if col not in ['reimbursement_amount', 'y_oracle']]
    X = df_engineered[model_features]
    y_true = df_engineered['reimbursement_amount']
    
    oracle_model = GradientBoostingRegressor(
        n_estimators=best_params['oracle_n_estimators'], max_depth=best_params['oracle_max_depth'],
        learning_rate=best_params['oracle_learning_rate'], loss='huber', random_state=42
    ).fit(X, y_true)
    y_oracle = oracle_model.predict(X)

    surrogate_tree = DecisionTreeRegressor(
        max_depth=best_params['surrogate_max_depth'], min_samples_leaf=best_params['surrogate_min_samples_leaf'],
        random_state=42
    ).fit(X, y_oracle)
    
    leaf_ids = surrogate_tree.apply(X)
    formulas = {}
    final_formula_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_tier1', 'miles_tier2', 'miles_tier3']

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf_id
        leaf_df = df_engineered[leaf_mask]
        
        if len(leaf_df) > len(final_formula_features) + 1:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            model = make_pipeline(poly, Ridge(alpha=0.5))
            model.fit(leaf_df[final_formula_features], leaf_df['reimbursement_amount'])
            formulas[leaf_id] = model
        else:
            formulas[leaf_id] = leaf_df['reimbursement_amount'].mean()

    model_state = {
        'scaler': scaler,
        'kmeans': kmeans,
        'surrogate_tree': surrogate_tree,
        'formulas': formulas,
        'model_features': model_features,
        'final_formula_features': final_formula_features
    }

    print("\n--- Generating model_state.pkl ---")
    joblib.dump(model_state, 'model_state.pkl')
    print("--- model_state.pkl generated successfully. ---")

if __name__ == "__main__":
    run_discovery() 