# Design and Methodology

This project reverse-engineers a legacy travel reimbursement system by employing a two-stage, data-driven modeling approach. The core strategy is to use a "model factory" (`discover_rules.py`) to find the optimal prediction model, which is then serialized and used by a lightweight execution engine (`calculate.py`).

Our final model achieved a score of **762.10**.

---

## 1. Core Strategy: The Model Factory Pattern

The solution is split into two distinct components:

-   **`discover_rules.py` (The Factory):** This script is responsible for all heavy lifting. It performs feature engineering, hyperparameter optimization, and model training. Its sole purpose is to analyze the `public_cases.json` data and produce the best possible "blueprint" for the legacy system's logic.
-   **`calculate.py` (The Engine):** This script is a lean, fast execution engine. It does no training. It loads a pre-built model state (`model_state.pkl`) and uses it to calculate a reimbursement amount for a given input. This separation ensures that the final prediction script is fast and has minimal dependencies, as required by the project constraints.
-   **`model_state.pkl` (The Blueprint):** This file acts as the bridge between the factory and the engine. It contains all the necessary components for prediction: trained scalers, clusterers, regression models, and a list of feature names. We use `joblib` for efficient serialization of scikit-learn objects.

## 2. Modeling Technique: Surrogate Modeling with Localized Regression

The core of our approach is a hybrid model designed to capture the complex, non-linear, and conditional rules of the legacy system.

1.  **The Oracle:** We first train a `GradientBoostingRegressor` on the dataset. This model acts as an "oracle"—it is powerful enough to capture the intricate patterns and noise in the data, generating a highly accurate set of predictions. However, it is a "black box" and too complex to be directly interpretable.

2.  **The Surrogate (Clustering and Segmentation):** To approximate the oracle's logic in a more structured way, we use a `DecisionTreeRegressor` as a surrogate model. The decision tree's primary role is not to predict the final value directly, but to partition the data into distinct, interpretable segments (its leaves). Each leaf represents a unique *type* of trip (e.g., "short trips with low mileage" or "long, high-expense trips"). K-Means clustering features are also provided to the tree to help it find more robust segments.

3.  **Localized Formulas:** Instead of a single, global formula, we fit a separate, simple regression model (`PolynomialFeatures` + `Ridge`) *within each leaf* of the decision tree. This allows the model to learn simple, localized rules that only apply to that specific trip segment. This hybrid approach combines the partitioning strength of decision trees with the classic modeling power of regression, allowing for high accuracy without overfitting to a single complex formula.

## 3. Iterative Feature Engineering

The features used in the model were developed iteratively, driven by insights from the employee interviews and analysis of high-error cases from the `eval.sh` script.

-   **Interview-Driven Features:**
    -   `miles_tier1`, `miles_tier2`, `miles_tier3`: Based on Lisa's comment about tiered mileage rates.
    -   `is_5_day_trip`, `is_sweet_spot_trip`: Based on comments about 5-day trips having special bonuses.
    -   `has_magic_cents`: Based on Lisa's observation about receipt amounts ending in `.49` or `.99`.

-   **Error-Driven Features:**
    -   `is_extreme_travel_day`: After observing massive errors on trips with impossibly high miles per day (e.g., >800), this feature was added to isolate these outliers and allow the model to treat them as a special case. This was a critical surgical improvement that significantly boosted the score.

## 4. Hyperparameter Optimization with Optuna

We used `Optuna` for comprehensive hyperparameter tuning. This was not limited to standard model parameters like `max_depth` but also included structural parameters like the optimal `n_clusters` for the K-Means algorithm. Our best score was achieved after extending the search from 50 to 100 trials, which allowed Optuna to discover a more refined set of parameters and produce a model with a validation score of **7.01**.

---

# Top Coder Challenge: Black Box Legacy Reimbursement System

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.

8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.

## What You Have

### Input Parameters

The system takes three inputs:

- `trip_duration_days` - Number of days spent traveling (integer)
- `miles_traveled` - Total miles traveled (integer)
- `total_receipts_amount` - Total dollar amount of receipts (float)

## Documentation

- A PRD (Product Requirements Document)
- Employee interviews with system hints

### Output

- Single numeric reimbursement amount (float, rounded to 2 decimal places)

### Historical Data

- `public_cases.json` - 1,000 historical input/output examples

## Getting Started

1. **Analyze the data**: 
   - Look at `public_cases.json` to understand patterns
   - Look at `PRD.md` to understand the business problem
   - Look at `INTERVIEWS.md` to understand the business logic
2. **Create your implementation**:
   - Copy `run.sh.template` to `run.sh`
   - Implement your calculation logic
   - Make sure it outputs just the reimbursement amount
3. **Test your solution**: 
   - Run `./eval.sh` to see how you're doing
   - Use the feedback to improve your algorithm
4. **Submit**:
   - Run `./generate_results.sh` to get your final results.
   - Add `arjun-krishna1` to your repo.
   - Complete [the submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).

## Implementation Requirements

Your `run.sh` script must:

- Take exactly 3 parameters: `trip_duration_days`, `miles_traveled`, `total_receipts_amount`
- Output a single number (the reimbursement amount)
- Run in under 5 seconds per test case
- Work without external dependencies (no network calls, databases, etc.)

Example:

```bash
./run.sh 5 250 150.75
# Should output something like: 487.25
```

## Evaluation

Run `./eval.sh` to test your solution against all 1,000 cases. The script will show:

- **Exact matches**: Cases within ±$0.01 of the expected output
- **Close matches**: Cases within ±$1.00 of the expected output
- **Average error**: Mean absolute difference from expected outputs
- **Score**: Lower is better (combines accuracy and precision)

Your submission will be tested against `private_cases.json` which does not include the outputs.

## Submission

When you're ready to submit:

1. Push your solution to a GitHub repository
2. Add `arjun-krishna1` to your repository
3. Submit via the [submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).
4. When you submit the form you will submit your `private_results.txt` which will be used for your final score.

---

**Good luck and Bon Voyage!**
