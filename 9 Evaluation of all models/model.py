import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import UnifiedSegmentedModel if available
try:
    from model import UnifiedSegmentedModel
    unified_model_available = True
    print("Successfully imported UnifiedSegmentedModel class")
except ImportError:
    unified_model_available = False
    print("UnifiedSegmentedModel not available, will skip this model")

# Create output directory
os.makedirs('test_results', exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure logging
log_file = open(f'test_results/model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w')

def log(message):
    """Log message to both console and file"""
    print(message)
    log_file.write(f"{message}\n")
    log_file.flush()

def load_data(file_path):
    """Load and do basic preprocessing on the data"""
    log(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    log(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Remove rows with missing target values
    missing_in_target = df['SettlementValue'].isna().sum()
    if missing_in_target > 0:
        log(f"Removing {missing_in_target} rows with missing SettlementValue")
        df = df.dropna(subset=['SettlementValue'])
    
    # Filter out entries with "No_Other side drove on wrong side of the road"
    initial_count = len(df)
    
    # Check which column might contain this text
    found_in_column = None
    wrong_side_entries = 0
    
    # Look for the text in possible columns
    potential_columns = ['Exceptional_Circumstances', 'AccidentType', 'Accident Description']
    for col in potential_columns:
        if col in df.columns:
            mask = df[col].astype(str).str.contains('No_Other side drove on wrong side of the road', case=False, na=False)
            count = mask.sum()
            if count > 0:
                found_in_column = col
                wrong_side_entries = count
                break
    
    # If found, filter out these entries
    if found_in_column and wrong_side_entries > 0:
        df = df[~df[found_in_column].astype(str).str.contains('No_Other side drove on wrong side of the road', case=False, na=False)]
        log(f"Removed {wrong_side_entries} entries containing 'No_Other side drove on wrong side of the road' from column '{found_in_column}'")
        log(f"Dataset size after filtering: {len(df)} rows (removed {initial_count - len(df)} rows in total)")
    else:
        log("No entries with 'No_Other side drove on wrong side of the road' were found")
    
    return df

def prepare_data(df, test_size=0.2):
    """Split data and create preprocessor"""
    # Define features and target
    X = df.drop('SettlementValue', axis=1)
    y = df['SettlementValue']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    log(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Identify categorical and numeric columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessor
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor=None):
    """Evaluate a single model and return metrics"""
    log(f"\n--- Testing {name} ---")
    start_time = time.time()
    
    try:
        # Handle both sklearn pipeline models and custom models
        if name == 'Unified Segmented Model' and preprocessor is None:
            # For unified model, we need to combine X and y for training
            train_df = pd.concat([X_train, y_train], axis=1)
            model.fit(train_df)
            y_pred = model.predict(X_test)
        elif preprocessor is not None:
            # For sklearn models, use a pipeline with preprocessor
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
        else:
            # For pre-trained models
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate time
        train_time = time.time() - start_time
        
        # Log results
        log(f"Results for {name}:")
        log(f"  MSE: {mse:.2f}")
        log(f"  RMSE: {rmse:.2f}")
        log(f"  MAE: {mae:.2f}")
        log(f"  R² Score: {r2:.4f}")
        log(f"  Training time: {train_time:.2f} seconds")
        
        return {
            'model': model if preprocessor is None else pipeline,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': train_time,
            'predictions': y_pred
        }
        
    except Exception as e:
        log(f"Error evaluating {name}: {str(e)}")
        return None

def test_all_models(X_train, X_test, y_train, y_test, preprocessor):
    """Test all the models consistently"""
    results = {}
    
    # 1. Linear Regression
    results['Linear Regression'] = evaluate_model(
        'Linear Regression',
        LinearRegression(),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 2. Ridge Regression
    results['Ridge Regression'] = evaluate_model(
        'Ridge Regression',
        Ridge(alpha=1.0, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 3. Lasso Regression
    results['Lasso Regression'] = evaluate_model(
        'Lasso Regression',
        Lasso(alpha=0.001, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 4. Random Forest
    results['Random Forest'] = evaluate_model(
        'Random Forest',
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 5. Gradient Boosting
    results['Gradient Boosting'] = evaluate_model(
        'Gradient Boosting',
        GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 6. XGBoost
    results['XGBoost'] = evaluate_model(
        'XGBoost',
        XGBRegressor(n_estimators=100, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 7. Tuned XGBoost
    best_params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'gamma': 0
    }
    
    results['Tuned XGBoost'] = evaluate_model(
        'Tuned XGBoost',
        XGBRegressor(**best_params, random_state=RANDOM_SEED),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 8. Voting Ensemble
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)),
        ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=RANDOM_SEED))
    ]
    
    results['Voting Ensemble'] = evaluate_model(
        'Voting Ensemble',
        VotingRegressor(estimators=base_models),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 9. Stacking Ensemble
    meta_learner = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_SEED)
    
    results['Stacking Ensemble'] = evaluate_model(
        'Stacking Ensemble',
        StackingRegressor(estimators=base_models, final_estimator=meta_learner, cv=5),
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 10. Unified Segmented Model (if available)
    if unified_model_available:
        try:
            unified_model = UnifiedSegmentedModel(
                segmentation_cols=['Exceptional_Circumstances', 'AccidentType'],
                min_segment_size=50,
                use_feature_selection=True,
                verbose=False
            )
            
            results['Unified Segmented Model'] = evaluate_model(
                'Unified Segmented Model',
                unified_model,
                X_train, X_test, y_train, y_test, None  # No preprocessor for this model
            )
        except Exception as e:
            log(f"Error initializing Unified Segmented Model: {str(e)}")
    
    return results

def create_comparison_table(results):
    """Create comparison table from results"""
    # Extract metrics for comparison
    comparison_data = {
        'Model': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R² Score': [],
        'Training Time (s)': []
    }
    
    for model_name, result in results.items():
        if result:
            comparison_data['Model'].append(model_name)
            comparison_data['MSE'].append(result['mse'])
            comparison_data['RMSE'].append(result['rmse'])
            comparison_data['MAE'].append(result['mae'])
            comparison_data['R² Score'].append(result['r2'])
            comparison_data['Training Time (s)'].append(result['training_time'])
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by R² Score (descending)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    
    # Log the table
    log("\nModel Performance Metrics:")
    log(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('test_results/model_comparison.csv', index=False)
    log(f"\nComparison table saved to test_results/model_comparison.csv")
    
    return comparison_df

def main(data_file='Synthetic_Data_For_Students.csv'):
    """Run the main testing process"""
    start_time = time.time()
    
    log("=== Starting Simple Model Testing ===")
    log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data(data_file)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Test all models
    results = test_all_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    log(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    log("\nTesting complete!")
    
    return results, comparison_df

if __name__ == "__main__":
    import sys
    
    # Get file path from command line or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'Synthetic_Data_For_Students.csv'
    
    # Run tests
    main(data_file)