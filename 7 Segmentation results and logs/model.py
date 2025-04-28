import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

os.makedirs('output', exist_ok=True)

log_filename = f'output/enhanced_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file = open(log_filename, 'w')

def log(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def load_data(file_path):
    df = pd.read_csv(file_path)
    log(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df):
    log("\n--- Handling Missing Values and Outliers ---")
    missing_in_target = df['SettlementValue'].isna().sum()
    log(f"Missing values in SettlementValue: {missing_in_target}")
    
    if missing_in_target > 0:
        log(f"Removing {missing_in_target} rows with missing SettlementValue")
        df = df.dropna(subset=['SettlementValue'])
    
    df = df[np.isfinite(df['SettlementValue'])]
    log(f"Shape after removing infinity values from target: {df.shape}")
    
    upper_threshold = df['SettlementValue'].quantile(0.999)
    df = df[df['SettlementValue'] <= upper_threshold]
    log(f"Shape after removing extreme outliers from target: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

def engineer_features(df):
    log("\n--- Adding Engineered Features ---")
    
    df_new = df.copy()
    feature_count = 0
    
    if 'Accident Date' in df.columns and 'Claim Date' in df.columns:
        try:
            df_new['Days_To_Claim'] = (pd.to_datetime(df_new['Claim Date']) - 
                                      pd.to_datetime(df_new['Accident Date'])).dt.days
            feature_count += 1
        except:
            log("Could not calculate days between dates")
    
    special_columns = [col for col in df.columns if col.startswith('Special')]
    if len(special_columns) > 0:
        df_new['Total_Special_Damages'] = df_new[special_columns].sum(axis=1)
        df_new['Log_Special_Damages'] = np.log1p(df_new['Total_Special_Damages'])
        feature_count += 2
    
    general_columns = [col for col in df.columns if col.startswith('General')]
    if len(general_columns) > 0:
        df_new['Total_General_Damages'] = df_new[general_columns].sum(axis=1)
        df_new['Log_General_Damages'] = np.log1p(df_new['Total_General_Damages'])
        feature_count += 2
    
    if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
        df_new['Total_Damages'] = df_new['Total_Special_Damages'] + df_new['Total_General_Damages']
        df_new['Log_Total_Damages'] = np.log1p(df_new['Total_Damages'])
        feature_count += 2
    
    if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
        df_new['Special_General_Ratio'] = df_new['Total_Special_Damages'] / df_new['Total_General_Damages'].replace(0, 1)
        feature_count += 1
    
    if 'Driver Age' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Age_Damage_Ratio'] = df_new['Driver Age'] / df_new['Total_Special_Damages'].replace(0, 1)
        feature_count += 1
    
    injury_flag_created = False
    for col in df_new.select_dtypes(include=['object']).columns:
        if 'Injury' in col or 'injury' in col:
            if not injury_flag_created:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                injury_flag_created = True
                feature_count += 1
    
    if 'Whiplash' in df_new.columns:
        df_new['Whiplash_Numeric'] = (df_new['Whiplash'] == 'Yes').astype(int)
        feature_count += 1
    
    if 'Severe_Injury_Flag' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Severe_Injury_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_Special_Damages']
        feature_count += 1
    
    if 'Vehicle Age' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Vehicle_Age_Damage_Interaction'] = df_new['Vehicle Age'] * df_new['Total_Special_Damages'] / 1000
        feature_count += 1
    
    if 'Minor_Psychological_Injury' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Psych_Injury_Numeric'] = (df_new['Minor_Psychological_Injury'] == 'Yes').astype(int)
        df_new['Psych_Damage_Interaction'] = df_new['Psych_Injury_Numeric'] * df_new['Total_Special_Damages']
        feature_count += 2
    
    for col in ['Total_Special_Damages', 'Total_General_Damages']:
        if col in df_new.columns:
            df_new[f'{col}_Squared'] = df_new[col] ** 2
            feature_count += 1
    
    log(f"Added {feature_count} engineered features")
    return df_new

def create_multi_segments(df, primary_col='Exceptional_Circumstances', secondary_col='AccidentType', min_segment_size=50):
    log(f"\n--- Creating Multi-level Segmentation ({primary_col} + {secondary_col}) ---")
    
    segments = {}
    
    primary_values = df[primary_col].unique()
    
    for primary_value in primary_values:
        primary_df = df[df[primary_col] == primary_value]
        
        if len(primary_df) < min_segment_size:
            log(f"Primary segment '{primary_value}' has only {len(primary_df)} rows, below minimum {min_segment_size}")
            continue
        
        if len(primary_df) >= 3 * min_segment_size:
            secondary_values = primary_df[secondary_col].unique()
            
            for secondary_value in secondary_values:
                segment_df = primary_df[primary_df[secondary_col] == secondary_value]
                
                if len(segment_df) >= min_segment_size:
                    segment_name = f"{primary_value}_{secondary_value}"
                    segments[segment_name] = segment_df
                    log(f"Created segment '{segment_name}' with {len(segment_df)} rows")
        else:
            segments[primary_value] = primary_df
            log(f"Created segment '{primary_value}' with {len(primary_df)} rows")
    
    all_segmented_indices = []
    for segment_df in segments.values():
        all_segmented_indices.extend(segment_df.index.tolist())
    
    other_df = df.loc[~df.index.isin(all_segmented_indices)]
    
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
        log(f"Created 'Other' segment with {len(other_df)} rows")
    
    log(f"Created a total of {len(segments)} segments")
    return segments

def select_features_for_segment(X, y, segment_name):
    log(f"Selecting features for segment: {segment_name}")
    
    feature_selector = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    feature_selector.fit(X, y)
    
    selection_model = SelectFromModel(
        feature_selector, 
        threshold='mean',
        prefit=True
    )
    
    X_selected = selection_model.transform(X)
    
    selected_indices = selection_model.get_support()
    selected_features = X.columns[selected_indices].tolist()
    
    log(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    
    return X_selected, selected_features, selection_model

def tune_hyperparameters_for_segment(X, y, segment_name):
    log(f"Tuning hyperparameters for segment: {segment_name}")
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = XGBRegressor(random_state=42)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=kfold,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    log(f"Best hyperparameters: {best_params}")
    
    best_model = XGBRegressor(
        random_state=42,
        **best_params
    )
    
    return best_model, best_params

def build_model_for_segment(segment_df, segment_name, tune_hyperparams=True, select_features=True):
    log(f"\nBuilding model for segment: {segment_name} (n={len(segment_df)})")
    
    numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    df_numeric = segment_df[numeric_cols + ['SettlementValue']].copy()
    
    X = df_numeric.drop('SettlementValue', axis=1)
    y = df_numeric['SettlementValue']
    
    valid_indices = np.isfinite(y)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    selected_features = None
    feature_selector = None
    
    if select_features and X.shape[1] > 10:
        X_train_selected, selected_features, feature_selector = select_features_for_segment(X, y, segment_name)
    else:
        X_train_selected = X
        selected_features = X.columns.tolist()
    
    if tune_hyperparams:
        model, best_params = tune_hyperparameters_for_segment(X_train_selected, y, segment_name)
    else:
        model = XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42
        )
    
    model.fit(X_train_selected, y)
    
    y_pred = model.predict(X_train_selected)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    log(f"Segment model performance:")
    log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    model_info = {
        'model': model,
        'feature_selector': feature_selector,
        'selected_features': selected_features,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'size': len(segment_df)
    }
    
    return model_info

def build_all_segment_models(segments, tune_hyperparams=True, select_features=True):
    segment_models = {}
    
    for segment_name, segment_df in segments.items():
        model_info = build_model_for_segment(
            segment_df, 
            segment_name,
            tune_hyperparams=tune_hyperparams,
            select_features=select_features
        )
        segment_models[segment_name] = model_info
    
    return segment_models

def build_baseline_model(df):
    log(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    return build_model_for_segment(df, "Baseline", tune_hyperparams=False, select_features=False)

def create_meta_features(df, segments, segment_models):
    log("\nCreating meta-features for stacking ensemble")
    
    df_meta = df.copy()
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    primary_segmentation_col = 'Exceptional_Circumstances'
    
    df_meta['segment_id'] = 'Other'
    
    for segment_name in segment_models.keys():
        if '_' in segment_name:
            parts = segment_name.split('_')
            mask = (df[primary_segmentation_col] == parts[0])
            for i, part in enumerate(parts[1:], 1):
                if i == 1 and 'AccidentType' in df.columns:
                    mask = mask & (df['AccidentType'] == part)
            df_meta.loc[mask, 'segment_id'] = segment_name
        else:
            mask = df[primary_segmentation_col] == segment_name
            df_meta.loc[mask, 'segment_id'] = segment_name
    
    for segment_name in segment_models.keys():
        df_meta[f'pred_{segment_name}'] = np.nan
    
    for segment_name, model_info in segment_models.items():
        mask = df_meta['segment_id'] == segment_name
        
        if not any(mask):
            continue
        
        segment_X = df.loc[mask, numeric_cols]
        
        if model_info['feature_selector'] is not None:
            try:
                segment_X = model_info['feature_selector'].transform(segment_X)
            except Exception as e:
                log(f"Error applying feature selection for {segment_name}: {str(e)}")
                if model_info['selected_features'] is not None:
                    segment_X = segment_X[model_info['selected_features']]
        elif model_info['selected_features'] is not None:
            try:
                segment_X = segment_X[model_info['selected_features']]
            except Exception as e:
                log(f"Error selecting features for {segment_name}: {str(e)}")
        
        try:
            preds = model_info['model'].predict(segment_X)
            df_meta.loc[mask, f'pred_{segment_name}'] = preds
        except Exception as e:
            log(f"Error getting predictions for segment {segment_name}: {str(e)}")
    
    for segment_name, model_info in segment_models.items():
        try:
            all_X = df[numeric_cols]
            
            if model_info['feature_selector'] is not None:
                try:
                    all_X = model_info['feature_selector'].transform(all_X)
                except:
                    if model_info['selected_features'] is not None:
                        all_X = all_X[model_info['selected_features']]
            elif model_info['selected_features'] is not None:
                try:
                    all_X = all_X[model_info['selected_features']]
                except:
                    pass
            
            all_preds = model_info['model'].predict(all_X)
            df_meta[f'all_pred_{segment_name}'] = all_preds
        except Exception as e:
            log(f"Error getting all predictions from {segment_name} model: {str(e)}")
    
    pred_cols = [col for col in df_meta.columns if col.startswith('pred_') and not col.startswith('pred_all_')]
    all_pred_cols = [col for col in df_meta.columns if col.startswith('all_pred_')]
    
    if pred_cols:
        df_meta['pred_mean'] = df_meta[pred_cols].mean(axis=1)
    
    if all_pred_cols:
        df_meta['all_pred_mean'] = df_meta[all_pred_cols].mean(axis=1)
    
    df_meta['segment_prediction'] = np.nan
    for segment_name in segment_models.keys():
        mask = df_meta['segment_id'] == segment_name
        if any(mask) and f'pred_{segment_name}' in df_meta.columns:
            df_meta.loc[mask, 'segment_prediction'] = df_meta.loc[mask, f'pred_{segment_name}']
    
    return df_meta

def build_stacked_ensemble(df_meta):
    log("\nBuilding stacked ensemble model")
    
    meta_cols = [col for col in df_meta.columns if col.startswith('pred_') or 
                 col.startswith('all_pred_') or col == 'segment_prediction']
    
    numeric_cols = df_meta.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue' and 
                    not col.startswith('pred_') and not col.startswith('all_pred_')]
    
    if not meta_cols:
        log("Warning: No prediction columns found in meta-features. Using only numeric features.")
        X_meta = df_meta[numeric_cols].copy()
    else:
        valid_meta_cols = [col for col in meta_cols if not df_meta[col].isna().all()]
        if not valid_meta_cols:
            log("Warning: All prediction columns contain only NaN values. Using only numeric features.")
            X_meta = df_meta[numeric_cols].copy()
        else:
            X_meta = df_meta[valid_meta_cols + numeric_cols].copy()
    
    X_meta = X_meta.fillna(0)
    
    y_meta = df_meta['SettlementValue']
    
    if X_meta.isna().any().any():
        log("Warning: X_meta still contains NaN values after filling. Filling remaining NaNs with 0.")
        X_meta = X_meta.fillna(0)
    
    if y_meta.isna().any():
        log("Warning: y_meta contains NaN values. Dropping corresponding rows.")
        valid_indices = ~y_meta.isna()
        X_meta = X_meta.loc[valid_indices]
        y_meta = y_meta.loc[valid_indices]
    
    X_meta_np = X_meta.to_numpy()
    y_meta_np = y_meta.to_numpy()
    
    log(f"X_meta shape: {X_meta.shape}, y_meta shape: {y_meta.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta_np, y_meta_np, test_size=0.2, random_state=42
    )
    
    meta_model = XGBRegressor(
        learning_rate=0.05,
        max_depth=4,
        n_estimators=150,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    
    try:
        meta_model.fit(X_train, y_train)
        
        y_pred = meta_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"Stacked ensemble performance:")
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        feature_names = X_meta.columns.tolist()
        
        return meta_model, {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    except Exception as e:
        log(f"Error in stacked ensemble building: {str(e)}")
        import traceback
        log(traceback.format_exc())
        
        log("Creating a simple fallback model instead")
        
        fallback_X = df_meta.select_dtypes(include=['float64', 'int64']).drop('SettlementValue', axis=1)
        fallback_y = df_meta['SettlementValue']
        
        X_train, X_test, y_train, y_test = train_test_split(
            fallback_X, fallback_y, test_size=0.2, random_state=42
        )
        
        fallback_model = XGBRegressor(random_state=42)
        
        fallback_model.fit(X_train, y_train)
        
        y_pred = fallback_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"Fallback model performance:")
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return fallback_model, {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

def compare_and_visualize(baseline_metrics, segment_models, stacked_metrics):
    segment_metrics = {name: info['metrics'] for name, info in segment_models.items()}
    segment_sizes = {name: info['size'] for name, info in segment_models.items()}
    
    total_size = sum(segment_sizes.values())
    weighted_rmse = sum(segment_metrics[seg]['rmse'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    weighted_mae = sum(segment_metrics[seg]['mae'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    weighted_r2 = sum(segment_metrics[seg]['r2'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    
    comparison = pd.DataFrame({
        'Baseline': [
            baseline_metrics['rmse'],
            baseline_metrics['mae'],
            baseline_metrics['r2']
        ],
        'Segmented (Weighted)': [
            weighted_rmse,
            weighted_mae,
            weighted_r2
        ],
        'Stacked Ensemble': [
            stacked_metrics['rmse'],
            stacked_metrics['mae'],
            stacked_metrics['r2']
        ]
    }, index=['RMSE', 'MAE', 'R2'])
    
    log("\nModel Comparison:")
    log(comparison.to_string())
    
    baseline_r2 = baseline_metrics['r2']
    improvements = pd.DataFrame({
        'Model': ['Segmented (Weighted)', 'Stacked Ensemble'],
        'R2 Score': [weighted_r2, stacked_metrics['r2']],
        'Improvement (%)': [
            (weighted_r2 - baseline_r2) / baseline_r2 * 100,
            (stacked_metrics['r2'] - baseline_r2) / baseline_r2 * 100
        ]
    })
    
    log("\nImprovements over Baseline:")
    log(improvements.to_string())
    
    log("\nPerformance by Segment:")
    segment_performance = pd.DataFrame({
        'Segment': list(segment_metrics.keys()),
        'R2': [segment_metrics[seg]['r2'] for seg in segment_metrics],
        'RMSE': [segment_metrics[seg]['rmse'] for seg in segment_metrics],
        'Size': [segment_sizes[seg] for seg in segment_metrics]
    }).sort_values('R2', ascending=False)
    log(segment_performance.to_string())
    
    return comparison, improvements, segment_performance

def save_results(baseline_model, segment_models, stacked_model, comparison, improvements, segment_performance):
    os.makedirs('output/models', exist_ok=True)
    
    joblib.dump(baseline_model['model'], 'output/models/baseline_model.pkl')
    
    for segment_name, model_info in segment_models.items():
        safe_name = segment_name.replace(" ", "_").replace("/", "_")
        joblib.dump(model_info['model'], f'output/models/{safe_name}_model.pkl')
    
    joblib.dump(stacked_model, 'output/models/stacked_model.pkl')
    
    comparison.to_csv('output/enhanced_model_comparison.csv')
    improvements.to_csv('output/enhanced_model_improvements.csv')
    segment_performance.to_csv('output/enhanced_segment_performance.csv')
    
    log("\nAll models and results saved to output directory")

def main():
    start_time = time.time()
    
    try:
        log("======= STARTING ENHANCED MODEL BUILDING PROCESS =======")
        log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        df = load_data('Synthetic_Data_For_Students.csv')
        
        df = preprocess_data(df)
        
        df = engineer_features(df)
        
        baseline_model = build_baseline_model(df)
        
        segments = create_multi_segments(df, 'Exceptional_Circumstances', 'AccidentType')
        
        segment_models = build_all_segment_models(segments, tune_hyperparams=True, select_features=True)
        
        df_meta = create_meta_features(df, segments, segment_models)
        
        stacked_model, stacked_metrics = build_stacked_ensemble(df_meta)
        
        comparison, improvements, segment_performance = compare_and_visualize(
            baseline_model['metrics'], 
            segment_models, 
            stacked_metrics
        )
        
        save_results(
            baseline_model,
            segment_models, 
            stacked_model,
            comparison, 
            improvements, 
            segment_performance
        )
        
        log("\n" + "="*50)
        log("MODEL BUILDING COMPLETE")
        log("="*50)
        
        if stacked_metrics['r2'] > baseline_model['metrics']['r2']:
            improvement = (stacked_metrics['r2'] - baseline_model['metrics']['r2']) / baseline_model['metrics']['r2'] * 100
            log(f"\nBest model: Stacked Ensemble")
            log(f"R² Score: {stacked_metrics['r2']:.4f}")
            log(f"Improvement over baseline: {improvement:.2f}%")
        else:
            log(f"\nStacked model did not improve over baseline")
        
        end_time = time.time()
        execution_time = end_time - start_time
        log(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        
    except Exception as e:
        log(f"\nERROR in main execution: {str(e)}")
        import traceback
        log(traceback.format_exc())
    finally:
        log("\n======= MODEL BUILDING PROCESS FINISHED =======")
        log_file.close()

if __name__ == "__main__":
    main()