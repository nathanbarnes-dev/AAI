import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime

os.makedirs('output', exist_ok=True)

log_filename = f'output/model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file = open(log_filename, 'w')

def log(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    log(f"Original dataset shape: {df.shape}")
    
    df = df.dropna(subset=['SettlementValue'])
    log(f"Shape after removing NaN from target: {df.shape}")
    
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

def create_segments(df, column='Exceptional_Circumstances', min_segment_size=50):
    segments = {}
    
    for value in df[column].unique():
        segment_df = df[df[column] == value]
        if len(segment_df) >= min_segment_size:
            segments[value] = segment_df
    
    processed_values = list(segments.keys())
    other_df = df[~df[column].isin(processed_values)]
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
    
    return segments

def get_xy_data(df):
    y = df['SettlementValue']
    
    X = df.drop('SettlementValue', axis=1)
    
    return X, y

def build_model_for_segment(segment_df, segment_name):
    log(f"Building model for segment: {segment_name} (n={len(segment_df)})")
    
    numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    df_numeric = segment_df[numeric_cols + ['SettlementValue']].copy()
    
    X, y = get_xy_data(df_numeric)
    
    valid_indices = np.isfinite(y)
    X = X[valid_indices]
    y = y[valid_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
    
    try:
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        log(f"\nTop 10 feature importances for {segment_name}:")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            log(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return model, {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
    except Exception as e:
        log(f"  Error building model for {segment_name}: {str(e)}")
        return None, None

def build_baseline_model(df):
    log(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    return build_model_for_segment(df, "Baseline")

def build_segmented_models(segments):
    log(f"\nBuilding segmented models:")
    
    segment_models = {}
    segment_scores = {}
    segment_sizes = {}
    
    for segment_value, segment_df in segments.items():
        model, scores = build_model_for_segment(segment_df, f"Segment_{segment_value}")
        
        if model is not None and scores is not None:
            segment_models[segment_value] = model
            segment_scores[segment_value] = scores
            segment_sizes[segment_value] = len(segment_df)
    
    return segment_models, segment_scores, segment_sizes

def build_stacked_model(df, segments, segment_models):
    log("\nBuilding stacked ensemble model")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    df_numeric = df[numeric_cols + ['SettlementValue']].copy()
    
    segmentation_col = 'Exceptional_Circumstances'
    
    df_numeric['segment'] = df[segmentation_col].apply(
        lambda x: x if x in segment_models else 'Other'
    )
    
    df_numeric['segment_prediction'] = 0
    
    for segment, model in segment_models.items():
        mask = df_numeric['segment'] == segment
        if not any(mask):
            continue
        
        segment_data = df_numeric.loc[mask, numeric_cols]
        
        try:
            preds = model.predict(segment_data)
            df_numeric.loc[mask, 'segment_prediction'] = preds
        except Exception as e:
            log(f"  Error getting predictions for segment {segment}: {str(e)}")
    
    df_numeric = df_numeric.drop('segment', axis=1)
    
    X = df_numeric.drop('SettlementValue', axis=1)
    y = df_numeric['SettlementValue']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    stacked_model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    
    stacked_model.fit(X_train, y_train)
    
    y_pred = stacked_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log(f"Stacked model performance:")
    log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    log("\nFeature importances for stacked model:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': stacked_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        log(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return stacked_model, {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

def compare_models(baseline_scores, segmented_scores, stacked_scores, segment_sizes):
    total_size = sum(segment_sizes.values())
    weighted_rmse = sum(segmented_scores[seg]['RMSE'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    weighted_mae = sum(segmented_scores[seg]['MAE'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    weighted_r2 = sum(segmented_scores[seg]['R2 Score'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    
    comparison = pd.DataFrame({
        'Baseline': [
            baseline_scores['RMSE'],
            baseline_scores['MAE'],
            baseline_scores['R2 Score']
        ],
        'Segmented Models (Weighted)': [
            weighted_rmse,
            weighted_mae,
            weighted_r2
        ],
        'Stacked Model': [
            stacked_scores['RMSE'],
            stacked_scores['MAE'],
            stacked_scores['R2 Score']
        ]
    }, index=['RMSE', 'MAE', 'R2 Score'])
    
    log("\nModel Comparison:")
    log(comparison.to_string())
    
    baseline_r2 = baseline_scores['R2 Score']
    improvements = pd.DataFrame({
        'Model': ['Segmented Models', 'Stacked Model'],
        'R2 Score': [weighted_r2, stacked_scores['R2 Score']],
        'Improvement (%)': [
            (weighted_r2 - baseline_r2) / baseline_r2 * 100,
            (stacked_scores['R2 Score'] - baseline_r2) / baseline_r2 * 100
        ]
    })
    
    log("\nImprovements over Baseline:")
    log(improvements.to_string())
    
    log("\nPerformance by segment:")
    segment_performance = pd.DataFrame({
        'R2 Score': pd.Series({k: v['R2 Score'] for k, v in segmented_scores.items()}),
        'Segment Size': pd.Series(segment_sizes)
    }).sort_values('R2 Score', ascending=False)
    log(segment_performance.to_string())
    
    return comparison, improvements

def main():
    try:
        log("======= STARTING MODEL BUILDING PROCESS =======")
        log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        df = load_and_preprocess_data('Synthetic_Data_For_Students.csv')
        
        log("\nDataset Information:")
        log(f"  Number of records: {len(df)}")
        log(f"  Number of features: {len(df.columns) - 1}")
        log(f"  Numeric features: {len(df.select_dtypes(include=['float64', 'int64']).columns)}")
        log(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
        
        log("\nTarget Variable (SettlementValue) Statistics:")
        log(f"  Mean: {df['SettlementValue'].mean():.2f}")
        log(f"  Median: {df['SettlementValue'].median():.2f}")
        log(f"  Min: {df['SettlementValue'].min():.2f}")
        log(f"  Max: {df['SettlementValue'].max():.2f}")
        log(f"  Standard Deviation: {df['SettlementValue'].std():.2f}")
        
        baseline_model, baseline_scores = build_baseline_model(df)
        
        segments = create_segments(df, 'Exceptional_Circumstances')
        
        log("\nSegmentation Information:")
        log(f"  Segmentation variable: Exceptional_Circumstances")
        log(f"  Number of segments: {len(segments)}")
        for segment, segment_df in segments.items():
            log(f"  Segment '{segment}': {len(segment_df)} records ({len(segment_df)/len(df)*100:.1f}%)")
        
        segment_models, segment_scores, segment_sizes = build_segmented_models(segments)
        
        stacked_model, stacked_scores = build_stacked_model(df, segments, segment_models)
        
        comparison, improvements = compare_models(
            baseline_scores, 
            segment_scores, 
            stacked_scores, 
            segment_sizes
        )
        
        comparison.to_csv('output/model_comparison_summary.csv')
        improvements.to_csv('output/model_improvements_summary.csv')
        
        log("\n" + "="*50)
        log("MODEL BUILDING COMPLETE")
        log("="*50)
        log(f"Results saved in the 'output' directory")
        log(f"Detailed log saved to: {log_filename}")
        
        best_model = improvements.loc[improvements['R2 Score'].idxmax(), 'Model']
        best_r2 = improvements.loc[improvements['R2 Score'].idxmax(), 'R2 Score']
        best_improvement = improvements.loc[improvements['R2 Score'].idxmax(), 'Improvement (%)']
        
        log("\nBest model performance:")
        log(f"  Model: {best_model}")
        log(f"  R² Score: {best_r2:.4f}")
        log(f"  Improvement: {best_improvement:.2f}%")
        
        log("\nConclusion and Recommendations:")
        if best_improvement > 5:
            log("  The segmented modeling approach shows substantial improvement over the baseline model.")
            log("  Recommendation: Implement the stacked model approach in production.")
        elif best_improvement > 2:
            log("  The segmented modeling approach shows moderate improvement over the baseline model.")
            log("  Recommendation: Consider implementing the segmented approach, particularly for critical segments.")
        else:
            log("  The segmented modeling approach shows minimal improvement over the baseline model.")
            log("  Recommendation: Further investigate additional segmentation variables or feature engineering approaches.")
        
    except Exception as e:
        log(f"\nERROR in main execution: {str(e)}")
    finally:
        log("\n======= MODEL BUILDING PROCESS FINISHED =======")
        log_file.close()

if __name__ == "__main__":
    main()