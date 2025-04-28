import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from itertools import combinations

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    
    df = df.dropna(subset=['SettlementValue'])
    print(f"Shape after removing NaN from target: {df.shape}")
    
    df = df[np.isfinite(df['SettlementValue'])]
    print(f"Shape after removing infinity values from target: {df.shape}")
    
    upper_threshold = df['SettlementValue'].quantile(0.999)
    df = df[df['SettlementValue'] <= upper_threshold]
    print(f"Shape after removing extreme outliers from target: {df.shape}")
    
    try:
        df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
        df['Claim Date'] = pd.to_datetime(df['Claim Date'], errors='coerce')
        df['Claim Delay Days'] = (df['Claim Date'] - df['Accident Date']).dt.days
        df['Claim Delay Days'] = df['Claim Delay Days'].fillna(0)
    except:
        print("Error processing date columns, skipping Claim Delay Days calculation")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

def get_categorical_columns(df):
    return [col for col in df.select_dtypes(include=['object']).columns]

def segment_data(df, column_name, min_segment_size=50):
    segments = {}
    for value in df[column_name].unique():
        segment_df = df[df[column_name] == value]
        if len(segment_df) >= min_segment_size:
            segments[value] = segment_df
    
    processed_values = list(segments.keys())
    other_df = df[~df[column_name].isin(processed_values)]
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
    
    return segments

def segment_data_multi(df, column_names, min_segment_size=100):
    segments = {}
    
    value_combinations = df[column_names].drop_duplicates()
    
    for _, row in value_combinations.iterrows():
        filter_condition = True
        segment_name_parts = []
        
        for col in column_names:
            filter_condition = filter_condition & (df[col] == row[col])
            segment_name_parts.append(f"{col}={row[col]}")
        
        segment_name = " & ".join(segment_name_parts)
        segment_df = df[filter_condition]
        
        if len(segment_df) >= min_segment_size:
            segments[segment_name] = segment_df
    
    all_segmented = pd.concat(segments.values())
    remaining = df.loc[~df.index.isin(all_segmented.index)]
    
    if len(remaining) >= min_segment_size:
        segments['Other'] = remaining
    
    return segments

def prepare_features(df):
    numeric_features = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                        if col != 'SettlementValue']
    
    categorical_features = [col for col in df.select_dtypes(include=['object']).columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    y = df['SettlementValue']
    
    X = df.drop('SettlementValue', axis=1)
    
    X = X[numeric_features + categorical_features]
    
    return X, y, preprocessor

def build_model_for_segment(segment_df, segment_name):
    print(f"Building model for segment: {segment_name} (n={len(segment_df)})")
    
    X, y, preprocessor = prepare_features(segment_df)
    
    valid_indices = np.isfinite(y)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42
        ))
    ])
    
    try:
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return pipeline, {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
    except Exception as e:
        print(f"  Error building model for {segment_name}: {str(e)}")
        return None, None

def build_baseline_model(df):
    print(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    return build_model_for_segment(df, "Baseline")

def build_segmented_models(segments, segment_type):
    print(f"\nBuilding {segment_type} models:")
    
    segment_models = {}
    segment_scores = {}
    segment_sizes = {}
    
    for segment_value, segment_df in segments.items():
        model, scores = build_model_for_segment(segment_df, f"{segment_type}: {segment_value}")
        
        if model is not None and scores is not None:
            segment_models[segment_value] = model
            segment_scores[segment_value] = scores
            segment_sizes[segment_value] = len(segment_df)
    
    return segment_models, segment_scores, segment_sizes

def calculate_weighted_average(scores, sizes):
    if not scores or not sizes:
        return None
    
    total_size = sum(sizes.values())
    weighted_rmse = sum(scores[seg]['RMSE'] * sizes[seg] / total_size for seg in scores)
    weighted_mae = sum(scores[seg]['MAE'] * sizes[seg] / total_size for seg in scores)
    weighted_r2 = sum(scores[seg]['R2 Score'] * sizes[seg] / total_size for seg in scores)
    
    return {
        'RMSE': weighted_rmse,
        'MAE': weighted_mae,
        'R2 Score': weighted_r2
    }

def compare_all_segmentations(baseline_scores, all_segmentation_results):
    comparison = {'Baseline': [
        baseline_scores['RMSE'],
        baseline_scores['MAE'],
        baseline_scores['R2 Score']
    ]}
    
    for segment_type, (_, scores, sizes) in all_segmentation_results.items():
        weighted_metrics = calculate_weighted_average(scores, sizes)
        if weighted_metrics:
            comparison[segment_type] = [
                weighted_metrics['RMSE'],
                weighted_metrics['MAE'],
                weighted_metrics['R2 Score']
            ]
    
    comparison_df = pd.DataFrame(comparison, index=['RMSE', 'MAE', 'R2 Score'])
    
    comparison_df = comparison_df.sort_values(by='R2 Score', axis=1, ascending=False)
    
    print("\nModel Comparison (sorted by R2 Score):")
    print(comparison_df.loc['R2 Score'])
    
    return comparison_df

def main():
    start_time = time.time()
    
    df = load_and_preprocess_data('Synthetic_Data_For_Students.csv')
    
    baseline_model, baseline_scores = build_baseline_model(df)
    
    categorical_columns = get_categorical_columns(df)
    print(f"\nCategorical columns found: {categorical_columns}")
    
    all_segmentation_results = {}
    
    for column in categorical_columns:
        print(f"\n{'='*50}")
        print(f"Testing segmentation by: {column}")
        print(f"{'='*50}")
        
        unique_values = df[column].nunique()
        print(f"Number of unique values: {unique_values}")
        
        if unique_values > 20:
            print(f"Skipping {column} due to high cardinality ({unique_values} unique values)")
            continue
        
        segments = segment_data(df, column)
        
        if len(segments) <= 1:
            print(f"Skipping {column} as it didn't create meaningful segments")
            continue
        
        models, scores, sizes = build_segmented_models(segments, column)
        
        all_segmentation_results[column] = (models, scores, sizes)
    
    print("\nFinding top performing individual segmentations...")
    top_performers = []
    
    for segment_type, (_, scores, sizes) in all_segmentation_results.items():
        weighted_metrics = calculate_weighted_average(scores, sizes)
        if weighted_metrics:
            top_performers.append((segment_type, weighted_metrics['R2 Score']))
    
    top_performers.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 individual segmentation variables:")
    for i, (segment_type, r2) in enumerate(top_performers[:5], 1):
        print(f"{i}. {segment_type}: R² = {r2:.4f}")
    
    if len(top_performers) >= 2:
        top_columns = [col for col, _ in top_performers[:5]]
        
        print("\nTesting combinations of top performing variables...")
        
        for col1, col2 in combinations(top_columns, 2):
            combination_name = f"{col1} + {col2}"
            print(f"\n{'='*50}")
            print(f"Testing combination: {combination_name}")
            print(f"{'='*50}")
            
            segments = segment_data_multi(df, [col1, col2], min_segment_size=100)
            
            if len(segments) <= 1:
                print(f"Skipping {combination_name} as it didn't create meaningful segments")
                continue
            
            models, scores, sizes = build_segmented_models(segments, combination_name)
            
            all_segmentation_results[combination_name] = (models, scores, sizes)
    
    comparison = compare_all_segmentations(baseline_scores, all_segmentation_results)
    
    best_r2 = comparison.loc['R2 Score'].max()
    best_approach = comparison.loc['R2 Score'].idxmax()
    
    print(f"\nBest performing segmentation approach: {best_approach}")
    print(f"R² Score: {best_r2:.4f}")
    
    baseline_r2 = comparison.loc['R2 Score']['Baseline']
    improvement = (best_r2 - baseline_r2) / baseline_r2 * 100
    print(f"Improvement over baseline: {improvement:.2f}%")
    
    summary = pd.DataFrame({
        'Segmentation Approach': comparison.columns,
        'R² Score': comparison.loc['R2 Score'].values,
        'RMSE': comparison.loc['RMSE'].values,
        'MAE': comparison.loc['MAE'].values,
        'Improvement (%)': [(r2 - baseline_r2) / baseline_r2 * 100 for r2 in comparison.loc['R2 Score'].values]
    })
    
    summary = summary.sort_values('R² Score', ascending=False)
    
    summary.to_csv('segmentation_results_summary.csv', index=False)
    print("\nFull results saved to 'segmentation_results_summary.csv'")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()