import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class UnifiedSegmentedModel:
    def __init__(self, segmentation_cols=['Exceptional_Circumstances', 'AccidentType'], 
                 min_segment_size=50, use_feature_selection=True, verbose=True):
        self.segmentation_cols = segmentation_cols
        self.min_segment_size = min_segment_size
        self.use_feature_selection = use_feature_selection
        self.verbose = verbose
        
        self.segments = {}
        self.segment_models = {}
        self.meta_model = None
        self.feature_columns = None
        self.is_trained = False
        
    def _log(self, message):
        if self.verbose:
            print(message)
    
    def _preprocess_data(self, df):
        df = df.copy()
        
        if 'SettlementValue' in df.columns and df['SettlementValue'].isna().any():
            self._log("Handling missing values in target variable")
            if self.is_trained:
                self._log(f"Note: {df['SettlementValue'].isna().sum()} rows have missing target values")
            else:
                df = df.dropna(subset=['SettlementValue'])
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if 'SettlementValue' in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _engineer_features(self, df):
        self._log("Engineering features")
        df_new = df.copy()
        
        if 'Accident Date' in df.columns and 'Claim Date' in df.columns:
            try:
                df_new['Days_To_Claim'] = (pd.to_datetime(df_new['Claim Date']) - 
                                          pd.to_datetime(df_new['Accident Date'])).dt.days
            except:
                pass
        
        special_columns = [col for col in df.columns if col.startswith('Special')]
        if len(special_columns) > 0:
            df_new['Total_Special_Damages'] = df_new[special_columns].sum(axis=1)
            df_new['Log_Special_Damages'] = np.log1p(df_new['Total_Special_Damages'])
        
        general_columns = [col for col in df.columns if col.startswith('General')]
        if len(general_columns) > 0:
            df_new['Total_General_Damages'] = df_new[general_columns].sum(axis=1)
            df_new['Log_General_Damages'] = np.log1p(df_new['Total_General_Damages'])
        
        if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
            df_new['Total_Damages'] = df_new['Total_Special_Damages'] + df_new['Total_General_Damages']
            df_new['Log_Total_Damages'] = np.log1p(df_new['Total_Damages'])
            df_new['Special_General_Ratio'] = df_new['Total_Special_Damages'] / df_new['Total_General_Damages'].replace(0, 1)
        
        for col in df_new.select_dtypes(include=['object']).columns:
            if 'Injury' in col or 'injury' in col:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                break
        
        if 'Whiplash' in df_new.columns:
            df_new['Whiplash_Numeric'] = (df_new['Whiplash'] == 'Yes').astype(int)
        
        if 'Minor_Psychological_Injury' in df_new.columns:
            df_new['Psych_Injury_Numeric'] = (df_new['Minor_Psychological_Injury'] == 'Yes').astype(int)
        
        if 'Severe_Injury_Flag' in df_new.columns and 'Total_Damages' in df_new.columns:
            df_new['Severe_Injury_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_Damages']
        
        if 'Psych_Injury_Numeric' in df_new.columns and 'Total_Damages' in df_new.columns:
            df_new['Psych_Damage_Interaction'] = df_new['Psych_Injury_Numeric'] * df_new['Total_Damages']
        
        return df_new
    
    def _create_segments(self, df):
        self._log(f"Creating segments based on {self.segmentation_cols}")
        segments = {}
        
        if len(self.segmentation_cols) == 1:
            col = self.segmentation_cols[0]
            for value in df[col].unique():
                segment_df = df[df[col] == value]
                if len(segment_df) >= self.min_segment_size:
                    segments[f"{value}"] = segment_df
        
        else:
            df = df.copy()
            df['_segment_id'] = ''
            for col in self.segmentation_cols:
                df['_segment_id'] += df[col].astype(str) + '_'
            
            for seg_id, segment_df in df.groupby('_segment_id'):
                if len(segment_df) >= self.min_segment_size:
                    segments[seg_id] = segment_df.drop('_segment_id', axis=1)
        
        all_segmented = pd.concat([segment_df for segment_df in segments.values()])
        other_records = df.loc[~df.index.isin(all_segmented.index)]
        
        if len(other_records) >= self.min_segment_size:
            segments['Other'] = other_records
        else:
            largest_segment = max(segments.items(), key=lambda x: len(x[1]))
            segments[largest_segment[0]] = pd.concat([largest_segment[1], other_records])
        
        self._log(f"Created {len(segments)} segments")
        return segments
    
    def _get_segment_for_record(self, record):
        if len(self.segmentation_cols) == 1:
            col = self.segmentation_cols[0]
            value = record[col]
            segment_id = f"{value}"
            if segment_id in self.segments:
                return segment_id
        
        else:
            segment_id = ''
            for col in self.segmentation_cols:
                segment_id += str(record[col]) + '_'
            
            if segment_id in self.segments:
                return segment_id
        
        return 'Other'
    
    def _select_features(self, X, y):
        if X.shape[1] <= 5:
            return X, X.columns.tolist()
        
        feature_selector = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        feature_selector.fit(X, y)
        
        selection_model = SelectFromModel(
            feature_selector, 
            threshold='mean',
            prefit=True
        )
        
        selected_indices = selection_model.get_support()
        selected_features = X.columns[selected_indices].tolist()
        
        if len(selected_features) < 3:
            importances = feature_selector.feature_importances_
            top_indices = np.argsort(importances)[-3:]
            selected_features = X.columns[top_indices].tolist()
        
        return X[selected_features], selected_features
    
    def _train_segment_model(self, segment_df, segment_name):
        self._log(f"Training model for segment: {segment_name} (n={len(segment_df)})")
        
        numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        X = segment_df[numeric_cols].copy()
        y = segment_df['SettlementValue']
        
        if self.use_feature_selection and X.shape[1] > 5:
            X_selected, selected_features = self._select_features(X, y)
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        model = XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_selected, y)
        
        y_pred = model.predict(X_selected)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        self._log(f"  RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return {
            'model': model,
            'features': selected_features,
            'metrics': {
                'rmse': rmse,
                'r2': r2
            },
            'size': len(segment_df)
        }
    
    def _create_meta_features(self, df):
        self._log("Creating meta-features for stacking ensemble")
        
        df_meta = df.copy()
        
        try:
            df_meta['segment_id'] = df_meta.apply(self._get_segment_for_record, axis=1)
        except Exception as e:
            self._log(f"Error assigning segments: {str(e)}")
            df_meta['segment_id'] = 'Other'
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        for segment_name, model_info in self.segment_models.items():
            df_meta[f'pred_{segment_name}'] = np.nan
            
            segment_model = model_info['model']
            segment_features = model_info['features']
            
            missing_features = [f for f in segment_features if f not in df_meta.columns]
            if missing_features:
                self._log(f"Warning: Missing features for segment {segment_name}: {missing_features}")
                continue
            
            segment_mask = df_meta['segment_id'] == segment_name
            
            if any(segment_mask):
                try:
                    X_segment = df_meta.loc[segment_mask, segment_features]
                    preds = segment_model.predict(X_segment)
                    df_meta.loc[segment_mask, f'pred_{segment_name}'] = preds
                except Exception as e:
                    self._log(f"Error predicting for segment {segment_name}: {str(e)}")
        
        df_meta['primary_segment_pred'] = df['SettlementValue'].mean() if 'SettlementValue' in df.columns else 0
        
        if 'Total_Special_Damages' in df_meta.columns and 'Total_General_Damages' in df_meta.columns:
            df_meta['total_damages_meta'] = df_meta['Total_Special_Damages'] + df_meta['Total_General_Damages']
        
        meta_features = [col for col in df_meta.columns if 'pred' in col or 'meta' in col or 
                        col in ['Total_Special_Damages', 'Total_General_Damages']]
        
        if not meta_features:
            self._log("Warning: No meta-features created, using numeric columns")
            meta_features = numeric_cols
        
        final_features = [col for col in meta_features if col in df_meta.columns]
        return df_meta[final_features].fillna(0)
    
    def _build_meta_model(self, X_meta, y):
        self._log("Building meta-model")
        
        base_models = [
            ('xgb', XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=150, random_state=42)),
            ('gbm', GradientBoostingRegressor(learning_rate=0.05, max_depth=4, n_estimators=150, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42))
        ]
        
        final_estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
        
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
        
        stacking_regressor.fit(X_meta, y)
        
        y_pred = stacking_regressor.predict(X_meta)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        self._log(f"Meta-model performance - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return stacking_regressor
    
    def fit(self, df):
        start_time = time.time()
        self._log("Training unified segmented model")
        
        df = self._preprocess_data(df)
        df = self._engineer_features(df)
        self.feature_columns = df.columns.tolist()
        self.segments = self._create_segments(df)
        
        self.segment_models = {}
        for segment_name, segment_df in self.segments.items():
            self.segment_models[segment_name] = self._train_segment_model(segment_df, segment_name)
        
        meta_features = self._create_meta_features(df)
        self.meta_model = self._build_meta_model(meta_features, df['SettlementValue'])
        self.is_trained = True
        
        end_time = time.time()
        training_time = end_time - start_time
        self._log(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, df):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        original_indices = df.index.copy()
        df = self._preprocess_data(df)
        df = self._engineer_features(df)
        meta_features = self._create_meta_features(df)
        
        try:
            predictions = self.meta_model.predict(meta_features)
            self._log(f"Made predictions for {len(predictions)} records")
            
            if len(predictions) != len(original_indices):
                self._log(f"Warning: Original data had {len(original_indices)} records, but predictions were made for {len(predictions)}")
            
            return predictions
        except Exception as e:
            self._log(f"Error during prediction: {str(e)}")
            return np.array([])
    
    def score(self, df):
        if 'SettlementValue' not in df.columns:
            raise ValueError("Target 'SettlementValue' not found in data")
        
        df_copy = df.copy()
        
        try:
            y_pred = self.predict(df_copy)
            
            if len(y_pred) != len(df_copy):
                self._log(f"Warning: Predictions were made for {len(y_pred)} out of {len(df_copy)} records")
                df_copy = df_copy.iloc[:len(y_pred)]
            
            y_true = df_copy['SettlementValue']
            
            min_len = min(len(y_true), len(y_pred))
            return r2_score(y_true[:min_len], y_pred[:min_len])
            
        except Exception as e:
            self._log(f"Error during scoring: {str(e)}")
            return float('nan')
    
    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        joblib.dump(self, filepath)
        self._log(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def evaluate_model(model, test_df, log_file=None):
    predictions = model.predict(test_df)
    y_true = test_df['SettlementValue']
    
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    report = f"""
=== MODEL EVALUATION REPORT ===
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of test records: {len(test_df)}

Performance Metrics:
  R² Score: {r2:.4f}
  RMSE: {rmse:.2f}
  MAE: {mae:.2f}

Segmentation Info:
  Segmentation columns: {model.segmentation_cols}
  Number of segments: {len(model.segments)}
  
Segment Details:
"""
    
    for segment_name, model_info in model.segment_models.items():
        segment_size = model_info['size']
        segment_r2 = model_info['metrics']['r2']
        segment_rmse = model_info['metrics']['rmse']
        
        report += f"  {segment_name}: size={segment_size}, R²={segment_r2:.4f}, RMSE={segment_rmse:.2f}\n"
    
    print(report)
    
    if log_file:
        with open(log_file, 'w') as f:
            f.write(report)
        print(f"Evaluation report saved to {log_file}")
    
    return r2

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    
    df = pd.read_csv('Synthetic_Data_For_Students.csv')
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    model = UnifiedSegmentedModel(
        segmentation_cols=['Exceptional_Circumstances', 'AccidentType'],
        min_segment_size=50,
        use_feature_selection=True,
        verbose=True
    )
    
    model.fit(train_df)
    
    evaluate_model(model, test_df, log_file='output/model_evaluation.txt')
    
    model.save('output/unified_segmented_model.pkl')
    
    print("Model training and evaluation complete.")