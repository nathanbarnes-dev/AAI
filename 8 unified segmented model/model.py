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
        self.required_columns = []
        self.is_trained = False
        self.fallback_mean = None
        
    def _log(self, message):
        if self.verbose:
            print(message)
    
    def _preprocess_data(self, df):
        df = df.copy()
        
        # Store the fallback mean during training
        if not self.is_trained and 'SettlementValue' in df.columns:
            self.fallback_mean = df['SettlementValue'].mean()
        
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
                # If dates can't be parsed, add a default value
                df_new['Days_To_Claim'] = 0
        
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
            # Avoid division by zero
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
        
        # Store required columns during training
        if not self.is_trained:
            self.required_columns = list(df.columns)
        
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
        
        all_segmented = pd.concat([segment_df for segment_df in segments.values()]) if segments else pd.DataFrame()
        
        if not all_segmented.empty:
            other_records = df.loc[~df.index.isin(all_segmented.index)]
            
            if len(other_records) >= self.min_segment_size:
                segments['Other'] = other_records
            elif len(other_records) > 0 and segments:
                # Add remaining records to the largest segment
                largest_segment = max(segments.items(), key=lambda x: len(x[1]))
                segments[largest_segment[0]] = pd.concat([largest_segment[1], other_records])
        
        self._log(f"Created {len(segments)} segments")
        return segments
    
    def _get_segment_for_record(self, record):
        """Determine which segment a single record belongs to"""
        if len(self.segmentation_cols) == 1:
            col = self.segmentation_cols[0]
            if col in record:
                value = record[col]
                segment_id = f"{value}"
                if segment_id in self.segments:
                    return segment_id
        
        else:
            segment_id = ''
            all_cols_present = True
            
            for col in self.segmentation_cols:
                if col not in record:
                    all_cols_present = False
                    break
                segment_id += str(record[col]) + '_'
            
            if all_cols_present and segment_id in self.segments:
                return segment_id
        
        return 'Other'
    
    def _select_features(self, X, y):
        """Select the most important features for a segment"""
        if X.shape[1] <= 5:
            return X, X.columns.tolist()
        
        try:
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
                # Ensure at least 3 features are selected
                importances = feature_selector.feature_importances_
                top_indices = np.argsort(importances)[-3:]
                selected_features = X.columns[top_indices].tolist()
                
            return X[selected_features], selected_features
        except Exception as e:
            self._log(f"Feature selection failed: {str(e)}")
            # In case of error, return original features
            return X, X.columns.tolist()
    
    def _train_segment_model(self, segment_df, segment_name):
        """Train a model for a specific segment"""
        self._log(f"Training model for segment: {segment_name} (n={len(segment_df)})")
        
        numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        X = segment_df[numeric_cols].copy()
        y = segment_df['SettlementValue']
        
        if self.use_feature_selection and X.shape[1] > 5:
            try:
                X_selected, selected_features = self._select_features(X, y)
            except Exception as e:
                self._log(f"Feature selection error: {str(e)}")
                X_selected = X
                selected_features = X.columns.tolist()
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        try:
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
            
            # Store the mean value for fallback
            segment_mean = y.mean()
            
            return {
                'model': model,
                'features': selected_features,
                'metrics': {
                    'rmse': rmse,
                    'r2': r2
                },
                'size': len(segment_df),
                'mean_value': segment_mean
            }
        except Exception as e:
            self._log(f"Error training model for segment {segment_name}: {str(e)}")
            # Return a placeholder with the mean value
            return {
                'model': None,
                'features': [],
                'metrics': {
                    'rmse': float('inf'),
                    'r2': 0
                },
                'size': len(segment_df),
                'mean_value': y.mean()
            }
    
    def _create_meta_features(self, df):
        """Create meta-features for the stacking ensemble"""
        self._log("Creating meta-features for stacking ensemble")
        
        df_meta = df.copy()
        
        # Identify which segment each record belongs to
        try:
            if len(df_meta) == 1:
                # Single record prediction
                df_meta['segment_id'] = self._get_segment_for_record(df_meta.iloc[0])
            else:
                # Batch prediction
                df_meta['segment_id'] = df_meta.apply(self._get_segment_for_record, axis=1)
        except Exception as e:
            self._log(f"Error assigning segments: {str(e)}")
            df_meta['segment_id'] = 'Other'
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if 'SettlementValue' in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        # Create prediction columns for each segment model
        for segment_name, model_info in self.segment_models.items():
            df_meta[f'pred_{segment_name}'] = np.nan
            
            if model_info['model'] is None:
                # Use mean value if model is None
                segment_mask = df_meta['segment_id'] == segment_name
                if any(segment_mask):
                    df_meta.loc[segment_mask, f'pred_{segment_name}'] = model_info['mean_value']
                continue
                
            segment_model = model_info['model']
            segment_features = model_info['features']
            
            # Check if all required features are present
            missing_features = [f for f in segment_features if f not in df_meta.columns]
            if missing_features:
                self._log(f"Warning: Missing features for segment {segment_name}: {missing_features}")
                continue
            
            # Apply segment model to matching records
            segment_mask = df_meta['segment_id'] == segment_name
            
            if any(segment_mask):
                try:
                    X_segment = df_meta.loc[segment_mask, segment_features]
                    preds = segment_model.predict(X_segment)
                    df_meta.loc[segment_mask, f'pred_{segment_name}'] = preds
                except Exception as e:
                    self._log(f"Error predicting for segment {segment_name}: {str(e)}")
                    # Fallback to mean value
                    df_meta.loc[segment_mask, f'pred_{segment_name}'] = model_info['mean_value']
        
        # Create aggregate meta-features
        df_meta['primary_segment_pred'] = self.fallback_mean if self.fallback_mean is not None else 0
        
        if 'Total_Special_Damages' in df_meta.columns and 'Total_General_Damages' in df_meta.columns:
            df_meta['total_damages_meta'] = df_meta['Total_Special_Damages'] + df_meta['Total_General_Damages']
        
        meta_features = [col for col in df_meta.columns if 'pred_' in col or 'meta' in col or 
                        col in ['Total_Special_Damages', 'Total_General_Damages']]
        
        if not meta_features:
            self._log("Warning: No meta-features created, using numeric columns")
            meta_features = numeric_cols
        
        final_features = [col for col in meta_features if col in df_meta.columns]
        
        # Prepare return value and handle missing columns
        meta_df = df_meta[final_features].copy()
        meta_df = meta_df.fillna(0)
        
        return meta_df
    
    def _build_meta_model(self, X_meta, y):
        """Build the meta-model that combines segment predictions"""
        self._log("Building meta-model")
        
        try:
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
            
            # Convert to numpy arrays to avoid pandas-related serialization issues
            X_meta_np = X_meta.to_numpy()
            y_np = y.to_numpy()
            
            stacking_regressor.fit(X_meta_np, y_np)
            
            y_pred = stacking_regressor.predict(X_meta_np)
            rmse = np.sqrt(mean_squared_error(y_np, y_pred))
            r2 = r2_score(y_np, y_pred)
            
            self._log(f"Meta-model performance - RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return stacking_regressor, list(X_meta.columns)
        
        except Exception as e:
            self._log(f"Error building meta-model: {str(e)}")
            # Return a simple XGBoost model as fallback
            fallback_model = XGBRegressor(random_state=42)
            fallback_model.fit(X_meta.to_numpy(), y.to_numpy())
            return fallback_model, list(X_meta.columns)
    
    def fit(self, df):
        """Train the unified segmented model"""
        start_time = time.time()
        self._log("Training unified segmented model")
        
        # Make sure we're working with a copy
        df = df.copy()
        
        # Preprocess and engineer features
        df = self._preprocess_data(df)
        df = self._engineer_features(df)
        
        # Store feature columns for later verification
        self.feature_columns = df.columns.tolist()
        
        # Create segments
        self.segments = self._create_segments(df)
        
        # Train segment models
        self.segment_models = {}
        for segment_name, segment_df in self.segments.items():
            self.segment_models[segment_name] = self._train_segment_model(segment_df, segment_name)
        
        # Create meta-features and train meta-model
        meta_features = self._create_meta_features(df)
        self.meta_model, self.meta_features = self._build_meta_model(meta_features, df['SettlementValue'])
        
        self.is_trained = True
        
        end_time = time.time()
        training_time = end_time - start_time
        self._log(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Support for single-row DataFrame or Series input
        if isinstance(df, pd.Series):
            df = pd.DataFrame([df])
        
        # Handle a single dictionary input
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        
        original_indices = df.index.copy()
        
        try:
            # Preprocess and engineer features
            df = self._preprocess_data(df)
            df = self._engineer_features(df)
            
            # Create meta-features
            meta_features = self._create_meta_features(df)
            
            # Ensure all required meta-features are present
            for feature in self.meta_features:
                if feature not in meta_features.columns:
                    meta_features[feature] = 0
            
            # Keep only the features the meta-model knows about
            meta_features = meta_features[self.meta_features]
            
            # Convert to numpy array for prediction
            X_meta_np = meta_features.to_numpy()
            
            # Make predictions
            predictions = self.meta_model.predict(X_meta_np)
            
            self._log(f"Made predictions for {len(predictions)} records")
            
            # Make sure predictions are positive
            predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            self._log(f"Error during prediction: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            
            # Return fallback prediction
            if self.fallback_mean is not None:
                return np.array([self.fallback_mean] * len(df))
            else:
                return np.array([0] * len(df))
    
    def predict_single(self, record):
        """Make a prediction for a single record"""
        if isinstance(record, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame([record])
        elif isinstance(record, pd.Series):
            # Convert Series to DataFrame
            df = pd.DataFrame([record])
        else:
            raise ValueError("Input must be a dictionary or pandas Series")
        
        # Use the batch prediction method
        predictions = self.predict(df)
        
        # Return the single prediction
        return predictions[0]
    
    def score(self, df):
        """Calculate R² score on a test dataset"""
        if 'SettlementValue' not in df.columns:
            raise ValueError("Target 'SettlementValue' not found in data")
        
        df_copy = df.copy()
        
        try:
            # Make predictions
            y_pred = self.predict(df_copy)
            
            # Handle length mismatch (shouldn't happen with fixed code)
            if len(y_pred) != len(df_copy):
                self._log(f"Warning: Predictions were made for {len(y_pred)} out of {len(df_copy)} records")
                df_copy = df_copy.iloc[:len(y_pred)]
            
            # Get true values
            y_true = df_copy['SettlementValue']
            
            # Calculate and return R² score
            return r2_score(y_true, y_pred)
            
        except Exception as e:
            self._log(f"Error during scoring: {str(e)}")
            return float('nan')
    
    def save(self, filepath):
        """Save the model to disk"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Reset verbosity to avoid large log outputs when loading
        temp_verbose = self.verbose
        self.verbose = False
        
        try:
            joblib.dump(self, filepath)
            self._log(f"Model saved to {filepath}")
        except Exception as e:
            self._log(f"Error saving model: {str(e)}")
            raise
        finally:
            # Restore verbosity
            self.verbose = temp_verbose
    
    @classmethod
    def load(cls, filepath):
        """Load a model from disk"""
        try:
            model = joblib.load(filepath)
            if isinstance(model, cls):
                print(f"Model loaded from {filepath}")
                return model
            else:
                raise TypeError(f"Loaded object is not a {cls.__name__}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load data
    df = pd.read_csv('Synthetic_Data_For_Students.csv')
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and train model
    model = UnifiedSegmentedModel(
        segmentation_cols=['Exceptional_Circumstances', 'AccidentType'],
        min_segment_size=50,
        use_feature_selection=True,
        verbose=True
    )
    
    model.fit(train_df)
    
    # Evaluate model
    r2 = model.score(test_df)
    print(f"Test R² score: {r2:.4f}")
    
    # Save model
    model.save('output/unified_segmented_model.pkl')
    
    # Test single prediction
    single_record = test_df.iloc[0]
    prediction = model.predict_single(single_record)
    print(f"Prediction for single record: {prediction:.2f}")
    print(f"Actual value: {single_record['SettlementValue']:.2f}")
    
    # Test loading model
    loaded_model = UnifiedSegmentedModel.load('output/unified_segmented_model.pkl')
    loaded_prediction = loaded_model.predict_single(single_record)
    print(f"Prediction from loaded model: {loaded_prediction:.2f}")