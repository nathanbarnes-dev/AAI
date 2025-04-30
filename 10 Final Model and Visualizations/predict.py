import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings
from datetime import datetime

# Add current directory to path to ensure model.py can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the UnifiedSegmentedModel class from model.py
# This is needed for joblib to properly load the model
from model import UnifiedSegmentedModel

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    def __init__(self, model_path):
        """
        Initialize the explainer with a trained UnifiedSegmentedModel
        
        Args:
            model_path: Path to the .pkl file containing the trained model
        """
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model has {len(self.model.segments)} segments")
        
    def predict_and_explain(self, record, output_dir='model_explanation'):
        """
        Make a prediction for a single record and generate explanations
        
        Args:
            record: DataFrame containing a single record to predict
            output_dir: Directory to save explanation visualizations
        
        Returns:
            Dictionary containing prediction and explanation data
        """
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Make prediction
        if isinstance(record, pd.Series):
            record = pd.DataFrame([record])
            
        # Get the prediction
        prediction = self.model.predict(record)[0]
        
        # Preprocess the record as the model would
        processed_record = self._preprocess_record(record)
        
        # Determine which segment was used
        segment_id = self._get_segment_id(processed_record.iloc[0])
        
        # Get segment-specific prediction
        segment_prediction = self._get_segment_prediction(processed_record, segment_id)
        
        # Generate visualizations
        self._visualize_segment_contributions(processed_record, segment_id, prediction, output_dir)
        self._visualize_feature_importance(processed_record, segment_id, output_dir)
        self._visualize_segment_details(processed_record, segment_id, prediction, segment_prediction, output_dir)
        
        # Create text summary
        self._create_prediction_summary(record, processed_record, prediction, segment_id, segment_prediction, output_dir)
        
        return {
            'prediction': prediction,
            'segment_id': segment_id,
            'segment_prediction': segment_prediction
        }
    
    def _preprocess_record(self, record):
        """Apply the same preprocessing steps the model would use"""
        record_copy = record.copy()
        
        # Apply preprocessing from the model
        record_copy = self.model._preprocess_data(record_copy)
        record_copy = self.model._engineer_features(record_copy)
        
        return record_copy
    
    def _get_segment_id(self, record):
        """Determine which segment the record belongs to"""
        return self.model._get_segment_for_record(record)
    
    def _get_segment_prediction(self, record, segment_id):
        """Get the prediction from just the segment model"""
        if segment_id in self.model.segment_models:
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            # Check if all features exist in the record
            missing_features = [f for f in segment_features if f not in record.columns]
            if missing_features:
                print(f"Warning: Missing features for segment {segment_id}: {missing_features}")
                return None
            
            return segment_model.predict(record[segment_features])[0]
        
        return None
    
    def _visualize_segment_contributions(self, record, segment_id, final_prediction, output_dir):
        """
        Create visualization showing how different segments contribute to the final prediction
        """
        # Create meta features as the model would
        meta_features = self.model._create_meta_features(record)
        
        # Get the predictions from all segment models
        segment_predictions = {}
        for seg_id, model_info in self.model.segment_models.items():
            pred_col = f'pred_{seg_id}'
            if pred_col in meta_features.columns:
                segment_predictions[seg_id] = meta_features[pred_col].iloc[0]
            else:
                segment_predictions[seg_id] = None
        
        # Filter out None values
        segment_predictions = {k: v for k, v in segment_predictions.items() if v is not None}
        
        # Sort segments by prediction value
        sorted_segments = dict(sorted(segment_predictions.items(), key=lambda x: x[1], reverse=True))
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bars
        names = list(sorted_segments.keys())
        values = list(sorted_segments.values())
        y_pos = np.arange(len(names))
        
        # Create the bars with a light color
        bars = plt.barh(y_pos, values, alpha=0.6)
        
        # Highlight the segment used for this record
        for i, seg_id in enumerate(names):
            if seg_id == segment_id:
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
                plt.text(values[i] - (max(values) * 0.1), i, '← Used for prediction', 
                        ha='right', va='center', color='white', fontweight='bold', fontsize=12,
                        bbox=dict(facecolor='red', alpha=0.6, boxstyle='round,pad=0.3'))
        
        # Add line for final prediction 
        plt.axvline(x=final_prediction, color='green', linestyle='-', linewidth=3, 
                  label=f'Final Prediction: ${final_prediction:.2f}')
        
        # Add labels and title
        plt.yticks(y_pos, names)
        plt.xlabel('Prediction Value ($)')
        plt.title('Segment Model Predictions', fontsize=16, pad=20)
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add grid for easier reading
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            plt.text(v + (max(values) * 0.01), i, f'${v:.2f}', va='center')
        
        # Ensure there's some margin to the right
        plt.xlim(0, max(values) * 1.15)
        
        # Add explanation text
        explanation = (
            "This chart shows predictions from all segment models.\n"
            "The highlighted segment was chosen based on your input factors.\n"
            "The green line shows the final prediction after ensemble processing."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.6))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'segment_contributions.png'), dpi=150)
        plt.close()

    def _visualize_feature_importance(self, record, segment_id, output_dir):
        """
        Create visualization showing feature importance for the segment model
        """
        if segment_id in self.model.segment_models:
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            # Only create visualization if we're using XGBoost which has feature_importances_
            if hasattr(segment_model, 'feature_importances_'):
                importances = segment_model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[-15:]  # Show top 15 features
                features = [segment_features[i] for i in indices]
                importance_values = importances[indices]
                
                # Get the feature values for the current record
                feature_values = []
                for feature in features:
                    if feature in record.columns:
                        value = record[feature].iloc[0]
                        if isinstance(value, (int, float)):
                            value_text = f"{value:.2f}" if isinstance(value, float) else f"{value}"
                        else:
                            value_text = str(value)
                        feature_values.append(value_text)
                    else:
                        feature_values.append("N/A")
                
                # Create a DataFrame for better visualization
                feature_df = pd.DataFrame({
                    'Feature': features, 
                    'Importance': importance_values,
                    'Value': feature_values
                })
                
                # Sort by importance
                feature_df = feature_df.sort_values('Importance', ascending=True)
                
                # Create the visualization
                plt.figure(figsize=(12, 10))
                
                # Create colormap based on importance
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(feature_df)))
                
                # Create horizontal bars
                bars = plt.barh(feature_df['Feature'], feature_df['Importance'], color=colors)
                
                # Add feature values as text on the right side of the chart
                for i, (_, row) in enumerate(feature_df.iterrows()):
                    plt.text(row['Importance'] + max(feature_df['Importance']) * 0.01, i, 
                            f"Value: {row['Value']}", va='center')
                
                # Add title and labels
                plt.title(f'Top Features for Segment: {segment_id}', fontsize=16, pad=20)
                plt.xlabel('Importance')
                
                # Add grid for easier reading
                plt.grid(axis='x', linestyle='--', alpha=0.4)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
                plt.close()
                
    def _visualize_segment_details(self, record, segment_id, prediction, segment_prediction, output_dir):
        """
        Create in-depth visualization of the specific segment used for prediction
        """
        # Get the original segments from the model
        segments = self.model.segments
        
        # Get the segmentation criteria
        segmentation_cols = self.model.segmentation_cols
        
        # Get the segment data if available
        if segment_id in segments:
            segment_data = segments[segment_id]
            segment_size = len(segment_data)
        else:
            segment_size = "Unknown"
        
        # Get the segment model details
        segment_model_info = self.model.segment_models.get(segment_id, {})
        segment_metrics = segment_model_info.get('metrics', {})
        segment_r2 = segment_metrics.get('r2', "Unknown")
        segment_rmse = segment_metrics.get('rmse', "Unknown")
        
        # Create the visualization
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1.5]})
        
        # 1. Segment Information Box (Top Left)
        axs[0, 0].axis('off')
        info_text = [
            f"SEGMENT INFORMATION",
            f"------------------------",
            f"Segment ID: {segment_id}",
            f"Segment Size: {segment_size} records",
            f"Segment R²: {segment_r2:.4f}" if isinstance(segment_r2, (int, float)) else f"Segment R²: {segment_r2}",
            f"Segment RMSE: {segment_rmse:.2f}" if isinstance(segment_rmse, (int, float)) else f"Segment RMSE: {segment_rmse}",
            f"",
            f"SEGMENTATION CRITERIA:"
        ]
        
        # Add segmentation criteria values
        for col in segmentation_cols:
            if col in record.columns:
                info_text.append(f"  {col}: {record[col].iloc[0]}")
        
        # Add the text to the axis
        axs[0, 0].text(0, 1, '\n'.join(info_text), fontsize=12, verticalalignment='top')
        
        # 2. Prediction Comparison (Top Right)
        if segment_prediction is not None:
            axs[0, 1].bar(['Segment Model', 'Final Prediction'], 
                        [segment_prediction, prediction], 
                        color=['#1f77b4', '#ff7f0e'])
            
            # Add value labels on top of bars
            axs[0, 1].text(0, segment_prediction * 1.01, f'${segment_prediction:.2f}', 
                         ha='center', va='bottom', fontweight='bold')
            axs[0, 1].text(1, prediction * 1.01, f'${prediction:.2f}', 
                         ha='center', va='bottom', fontweight='bold')
            
            # Add title and adjust y-axis
            axs[0, 1].set_title('Segment Model vs Final Prediction', fontsize=14)
            axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set the y-axis to start from 0
            y_max = max(segment_prediction, prediction) * 1.2
            axs[0, 1].set_ylim(0, y_max)
        else:
            axs[0, 1].text(0.5, 0.5, "Segment prediction not available", 
                         ha='center', va='center', fontsize=14)
            axs[0, 1].axis('off')
        
        # 3. Distribution of Settlement Values in Segment (Bottom Left)
        try:
            if segment_id in segments:
                segment_df = segments[segment_id]
                if 'SettlementValue' in segment_df.columns:
                    # Create a histogram of settlement values
                    settlement_values = segment_df['SettlementValue']
                    axs[1, 0].hist(settlement_values, bins=20, alpha=0.7, color='#2ca02c')
                    
                    # Add markers for the current prediction and segment prediction
                    if segment_prediction is not None:
                        axs[1, 0].axvline(x=segment_prediction, color='#1f77b4', linestyle='--', 
                                        linewidth=2, label=f'Segment Prediction: ${segment_prediction:.2f}')
                    
                    axs[1, 0].axvline(x=prediction, color='#ff7f0e', linestyle='-', 
                                    linewidth=2, label=f'Final Prediction: ${prediction:.2f}')
                    
                    # Add labels and title
                    axs[1, 0].set_xlabel('Settlement Value ($)')
                    axs[1, 0].set_ylabel('Frequency')
                    axs[1, 0].set_title('Distribution of Settlement Values in Segment', fontsize=14)
                    axs[1, 0].legend()
                    
                    # Add vertical grid lines
                    axs[1, 0].grid(axis='x', linestyle='--', alpha=0.5)
                else:
                    axs[1, 0].text(0.5, 0.5, "SettlementValue not available in segment data", 
                                ha='center', va='center', fontsize=12)
                    axs[1, 0].axis('off')
            else:
                axs[1, 0].text(0.5, 0.5, "Segment data not available", 
                            ha='center', va='center', fontsize=12)
                axs[1, 0].axis('off')
        except Exception as e:
            axs[1, 0].text(0.5, 0.5, f"Error creating distribution: {str(e)}", 
                        ha='center', va='center', fontsize=10)
            axs[1, 0].axis('off')
        
        # 4. Key Feature Values in Current Record vs. Segment Average (Bottom Right)
        try:
            if segment_id in segments and segment_id in self.model.segment_models:
                segment_df = segments[segment_id]
                segment_model = self.model.segment_models[segment_id]['model']
                segment_features = self.model.segment_models[segment_id]['features']
                
                if hasattr(segment_model, 'feature_importances_'):
                    # Get top 5 features by importance
                    importances = segment_model.feature_importances_
                    indices = np.argsort(importances)[-5:]  # Top 5 features
                    top_features = [segment_features[i] for i in indices]
                    
                    # Compare record values to segment averages for these features
                    feature_comparison = []
                    feature_names = []
                    
                    for feature in top_features:
                        if feature in record.columns and feature in segment_df.columns:
                            if pd.api.types.is_numeric_dtype(segment_df[feature]):
                                # For numeric features, get the current value and segment average
                                current_value = record[feature].iloc[0]
                                segment_avg = segment_df[feature].mean()
                                feature_comparison.append((current_value, segment_avg))
                                feature_names.append(feature)
                    
                    if feature_comparison:
                        # Create a grouped bar chart
                        x = np.arange(len(feature_names))
                        width = 0.35
                        
                        current_values = [fc[0] for fc in feature_comparison]
                        segment_avgs = [fc[1] for fc in feature_comparison]
                        
                        axs[1, 1].bar(x - width/2, current_values, width, label='Current Record', color='#d62728')
                        axs[1, 1].bar(x + width/2, segment_avgs, width, label='Segment Average', color='#9467bd')
                        
                        # Set the labels and title
                        axs[1, 1].set_xlabel('Feature')
                        axs[1, 1].set_ylabel('Value')
                        axs[1, 1].set_title('Key Feature Values: Current vs. Segment Average', fontsize=14)
                        axs[1, 1].set_xticks(x)
                        axs[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
                        axs[1, 1].legend()
                        
                        # Add horizontal grid lines
                        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
                    else:
                        axs[1, 1].text(0.5, 0.5, "No comparable numeric features available", 
                                    ha='center', va='center', fontsize=12)
                        axs[1, 1].axis('off')
                else:
                    axs[1, 1].text(0.5, 0.5, "Feature importance not available", 
                                ha='center', va='center', fontsize=12)
                    axs[1, 1].axis('off')
            else:
                axs[1, 1].text(0.5, 0.5, "Segment data or model not available", 
                            ha='center', va='center', fontsize=12)
                axs[1, 1].axis('off')
        except Exception as e:
            axs[1, 1].text(0.5, 0.5, f"Error creating feature comparison: {str(e)}", 
                        ha='center', va='center', fontsize=10)
            axs[1, 1].axis('off')
        
        # Main title for the entire figure
        plt.suptitle(f'Detailed Analysis of Segment: {segment_id}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'segment_details.png'), dpi=150)
        plt.close()

    def _create_prediction_summary(self, original_record, processed_record, prediction, segment_id, segment_prediction, output_dir):
        """Create a detailed text summary of the prediction and key factors"""
        summary = {
            'Prediction': prediction,
            'Segment Used': segment_id,
            'Segment-only Prediction': segment_prediction if segment_prediction is not None else "N/A"
        }
        
        # Add key features and their values
        if segment_id in self.model.segment_models:
            segment_features = self.model.segment_models[segment_id]['features']
            
            # If we have model with feature importances
            if hasattr(self.model.segment_models[segment_id]['model'], 'feature_importances_'):
                importances = self.model.segment_models[segment_id]['model'].feature_importances_
                
                # Get top 10 most important features
                indices = np.argsort(importances)[-10:]
                top_features = [segment_features[i] for i in indices]
                
                summary['Key Features'] = {}
                for feature in top_features:
                    if feature in processed_record.columns:
                        summary['Key Features'][feature] = processed_record[feature].iloc[0]
        
        # Original input values
        summary['Original Input'] = {}
        
        # Add segmentation columns
        for col in self.model.segmentation_cols:
            if col in original_record.columns:
                summary['Original Input'][col] = original_record[col].iloc[0]
        
        # Add special damages
        special_cols = [col for col in original_record.columns if 'Special_' in col]
        if special_cols:
            summary['Special Damages'] = {}
            for col in special_cols:
                summary['Special Damages'][col] = original_record[col].iloc[0]
            
            # Calculate total special damages
            summary['Special Damages']['Total Special Damages'] = sum(original_record[col].iloc[0] for col in special_cols)
        
        # Add general damages
        general_cols = [col for col in original_record.columns if 'General_' in col]
        if general_cols:
            summary['General Damages'] = {}
            for col in general_cols:
                summary['General Damages'][col] = original_record[col].iloc[0]
            
            # Calculate total general damages
            summary['General Damages']['Total General Damages'] = sum(original_record[col].iloc[0] for col in general_cols)
        
        # Add total damages and ratio
        if 'Special Damages' in summary and 'General Damages' in summary:
            total_special = summary['Special Damages']['Total Special Damages']
            total_general = summary['General Damages']['Total General Damages']
            
            summary['Damage Summary'] = {
                'Total Damages': total_special + total_general,
                'Special/General Ratio': total_special / total_general if total_general > 0 else "N/A"
            }
        
        # Add other factors
        other_factors = {}
        for col in original_record.columns:
            if col not in self.model.segmentation_cols and 'Special_' not in col and 'General_' not in col:
                other_factors[col] = original_record[col].iloc[0]
        
        if other_factors:
            summary['Other Factors'] = other_factors
        
        # Save summary to file
        with open(os.path.join(output_dir, 'prediction_summary.txt'), 'w') as f:
            f.write("PREDICTION SUMMARY\n")
            f.write("=================\n\n")
            
            f.write(f"Final Prediction: ${summary['Prediction']:.2f}\n")
            f.write(f"Segment Used: {summary['Segment Used']}\n")
            f.write(f"Segment-only Prediction: {summary['Segment-only Prediction']}\n\n")
            
            if 'Key Features' in summary:
                f.write("KEY FEATURES (in order of importance):\n")
                for feature, value in summary['Key Features'].items():
                    if isinstance(value, float):
                        f.write(f"  {feature}: {value:.4f}\n")
                    else:
                        f.write(f"  {feature}: {value}\n")
                f.write("\n")
            
            if 'Original Input' in summary:
                f.write("SEGMENT CRITERIA:\n")
                for feature, value in summary['Original Input'].items():
                    f.write(f"  {feature}: {value}\n")
                f.write("\n")
            
            if 'Special Damages' in summary:
                f.write("SPECIAL DAMAGES:\n")
                for item, value in summary['Special Damages'].items():
                    f.write(f"  {item}: ${value:.2f}\n")
                f.write("\n")
            
            if 'General Damages' in summary:
                f.write("GENERAL DAMAGES:\n")
                for item, value in summary['General Damages'].items():
                    f.write(f"  {item}: ${value:.2f}\n")
                f.write("\n")
            
            if 'Damage Summary' in summary:
                f.write("DAMAGE SUMMARY:\n")
                for item, value in summary['Damage Summary'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {item}: ${value:.2f}\n")
                    else:
                        f.write(f"  {item}: {value}\n")
                f.write("\n")
            
            if 'Other Factors' in summary:
                f.write("OTHER FACTORS:\n")
                for item, value in summary['Other Factors'].items():
                    f.write(f"  {item}: {value}\n")


def create_synthetic_record():
    """Create a synthetic record with realistic default values for a vehicle accident case"""
    return pd.DataFrame({
        # Segmentation columns
        'Exceptional_Circumstances': ['No'],
        'AccidentType': ['Vehicle'],
        
        # Dates
        'Accident Date': ['2023-01-15'],
        'Claim Date': ['2023-02-25'],  # 40 days to claim
        
        # Special damages (economic losses)
        'Special_Medical': [8250],      # Medical expenses
        'Special_LossOfEarnings': [6300],  # Lost wages (3 weeks at average weekly wage)
        'Special_Care': [1200],         # Home care and assistance
        'Special_Other': [850],         # Travel to appointments, medication, etc.
        
        # General damages (non-economic losses)
        'General_Pain': [12500],        # Physical pain
        'General_Suffering': [9000],    # Emotional suffering
        'General_LossOfAmenity': [7500],  # Loss of enjoyment of life
        
        # Injury details
        'Whiplash': ['Yes'],
        'Minor_Psychological_Injury': ['Yes'],
        'Permanent_Injury': ['No'],
        'Recovery_Duration': [180],     # Expected recovery in days
        
        # Other potentially relevant factors
        'Age': [37],
        'Sex': ['Female'],
        'Employed': ['Yes'],
        'Dependents': [2]
    })


def load_sample_record():
    """Load a sample record from the training data"""
    try:
        df = pd.read_csv('Synthetic_Data_For_Students.csv')
        return df.iloc[[0]]  # Use the first row as sample
    except FileNotFoundError:
        print("Error: Could not find Synthetic_Data_For_Students.csv")
        print("Creating a synthetic record instead")
        return create_synthetic_record()


def collect_manual_input():
    """Collect record values through manual input"""
    print("\n=== Enter values for the record ===")
    
    record = {}
    
    # Numeric variables (with validation)
    def get_numeric_input(prompt, default):
        while True:
            try:
                value = input(f"{prompt} [default: {default}]: ") or default
                return float(value)
            except ValueError:
                print("Please enter a numeric value")
    
    # Segmentation columns
    print("\n\033[1mSegmentation Categories:\033[0m")
    record['Exceptional_Circumstances'] = input("Exceptional Circumstances (Yes/No) [default: No]: ") or "No"
    record['AccidentType'] = input("Accident Type (Vehicle/Work/Premises/Other) [default: Vehicle]: ") or "Vehicle"
    
    # Dates
    print("\n\033[1mDates:\033[0m")
    record['Accident Date'] = input("Accident Date (YYYY-MM-DD) [default: 2023-01-15]: ") or "2023-01-15"
    record['Claim Date'] = input("Claim Date (YYYY-MM-DD) [default: 2023-02-25]: ") or "2023-02-25"
    
    # Special damages
    print("\n\033[1mSpecial Damages (Economic Losses):\033[0m")
    record['Special_Medical'] = get_numeric_input("Medical Expenses ($)", 8250)
    record['Special_LossOfEarnings'] = get_numeric_input("Loss of Earnings ($)", 6300)
    record['Special_Care'] = get_numeric_input("Care Costs ($)", 1200)
    record['Special_Other'] = get_numeric_input("Other Special Damages ($)", 850)
    
    # General damages
    print("\n\033[1mGeneral Damages (Non-Economic Losses):\033[0m")
    record['General_Pain'] = get_numeric_input("Pain ($)", 12500)
    record['General_Suffering'] = get_numeric_input("Suffering ($)", 9000)
    record['General_LossOfAmenity'] = get_numeric_input("Loss of Amenity ($)", 7500)
    
    # Injury details
    print("\n\033[1mInjury Details:\033[0m")
    record['Whiplash'] = input("Whiplash (Yes/No) [default: Yes]: ") or "Yes"
    record['Minor_Psychological_Injury'] = input("Minor Psychological Injury (Yes/No) [default: Yes]: ") or "Yes"
    record['Permanent_Injury'] = input("Permanent Injury (Yes/No) [default: No]: ") or "No"
    record['Recovery_Duration'] = get_numeric_input("Expected Recovery Duration (days)", 180)
    
    # Other factors
    print("\n\033[1mOther Factors:\033[0m")
    record['Age'] = get_numeric_input("Age")

def print_colored_summary(result, record, output_dir):
    """Print a colored summary of the prediction and explanation"""
    print("\n" + "="*60)
    print("\033[1m\033[94mMODEL PREDICTION SUMMARY\033[0m")
    print("="*60)
    
    print(f"\033[1mPrediction:\033[0m \033[92m${result['prediction']:.2f}\033[0m")
    print(f"\033[1mSegment Used:\033[0m \033[93m{result['segment_id']}\033[0m")
    if result['segment_prediction'] is not None:
        print(f"\033[1mSegment-only Prediction:\033[0m \033[96m${result['segment_prediction']:.2f}\033[0m")
    
    print("\n\033[1m\033[95mINPUT RECORD SUMMARY:\033[0m")
    print("-"*60)
    
    # Display the segmentation columns prominently
    segmentation_cols = ['Exceptional_Circumstances', 'AccidentType']
    for col in segmentation_cols:
        if col in record.columns:
            print(f"\033[1m{col}:\033[0m \033[93m{record[col].iloc[0]}\033[0m")
    
    # Calculate totals for damages
    total_special = sum(record[col].iloc[0] for col in record.columns if 'Special_' in col)
    total_general = sum(record[col].iloc[0] for col in record.columns if 'General_' in col)
    
    # Display damages with totals
    print(f"\n\033[1mSpecial Damages (Total: \033[92m${total_special:.2f}\033[0m):\033[0m")
    special_cols = [col for col in record.columns if 'Special_' in col]
    for col in special_cols:
        print(f"  {col}: ${record[col].iloc[0]:.2f}")
    
    print(f"\n\033[1mGeneral Damages (Total: \033[92m${total_general:.2f}\033[0m):\033[0m")
    general_cols = [col for col in record.columns if 'General_' in col]
    for col in general_cols:
        print(f"  {col}: ${record[col].iloc[0]:.2f}")
    
    # Display total of all damages
    total_damages = total_special + total_general
    print(f"\n\033[1mTotal Damages: \033[92m${total_damages:.2f}\033[0m")
    
    # Display ratio if both special and general damages exist
    if total_special > 0 and total_general > 0:
        ratio = total_special / total_general
        print(f"\033[1mSpecial/General Ratio: \033[92m{ratio:.2f}\033[0m")
    
    print("\n\033[1mOther Factors:\033[0m")
    other_cols = [col for col in record.columns 
                 if col not in segmentation_cols and 'Special_' not in col and 'General_' not in col]
    for col in other_cols:
        print(f"  {col}: {record[col].iloc[0]}")
    
    print("\n" + "="*60)
    print(f"\033[1mExplanation files saved to:\033[0m \033[94m{os.path.abspath(output_dir)}\033[0m")
    print("="*60)


def main():
    # Set default values
    model_path = 'output/unified_segmented_model.pkl'
    output_dir = 'model_explanation'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print welcome message
    print("="*60)
    print("UNIFIED SEGMENTED MODEL PREDICTION EXPLAINER")
    print("="*60)
    print("\nThis tool will make a prediction using the model and explain the factors that influenced it.\n")
    
    # Load the model
    try:
        explainer = ModelExplainer(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Make sure the model file exists at: {os.path.abspath(model_path)}")
        input("\nPress Enter to exit...")
        exit(1)
    
    # Ask user which method to use for input
    print("\nHow would you like to provide the input data?")
    print("1. Use a sample from the training data (default)")
    print("2. Enter values manually")
    
    choice = input("\nEnter your choice (1-2) [default: 1]: ") or "1"
    
    # Get the record based on the user's choice
    if choice == "1":
        record = load_sample_record()
        print("Using sample record from training data")
    elif choice == "2":
        record = collect_manual_input()
        print("Using manually entered record")
    else:
        print(f"Invalid choice '{choice}'. Using sample record instead.")
        record = load_sample_record()
    
    # Make prediction and generate explanations
    result = explainer.predict_and_explain(record, output_dir)
    
    # Print colored summary
    print_colored_summary(result, record, output_dir)
    
    # Display the paths to the generated files
    print("\nExplanation files:")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")
    
    # Keep the console window open until the user presses Enter
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()

# Add current directory to path to ensure model.py can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the UnifiedSegmentedModel class from model.py
# This is needed for joblib to properly load the model
from model import UnifiedSegmentedModel

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    def __init__(self, model_path):
        """
        Initialize the explainer with a trained UnifiedSegmentedModel
        
        Args:
            model_path: Path to the .pkl file containing the trained model
        """
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model has {len(self.model.segments)} segments")
        
    def predict_and_explain(self, record, output_dir='model_explanation'):
        """
        Make a prediction for a single record and generate explanations
        
        Args:
            record: DataFrame containing a single record to predict
            output_dir: Directory to save explanation visualizations
        
        Returns:
            Dictionary containing prediction and explanation data
        """
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Make prediction
        if isinstance(record, pd.Series):
            record = pd.DataFrame([record])
            
        # Get the prediction
        prediction = self.model.predict(record)[0]
        
        # Preprocess the record as the model would
        processed_record = self._preprocess_record(record)
        
        # Determine which segment was used
        segment_id = self._get_segment_id(processed_record.iloc[0])
        
        # Get segment-specific prediction
        segment_prediction = self._get_segment_prediction(processed_record, segment_id)
        
        # Generate visualizations
        self._visualize_segment_contributions(processed_record, segment_id, prediction, output_dir)
        self._visualize_feature_importance(processed_record, segment_id, output_dir)
        self._visualize_segment_details(processed_record, segment_id, prediction, segment_prediction, output_dir)
        
        # Create text summary
        self._create_prediction_summary(record, processed_record, prediction, segment_id, segment_prediction, output_dir)
        
        return {
            'prediction': prediction,
            'segment_id': segment_id,
            'segment_prediction': segment_prediction
        }
    
    def _preprocess_record(self, record):
        """Apply the same preprocessing steps the model would use"""
        record_copy = record.copy()
        
        # Apply preprocessing from the model
        record_copy = self.model._preprocess_data(record_copy)
        record_copy = self.model._engineer_features(record_copy)
        
        return record_copy
    
    def _get_segment_id(self, record):
        """Determine which segment the record belongs to"""
        return self.model._get_segment_for_record(record)
    
    def _get_segment_prediction(self, record, segment_id):
        """Get the prediction from just the segment model"""
        if segment_id in self.model.segment_models:
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            # Check if all features exist in the record
            missing_features = [f for f in segment_features if f not in record.columns]
            if missing_features:
                print(f"Warning: Missing features for segment {segment_id}: {missing_features}")
                return None
            
            return segment_model.predict(record[segment_features])[0]
        
        return None
    
    def _visualize_segment_contributions(self, record, segment_id, final_prediction, output_dir):
        """
        Create visualization showing how different segments contribute to the final prediction
        """
        # Create meta features as the model would
        meta_features = self.model._create_meta_features(record)
        
        # Get the predictions from all segment models
        segment_predictions = {}
        for seg_id, model_info in self.model.segment_models.items():
            pred_col = f'pred_{seg_id}'
            if pred_col in meta_features.columns:
                segment_predictions[seg_id] = meta_features[pred_col].iloc[0]
            else:
                segment_predictions[seg_id] = None
        
        # Filter out None values
        segment_predictions = {k: v for k, v in segment_predictions.items() if v is not None}
        
        # Sort segments by prediction value
        sorted_segments = dict(sorted(segment_predictions.items(), key=lambda x: x[1], reverse=True))
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bars
        names = list(sorted_segments.keys())
        values = list(sorted_segments.values())
        y_pos = np.arange(len(names))
        
        # Create the bars with a light color
        bars = plt.barh(y_pos, values, alpha=0.6)
        
        # Highlight the segment used for this record
        for i, seg_id in enumerate(names):
            if seg_id == segment_id:
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
                plt.text(values[i] - (max(values) * 0.1), i, '← Used for prediction', 
                        ha='right', va='center', color='white', fontweight='bold', fontsize=12,
                        bbox=dict(facecolor='red', alpha=0.6, boxstyle='round,pad=0.3'))
        
        # Add line for final prediction 
        plt.axvline(x=final_prediction, color='green', linestyle='-', linewidth=3, 
                  label=f'Final Prediction: ${final_prediction:.2f}')
        
        # Add labels and title
        plt.yticks(y_pos, names)
        plt.xlabel('Prediction Value ($)')
        plt.title('Segment Model Predictions', fontsize=16, pad=20)
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add grid for easier reading
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            plt.text(v + (max(values) * 0.01), i, f'${v:.2f}', va='center')
        
        # Ensure there's some margin to the right
        plt.xlim(0, max(values) * 1.15)
        
        # Add explanation text
        explanation = (
            "This chart shows predictions from all segment models.\n"
            "The highlighted segment was chosen based on your input factors.\n"
            "The green line shows the final prediction after ensemble processing."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.6))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'segment_contributions.png'), dpi=150)
        plt.close()

    def _visualize_feature_importance(self, record, segment_id, output_dir):
        """
        Create visualization showing feature importance for the segment model
        """
        if segment_id in self.model.segment_models:
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            # Only create visualization if we're using XGBoost which has feature_importances_
            if hasattr(segment_model, 'feature_importances_'):
                importances = segment_model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[-15:]  # Show top 15 features
                features = [segment_features[i] for i in indices]
                importance_values = importances[indices]
                
                # Get the feature values for the current record
                feature_values = []
                for feature in features:
                    if feature in record.columns:
                        value = record[feature].iloc[0]
                        if isinstance(value, (int, float)):
                            value_text = f"{value:.2f}" if isinstance(value, float) else f"{value}"
                        else:
                            value_text = str(value)
                        feature_values.append(value_text)
                    else:
                        feature_values.append("N/A")
                
                # Create a DataFrame for better visualization
                feature_df = pd.DataFrame({
                    'Feature': features, 
                    'Importance': importance_values,
                    'Value': feature_values
                })
                
                # Sort by importance
                feature_df = feature_df.sort_values('Importance', ascending=True)
                
                # Create the visualization
                plt.figure(figsize=(12, 10))
                
                # Create colormap based on importance
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(feature_df)))
                
                # Create horizontal bars
                bars = plt.barh(feature_df['Feature'], feature_df['Importance'], color=colors)
                
                # Add feature values as text on the right side of the chart
                for i, (_, row) in enumerate(feature_df.iterrows()):
                    plt.text(row['Importance'] + max(feature_df['Importance']) * 0.01, i, 
                            f"Value: {row['Value']}", va='center')
                
                # Add title and labels
                plt.title(f'Top Features for Segment: {segment_id}', fontsize=16, pad=20)
                plt.xlabel('Importance')
                
                # Add grid for easier reading
                plt.grid(axis='x', linestyle='--', alpha=0.4)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
                plt.close()
                
    def _visualize_segment_details(self, record, segment_id, prediction, segment_prediction, output_dir):
        """
        Create in-depth visualization of the specific segment used for prediction
        """
        # Get the original segments from the model
        segments = self.model.segments
        
        # Get the segmentation criteria
        segmentation_cols = self.model.segmentation_cols
        
        # Get the segment data if available
        if segment_id in segments:
            segment_data = segments[segment_id]
            segment_size = len(segment_data)
        else:
            segment_size = "Unknown"
        
        # Get the segment model details
        segment_model_info = self.model.segment_models.get(segment_id, {})
        segment_metrics = segment_model_info.get('metrics', {})
        segment_r2 = segment_metrics.get('r2', "Unknown")
        segment_rmse = segment_metrics.get('rmse', "Unknown")
        
        # Create the visualization
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1.5]})
        
        # 1. Segment Information Box (Top Left)
        axs[0, 0].axis('off')
        info_text = [
            f"SEGMENT INFORMATION",
            f"------------------------",
            f"Segment ID: {segment_id}",
            f"Segment Size: {segment_size} records",
            f"Segment R²: {segment_r2:.4f}" if isinstance(segment_r2, (int, float)) else f"Segment R²: {segment_r2}",
            f"Segment RMSE: {segment_rmse:.2f}" if isinstance(segment_rmse, (int, float)) else f"Segment RMSE: {segment_rmse}",
            f"",
            f"SEGMENTATION CRITERIA:"
        ]
        
        # Add segmentation criteria values
        for col in segmentation_cols:
            if col in record.columns:
                info_text.append(f"  {col}: {record[col].iloc[0]}")
        
        # Add the text to the axis
        axs[0, 0].text(0, 1, '\n'.join(info_text), fontsize=12, verticalalignment='top')
        
        # 2. Prediction Comparison (Top Right)
        if segment_prediction is not None:
            axs[0, 1].bar(['Segment Model', 'Final Prediction'], 
                        [segment_prediction, prediction], 
                        color=['#1f77b4', '#ff7f0e'])
            
            # Add value labels on top of bars
            axs[0, 1].text(0, segment_prediction * 1.01, f'${segment_prediction:.2f}', 
                         ha='center', va='bottom', fontweight='bold')
            axs[0, 1].text(1, prediction * 1.01, f'${prediction:.2f}', 
                         ha='center', va='bottom', fontweight='bold')
            
            # Add title and adjust y-axis
            axs[0, 1].set_title('Segment Model vs Final Prediction', fontsize=14)
            axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set the y-axis to start from 0
            y_max = max(segment_prediction, prediction) * 1.2
            axs[0, 1].set_ylim(0, y_max)
        else:
            axs[0, 1].text(0.5, 0.5, "Segment prediction not available", 
                         ha='center', va='center', fontsize=14)
            axs[0, 1].axis('off')
        
        # 3. Distribution of Settlement Values in Segment (Bottom Left)
        try:
            if segment_id in segments:
                segment_df = segments[segment_id]
                if 'SettlementValue' in segment_df.columns:
                    # Create a histogram of settlement values
                    settlement_values = segment_df['SettlementValue']
                    axs[1, 0].hist(settlement_values, bins=20, alpha=0.7, color='#2ca02c')
                    
                    # Add markers for the current prediction and segment prediction
                    if segment_prediction is not None:
                        axs[1, 0].axvline(x=segment_prediction, color='#1f77b4', linestyle='--', 
                                        linewidth=2, label=f'Segment Prediction: ${segment_prediction:.2f}')
                    
                    axs[1, 0].axvline(x=prediction, color='#ff7f0e', linestyle='-', 
                                    linewidth=2, label=f'Final Prediction: ${prediction:.2f}')
                    
                    # Add labels and title
                    axs[1, 0].set_xlabel('Settlement Value ($)')
                    axs[1, 0].set_ylabel('Frequency')
                    axs[1, 0].set_title('Distribution of Settlement Values in Segment', fontsize=14)
                    axs[1, 0].legend()
                    
                    # Add vertical grid lines
                    axs[1, 0].grid(axis='x', linestyle='--', alpha=0.5)
                else:
                    axs[1, 0].text(0.5, 0.5, "SettlementValue not available in segment data", 
                                ha='center', va='center', fontsize=12)
                    axs[1, 0].axis('off')
            else:
                axs[1, 0].text(0.5, 0.5, "Segment data not available", 
                            ha='center', va='center', fontsize=12)
                axs[1, 0].axis('off')
        except Exception as e:
            axs[1, 0].text(0.5, 0.5, f"Error creating distribution: {str(e)}", 
                        ha='center', va='center', fontsize=10)
            axs[1, 0].axis('off')
        
        # 4. Key Feature Values in Current Record vs. Segment Average (Bottom Right)
        try:
            if segment_id in segments and segment_id in self.model.segment_models:
                segment_df = segments[segment_id]
                segment_model = self.model.segment_models[segment_id]['model']
                segment_features = self.model.segment_models[segment_id]['features']
                
                if hasattr(segment_model, 'feature_importances_'):
                    # Get top 5 features by importance
                    importances = segment_model.feature_importances_
                    indices = np.argsort(importances)[-5:]  # Top 5 features
                    top_features = [segment_features[i] for i in indices]
                    
                    # Compare record values to segment averages for these features
                    feature_comparison = []
                    feature_names = []
                    
                    for feature in top_features:
                        if feature in record.columns and feature in segment_df.columns:
                            if pd.api.types.is_numeric_dtype(segment_df[feature]):
                                # For numeric features, get the current value and segment average
                                current_value = record[feature].iloc[0]
                                segment_avg = segment_df[feature].mean()
                                feature_comparison.append((current_value, segment_avg))
                                feature_names.append(feature)
                    
                    if feature_comparison:
                        # Create a grouped bar chart
                        x = np.arange(len(feature_names))
                        width = 0.35
                        
                        current_values = [fc[0] for fc in feature_comparison]
                        segment_avgs = [fc[1] for fc in feature_comparison]
                        
                        axs[1, 1].bar(x - width/2, current_values, width, label='Current Record', color='#d62728')
                        axs[1, 1].bar(x + width/2, segment_avgs, width, label='Segment Average', color='#9467bd')
                        
                        # Set the labels and title
                        axs[1, 1].set_xlabel('Feature')
                        axs[1, 1].set_ylabel('Value')
                        axs[1, 1].set_title('Key Feature Values: Current vs. Segment Average', fontsize=14)
                        axs[1, 1].set_xticks(x)
                        axs[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
                        axs[1, 1].legend()
                        
                        # Add horizontal grid lines
                        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
                    else:
                        axs[1, 1].text(0.5, 0.5, "No comparable numeric features available", 
                                    ha='center', va='center', fontsize=12)
                        axs[1, 1].axis('off')
                else:
                    axs[1, 1].text(0.5, 0.5, "Feature importance not available", 
                                ha='center', va='center', fontsize=12)
                    axs[1, 1].axis('off')
            else:
                axs[1, 1].text(0.5, 0.5, "Segment data or model not available", 
                            ha='center', va='center', fontsize=12)
                axs[1, 1].axis('off')
        except Exception as e:
            axs[1, 1].text(0.5, 0.5, f"Error creating feature comparison: {str(e)}", 
                        ha='center', va='center', fontsize=10)
            axs[1, 1].axis('off')
        
        # Main title for the entire figure
        plt.suptitle(f'Detailed Analysis of Segment: {segment_id}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'segment_details.png'), dpi=150)
        plt.close()

    def _create_prediction_summary(self, original_record, processed_record, prediction, segment_id, segment_prediction, output_dir):
        """Create a detailed text summary of the prediction and key factors"""
        summary = {
            'Prediction': prediction,
            'Segment Used': segment_id,
            'Segment-only Prediction': segment_prediction if segment_prediction is not None else "N/A"
        }
        
        # Add key features and their values
        if segment_id in self.model.segment_models:
            segment_features = self.model.segment_models[segment_id]['features']
            
            # If we have model with feature importances
            if hasattr(self.model.segment_models[segment_id]['model'], 'feature_importances_'):
                importances = self.model.segment_models[segment_id]['model'].feature_importances_
                
                # Get top 10 most important features
                indices = np.argsort(importances)[-10:]
                top_features = [segment_features[i] for i in indices]
                
                summary['Key Features'] = {}
                for feature in top_features:
                    if feature in processed_record.columns:
                        summary['Key Features'][feature] = processed_record[feature].iloc[0]
        
        # Original input values
        summary['Original Input'] = {}
        
        # Add segmentation columns
        for col in self.model.segmentation_cols:
            if col in original_record.columns:
                summary['Original Input'][col] = original_record[col].iloc[0]
        
        # Add special damages
        special_cols = [col for col in original_record.columns if 'Special_' in col]
        if special_cols:
            summary['Special Damages'] = {}
            for col in special_cols:
                summary['Special Damages'][col] = original_record[col].iloc[0]
            
            # Calculate total special damages
            summary['Special Damages']['Total Special Damages'] = sum(original_record[col].iloc[0] for col in special_cols)
        
        # Add general damages
        general_cols = [col for col in original_record.columns if 'General_' in col]
        if general_cols:
            summary['General Damages'] = {}
            for col in general_cols:
                summary['General Damages'][col] = original_record[col].iloc[0]
            
            # Calculate total general damages
            summary['General Damages']['Total General Damages'] = sum(original_record[col].iloc[0] for col in general_cols)
        
        # Add total damages and ratio
        if 'Special Damages' in summary and 'General Damages' in summary:
            total_special = summary['Special Damages']['Total Special Damages']
            total_general = summary['General Damages']['Total General Damages']
            
            summary['Damage Summary'] = {
                'Total Damages': total_special + total_general,
                'Special/General Ratio': total_special / total_general if total_general > 0 else "N/A"
            }
        
        # Add other factors
        other_factors = {}
        for col in original_record.columns:
            if col not in self.model.segmentation_cols and 'Special_' not in col and 'General_' not in col:
                other_factors[col] = original_record[col].iloc[0]
        
        if other_factors:
            summary['Other Factors'] = other_factors
        
        # Save summary to file
        with open(os.path.join(output_dir, 'prediction_summary.txt'), 'w') as f:
            f.write("PREDICTION SUMMARY\n")
            f.write("=================\n\n")
            
            f.write(f"Final Prediction: ${summary['Prediction']:.2f}\n")
            f.write(f"Segment Used: {summary['Segment Used']}\n")
            f.write(f"Segment-only Prediction: {summary['Segment-only Prediction']}\n\n")
            
            if 'Key Features' in summary:
                f.write("KEY FEATURES (in order of importance):\n")
                for feature, value in summary['Key Features'].items():
                    if isinstance(value, float):
                        f.write(f"  {feature}: {value:.4f}\n")
                    else:
                        f.write(f"  {feature}: {value}\n")
                f.write("\n")
            
            if 'Original Input' in summary:
                f.write("SEGMENT CRITERIA:\n")
                for feature, value in summary['Original Input'].items():
                    f.write(f"  {feature}: {value}\n")
                f.write("\n")
            
            if 'Special Damages' in summary:
                f.write("SPECIAL DAMAGES:\n")
                for item, value in summary['Special Damages'].items():
                    f.write(f"  {item}: ${value:.2f}\n")
                f.write("\n")
            
            if 'General Damages' in summary:
                f.write("GENERAL DAMAGES:\n")
                for item, value in summary['General Damages'].items():
                    f.write(f"  {item}: ${value:.2f}\n")
                f.write("\n")
            
            if 'Damage Summary' in summary:
                f.write("DAMAGE SUMMARY:\n")
                for item, value in summary['Damage Summary'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {item}: ${value:.2f}\n")
                    else:
                        f.write(f"  {item}: {value}\n")
                f.write("\n")
            
            if 'Other Factors' in summary:
                f.write("OTHER FACTORS:\n")
                for item, value in summary['Other Factors'].items():
                    f.write(f"  {item}: {value}\n")


def create_synthetic_record():
    """Create a synthetic record with realistic default values for a vehicle accident case"""
    return pd.DataFrame({
        # Segmentation columns
        'Exceptional_Circumstances': ['No'],
        'AccidentType': ['Vehicle'],
        
        # Dates
        'Accident Date': ['2023-01-15'],
        'Claim Date': ['2023-02-25'],  # 40 days to claim
        
        # Special damages (economic losses)
        'Special_Medical': [8250],      # Medical expenses
        'Special_LossOfEarnings': [6300],  # Lost wages (3 weeks at average weekly wage)
        'Special_Care': [1200],         # Home care and assistance
        'Special_Other': [850],         # Travel to appointments, medication, etc.
        
        # General damages (non-economic losses)
        'General_Pain': [12500],        # Physical pain
        'General_Suffering': [9000],    # Emotional suffering
        'General_LossOfAmenity': [7500],  # Loss of enjoyment of life
        
        # Injury details
        'Whiplash': ['Yes'],
        'Minor_Psychological_Injury': ['Yes'],
        'Permanent_Injury': ['No'],
        'Recovery_Duration': [180],     # Expected recovery in days
        
        # Other potentially relevant factors
        'Age': [37],
        'Sex': ['Female'],
        'Employed': ['Yes'],
        'Dependents': [2]
    })


def load_sample_record():
    """Load a sample record from the training data"""
    try:
        df = pd.read_csv('Synthetic_Data_For_Students.csv')
        return df.iloc[[0]]  # Use the first row as sample
    except FileNotFoundError:
        print("Error: Could not find Synthetic_Data_For_Students.csv")
        print("Creating a synthetic record instead")
        return create_synthetic_record()


def collect_manual_input():
    """Collect record values through manual input"""
    print("\n=== Enter values for the record ===")
    
    record = {}
    
    # Numeric variables (with validation)
    def get_numeric_input(prompt, default):
        while True:
            try:
                value = input(f"{prompt} [default: {default}]: ") or default
                return float(value)
            except ValueError:
                print("Please enter a numeric value")
    
    # Segmentation columns
    print("\n\033[1mSegmentation Categories:\033[0m")
    record['Exceptional_Circumstances'] = input("Exceptional Circumstances (Yes/No) [default: No]: ") or "No"
    record['AccidentType'] = input("Accident Type (Vehicle/Work/Premises/Other) [default: Vehicle]: ") or "Vehicle"
    
    # Dates
    print("\n\033[1mDates:\033[0m")
    record['Accident Date'] = input("Accident Date (YYYY-MM-DD) [default: 2023-01-15]: ") or "2023-01-15"
    record['Claim Date'] = input("Claim Date (YYYY-MM-DD) [default: 2023-02-25]: ") or "2023-02-25"
    
    # Special damages
    print("\n\033[1mSpecial Damages (Economic Losses):\033[0m")
    record['Special_Medical'] = get_numeric_input("Medical Expenses ($)", 8250)
    record['Special_LossOfEarnings'] = get_numeric_input("Loss of Earnings ($)", 6300)
    record['Special_Care'] = get_numeric_input("Care Costs ($)", 1200)
    record['Special_Other'] = get_numeric_input("Other Special Damages ($)", 850)
    
    # General damages
    print("\n\033[1mGeneral Damages (Non-Economic Losses):\033[0m")
    record['General_Pain'] = get_numeric_input("Pain ($)", 12500)
    record['General_Suffering'] = get_numeric_input("Suffering ($)", 9000)
    record['General_LossOfAmenity'] = get_numeric_input("Loss of Amenity ($)", 7500)
    
    # Injury details
    print("\n\033[1mInjury Details:\033[0m")
    record['Whiplash'] = input("Whiplash (Yes/No) [default: Yes]: ") or "Yes"
    record['Minor_Psychological_Injury'] = input("Minor Psychological Injury (Yes/No) [default: Yes]: ") or "Yes"
    record['Permanent_Injury'] = input("Permanent Injury (Yes/No) [default: No]: ") or "No"
    record['Recovery_Duration'] = get_numeric_input("Expected Recovery Duration (days)", 180)
    
    # Other factors
    print("\n\033[1mOther Factors:\033[0m")
    record['Age'] = get_numeric_input("Age", 37)
    record['Sex'] = input("Sex (Male/Female) [default: Female]: ") or "Female"
    record['Employed'] = input("Employed (Yes/No) [default: Yes]: ") or "Yes"
    record['Dependents'] = get_numeric_input("Number of Dependents", 2)
    
    # Convert to DataFrame (ensuring each value is in a list to create a DataFrame row)
    return pd.DataFrame({k: [v] for k, v in record.items()})