import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings
from datetime import datetime
import json 
import traceback

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
    
    def _log(self, message):
        """Simple logging function to print messages"""
        print(message)   
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
        
        # Generate standard visualizations
        self._log("Generating standard visualizations...")
        self._visualize_segment_details(processed_record, segment_id, prediction, segment_prediction, output_dir)
        
        # Generate new advanced visualizations
        self._log("Generating advanced visualizations...")
        
        # Import json for decision path visualization
        import json
        import traceback
        
        # Feature contribution waterfall chart
        try:
            self._visualize_feature_contributions(processed_record, segment_id, prediction, output_dir)
            self._log("Created feature contribution waterfall chart")
        except Exception as e:
            self._log(f"Error creating feature contribution waterfall: {str(e)}")
        
        # Decision path visualization
        try:
            self._visualize_decision_paths(processed_record, segment_id, output_dir)
            self._log("Created decision path visualization")
        except Exception as e:
            self._log(f"Error creating decision path visualization: {str(e)}")
        
        # Similar cases comparison
        try:
            self._visualize_similar_cases(processed_record, segment_id, prediction, output_dir)
            self._log("Created similar cases comparison")
        except Exception as e:
            self._log(f"Error creating similar cases comparison: {str(e)}")
        
        # Sensitivity analysis
        try:
            self._visualize_sensitivity_analysis(processed_record, segment_id, prediction, output_dir)
            self._log("Created sensitivity analysis")
        except Exception as e:
            self._log(f"Error creating sensitivity analysis: {str(e)}")
        
        
        
        # Ensemble weights visualization
        try:
            self._visualize_ensemble_weights(processed_record, segment_id, output_dir)
            self._log("Created ensemble weights visualization")
        except Exception as e:
            self._log(f"Error creating ensemble weights visualization: {str(e)}")
        
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
    def _visualize_feature_contributions(self, record, segment_id, prediction, output_dir):
        """
        Create a waterfall chart showing how individual features contribute to the prediction
        """
        if segment_id in self.model.segment_models:
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            # Only works for tree-based models like XGBoost
            if hasattr(segment_model, 'feature_importances_'):
                # Get importance and sort features
                importances = segment_model.feature_importances_
                indices = np.argsort(importances)[-10:]  # Top 10 features
                top_features = [segment_features[i] for i in indices]
                top_importances = importances[indices]
                
                # Calculate contribution based on importance weight and feature value
                contributions = []
                feature_values = []
                
                # Baseline is the average prediction for this segment
                if segment_id in self.model.segments:
                    baseline = self.model.segments[segment_id]['SettlementValue'].mean()
                else:
                    baseline = self.model.fallback_mean if self.model.fallback_mean is not None else 0
                
                # Scale importances to match difference between prediction and baseline
                scaling_factor = (prediction - baseline) / sum(top_importances)
                
                for feature, importance in zip(top_features, top_importances):
                    if feature in record.columns:
                        value = record[feature].iloc[0]
                        if isinstance(value, (int, float)):
                            value_text = f"{value:.2f}" if isinstance(value, float) else f"{value}"
                        else:
                            value_text = str(value)
                        
                        # Calculate contribution (scaled importance)
                        contribution = importance * scaling_factor
                        
                        contributions.append(contribution)
                        feature_values.append((feature, value_text, contribution))
                
                # Sort by absolute contribution
                feature_values.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Create the waterfall chart
                plt.figure(figsize=(12, 8))
                
                # Starting point is baseline
                running_total = baseline
                x_labels = ["Baseline"]
                y_values = [running_total]
                
                # Add each feature's contribution
                for feature, value, contribution in feature_values:
                    x_labels.append(f"{feature}\n({value})")
                    running_total += contribution
                    y_values.append(running_total)
                
                # Add final prediction
                x_labels.append("Final Prediction")
                y_values.append(prediction)
                
                # Create the plot with connecting lines
                plt.plot(range(len(y_values)), y_values, 'o-', color='blue')
                
                # Add bars showing positive/negative contributions
                for i in range(1, len(feature_values) + 1):
                    feature, value, contribution = feature_values[i-1]
                    color = 'green' if contribution > 0 else 'red'
                    plt.bar(i, contribution, bottom=y_values[i] - contribution, color=color, alpha=0.4)
                
                # Set x-axis labels
                plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
                
                # Add value labels above each point
                for i, y in enumerate(y_values):
                    plt.text(i, y + max(y_values) * 0.01, f"${y:.2f}", ha='center')
                
                # Add a grid
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Set labels and title
                plt.ylabel('Prediction Value ($)')
                plt.title('Feature Contribution to Final Prediction', fontsize=16, pad=20)
                
                # Add explanation
                explanation = (
                    "This chart shows how each feature contributes to the final prediction.\n"
                    "Green bars indicate features that increase the prediction value, while red bars decrease it.\n"
                    "The baseline is the average settlement value for this segment."
                )
                plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.6))
                
                # Adjust layout
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                
                # Save figure
                plt.savefig(os.path.join(output_dir, 'feature_contributions.png'), dpi=150)
                plt.close()
    def _visualize_decision_paths(self, record, segment_id, output_dir):
        """
        Visualize the decision path through the first few trees in the XGBoost model
        """
        if segment_id in self.model.segment_models and hasattr(self.model.segment_models[segment_id]['model'], 'get_booster'):
            segment_model = self.model.segment_models[segment_id]['model']
            segment_features = self.model.segment_models[segment_id]['features']
            
            try:
                # Get the booster from the model
                booster = segment_model.get_booster()
                
                # Get the first tree (tree_id=0)
                tree_dump = booster.get_dump(dump_format='json')
                
                if len(tree_dump) > 0:
                    # We'll visualize the first 3 trees or fewer if there aren't that many
                    num_trees = min(3, len(tree_dump))
                    
                    # Create a figure with subplots
                    fig, axs = plt.subplots(1, num_trees, figsize=(18, 6))
                    if num_trees == 1:
                        axs = [axs]  # Make it iterable if there's only one subplot
                    
                    # Convert the record to X format
                    X = record[segment_features]
                    
                    for i in range(num_trees):
                        # Parse the tree JSON
                        tree_data = json.loads(tree_dump[i])
                        
                        # Define a recursive function to trace the decision path
                        def trace_path(node, record_row, level=0, x_pos=0.5, parent_x=None, parent_y=None):
                            if 'leaf' in node:
                                # This is a leaf node
                                value = node['leaf']
                                y_pos = -level
                                
                                # Draw node
                                axs[i].plot(x_pos, y_pos, 'ro', markersize=10)
                                
                                # Label with value
                                axs[i].text(x_pos, y_pos - 0.1, f"pred: {value:.4f}", ha='center', fontsize=8,
                                            bbox=dict(facecolor='lightyellow', alpha=0.8))
                                
                                # Connect to parent
                                if parent_x is not None and parent_y is not None:
                                    axs[i].plot([parent_x, x_pos], [parent_y, y_pos], 'k-')
                                
                                return (x_pos, y_pos)
                            else:
                                # This is a decision node
                                split_feature = node['split']
                                split_index = int(split_feature.replace('f', ''))
                                if split_index < len(segment_features):
                                    feature_name = segment_features[split_index]
                                else:
                                    feature_name = f"f{split_index}"
                                
                                threshold = node['split_condition']
                                feature_value = record_row[feature_name] if feature_name in record_row else None
                                
                                y_pos = -level
                                
                                # Draw node
                                axs[i].plot(x_pos, y_pos, 'bo', markersize=10)
                                
                                # Label with split condition
                                if feature_value is not None:
                                    node_text = f"{feature_name}\n< {threshold:.2f}\nvalue = {feature_value:.2f}"
                                else:
                                    node_text = f"{feature_name}\n< {threshold:.2f}"
                                
                                axs[i].text(x_pos, y_pos - 0.1, node_text, ha='center', fontsize=8,
                                            bbox=dict(facecolor='lightblue', alpha=0.8))
                                
                                # Connect to parent
                                if parent_x is not None and parent_y is not None:
                                    axs[i].plot([parent_x, x_pos], [parent_y, y_pos], 'k-')
                                
                                # Determine which path to follow based on the feature value
                                if feature_value is not None and feature_value < threshold:
                                    # Move left in the tree (feature < threshold)
                                    left_x_pos = x_pos - 0.15 * (2 ** (5 - level))
                                    child_pos = trace_path(node['children'][0], record_row, level + 1, left_x_pos, x_pos, y_pos)
                                    
                                    # Gray out the right path (not taken)
                                    right_x_pos = x_pos + 0.15 * (2 ** (5 - level))
                                    axs[i].plot([x_pos, right_x_pos], [y_pos, -(level+1)], 'k-', alpha=0.2)
                                    
                                    return child_pos
                                else:
                                    # Move right in the tree (feature >= threshold)
                                    right_x_pos = x_pos + 0.15 * (2 ** (5 - level))
                                    child_pos = trace_path(node['children'][1], record_row, level + 1, right_x_pos, x_pos, y_pos)
                                    
                                    # Gray out the left path (not taken)
                                    left_x_pos = x_pos - 0.15 * (2 ** (5 - level))
                                    axs[i].plot([x_pos, left_x_pos], [y_pos, -(level+1)], 'k-', alpha=0.2)
                                    
                                    return child_pos
                        
                        # Start tracing the path from the root
                        trace_path(tree_data, X.iloc[0])
                        
                        # Remove axes
                        axs[i].set_xticks([])
                        axs[i].set_yticks([])
                        
                        # Add tree title
                        axs[i].set_title(f"Tree #{i}", fontsize=14)
                    
                    # Add overall title
                    plt.suptitle(f"Decision Paths for First {num_trees} Trees", fontsize=16)
                    
                    # Add explanation
                    explanation = (
                        "This visualization shows the decision path through the model's trees for this record.\n"
                        "Blue nodes are decision points, red nodes are leaf predictions.\n"
                        "The path taken for this record is highlighted, with grayed-out paths not taken.\n"
                        "Each tree contributes to the ensemble's final prediction."
                    )
                    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=10, 
                                bbox=dict(facecolor='#f0f0f0', alpha=0.6))
                    
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    plt.savefig(os.path.join(output_dir, 'decision_paths.png'), dpi=150)
                    plt.close()
                        
            except Exception as e:
                self._log(f"Error creating decision path visualization: {str(e)}")
                traceback.print_exc()
    def _visualize_similar_cases(self, record, segment_id, prediction, output_dir):
        """
        Create a visualization comparing the current case to similar cases in the same segment
        """
        if segment_id in self.model.segments:
            segment_df = self.model.segments[segment_id]
            
            # We need to compute similarity between the record and other cases in the segment
            if len(segment_df) > 0:
                # Select only numeric features for similarity calculation
                numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
                numeric_cols = [col for col in numeric_cols if col != 'SettlementValue' and col in record.columns]
                
                if len(numeric_cols) > 0:
                    # Create a copy of the record with only numeric columns
                    record_numeric = record[numeric_cols].iloc[0].copy()
                    
                    # Standardize to calculate distances properly
                    segment_numeric = segment_df[numeric_cols].copy()
                    
                    # Calculate means and standard deviations for normalization
                    means = segment_numeric.mean()
                    stds = segment_numeric.std().replace(0, 1)  # Avoid division by zero
                    
                    # Normalize the segment data
                    segment_numeric_norm = (segment_numeric - means) / stds
                    
                    # Normalize the record
                    record_numeric_norm = (record_numeric - means) / stds
                    
                    # Calculate Euclidean distance between record and each case in segment
                    distances = []
                    for i, row in segment_numeric_norm.iterrows():
                        dist = np.sqrt(np.sum((row - record_numeric_norm) ** 2))
                        distances.append((i, dist))
                    
                    # Sort by distance and get the 5 most similar cases
                    distances.sort(key=lambda x: x[1])
                    similar_indices = [d[0] for d in distances[:5]]
                    similar_cases = segment_df.loc[similar_indices]
                    
                    # Create the visualization
                    fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 2]})
                    
                    # 1. Bar chart comparison of key damage values and settlement
                    damage_cols = [col for col in segment_df.columns if 'Special_' in col or 'General_' in col or 'Total_' in col]
                    damage_cols = [col for col in damage_cols if col in record.columns]
                    
                    if len(damage_cols) > 0 and 'SettlementValue' in segment_df.columns:
                        # Add settlement value to the list
                        all_cols = damage_cols + ['SettlementValue']
                        
                        # Create data for the bar chart
                        bar_data = []
                        bar_labels = []
                        
                        # Current case
                        current_case_values = [record[col].iloc[0] if col != 'SettlementValue' else prediction for col in all_cols]
                        bar_data.append(current_case_values)
                        bar_labels.append('Current Case')
                        
                        # Similar cases
                        for i, (idx, similar_case) in enumerate(similar_cases.iterrows()):
                            similar_case_values = [similar_case[col] for col in all_cols]
                            bar_data.append(similar_case_values)
                            bar_labels.append(f'Similar Case {i+1}')
                        
                        # Create grouped bar chart
                        x = np.arange(len(all_cols))
                        width = 0.15
                        
                        for i, (case_data, case_label) in enumerate(zip(bar_data, bar_labels)):
                            offset = width * (i - len(bar_data)/2 + 0.5)
                            bars = axs[0].bar(x + offset, case_data, width, label=case_label, 
                                            alpha=0.7 if case_label == 'Current Case' else 0.5)
                            
                            # Add value labels on top of bars for current case only
                            if case_label == 'Current Case':
                                for j, v in enumerate(case_data):
                                    axs[0].text(x[j] + offset, v + max(case_data) * 0.02, 
                                            f'${v:.0f}', ha='center', fontsize=8)
                        
                        # Add labels and legend
                        axs[0].set_xticks(x)
                        axs[0].set_xticklabels([col.replace('_', ' ') for col in all_cols], rotation=45, ha='right')
                        axs[0].legend(loc='upper left')
                        axs[0].set_title('Damage Amounts Comparison with Similar Cases', fontsize=14)
                        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
                        
                    # 2. Feature comparison table
                    # Use a table to show feature values for current and similar cases
                    key_features = []
                    
                    # Get segment model features if available
                    if segment_id in self.model.segment_models:
                        model_info = self.model.segment_models[segment_id]
                        if hasattr(model_info['model'], 'feature_importances_'):
                            importances = model_info['model'].feature_importances_
                            segment_features = model_info['features']
                            # Get top 10 important features
                            indices = np.argsort(importances)[-10:]
                            key_features = [segment_features[i] for i in indices]
                    
                    # If no model features, use damage columns
                    if not key_features:
                        key_features = damage_cols
                    
                    # Filter to features present in record
                    key_features = [f for f in key_features if f in record.columns]
                    
                    if key_features:
                        # Create a table to display the feature values
                        axs[1].axis('off')
                        
                        # Prepare data for the table
                        feature_names = [f.replace('_', ' ') for f in key_features]
                        
                        # Current case values
                        current_values = []
                        for feature in key_features:
                            value = record[feature].iloc[0]
                            if isinstance(value, (int, float)):
                                current_values.append(f"{value:.2f}" if isinstance(value, float) else f"{value}")
                            else:
                                current_values.append(str(value))
                        
                        # Similar case values
                        similar_values = []
                        for _, similar_case in similar_cases.iterrows():
                            case_values = []
                            for feature in key_features:
                                value = similar_case[feature]
                                if isinstance(value, (int, float)):
                                    case_values.append(f"{value:.2f}" if isinstance(value, float) else f"{value}")
                                else:
                                    case_values.append(str(value))
                            similar_values.append(case_values)
                        
                        # Add settlement values row
                        feature_names.append('Settlement Value')
                        current_values.append(f"${prediction:.2f}")
                        for i, (idx, similar_case) in enumerate(similar_cases.iterrows()):
                            if 'SettlementValue' in similar_case:
                                similar_values[i].append(f"${similar_case['SettlementValue']:.2f}")
                            else:
                                similar_values[i].append("N/A")
                        
                        # Create table data
                        table_data = [feature_names]
                        table_data.append(current_values)
                        for sv in similar_values:
                            table_data.append(sv)
                        
                        # Transpose the table data for better display
                        table_data = list(map(list, zip(*table_data)))
                        
                        # Create header row
                        header = ['Feature', 'Current Case'] + [f'Similar Case {i+1}' for i in range(len(similar_cases))]
                        
                        # Create the table
                        table = axs[1].table(
                            cellText=table_data,
                            colLabels=header,
                            loc='center',
                            cellLoc='center'
                        )
                        
                        # Style the table
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        table.scale(1, 1.5)
                        
                        # Highlight the current case column
                        for i in range(len(table_data) + 1):
                            cell = table[i, 1]
                            cell.set_facecolor('#FFEB9C')
                        
                        # Highlight the settlement value row
                        for j in range(len(header)):
                            cell = table[-1, j]
                            cell.set_facecolor('#E6F4EA')
                        
                        axs[1].set_title('Feature Comparison with Similar Cases', fontsize=14)
                    
                    # Add overall title
                    plt.suptitle(f'Comparison with Similar Cases in Segment: {segment_id}', fontsize=16, fontweight='bold')
                    
                    # Add explanation
                    explanation = (
                        "This visualization compares your case to the most similar cases in the same segment.\n"
                        "The top chart shows damage amounts and settlement values.\n"
                        "The table below shows values for key features that influence the prediction."
                    )
                    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                            bbox=dict(facecolor='#f0f0f0', alpha=0.6))
                    
                    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
                    plt.savefig(os.path.join(output_dir, 'similar_cases.png'), dpi=150)
                    plt.close()
    def _visualize_sensitivity_analysis(self, record, segment_id, prediction, output_dir):
        """
        Create a visualization showing how changing key input values affects the prediction
        """
        if segment_id in self.model.segment_models:
            model_info = self.model.segment_models[segment_id]
            segment_model = model_info['model']
            segment_features = model_info['features']
            
            # Only make a sensitivity analysis for models with feature importances
            if hasattr(segment_model, 'feature_importances_'):
                importances = segment_model.feature_importances_
                
                # Get the top 5 most important numeric features
                indices = np.argsort(importances)[-10:]
                top_features = [segment_features[i] for i in indices]
                
                # Filter to only include numeric features
                numeric_features = []
                for feature in top_features:
                    if feature in record.columns:
                        if pd.api.types.is_numeric_dtype(record[feature]):
                            numeric_features.append(feature)
                
                # Limit to top 5 numeric features
                numeric_features = numeric_features[:5]
                
                if numeric_features:
                    # Create the figure for sensitivity analysis
                    fig, axs = plt.subplots(len(numeric_features), 1, figsize=(12, 3 * len(numeric_features)))
                    if len(numeric_features) == 1:
                        axs = [axs]  # Ensure axs is iterable
                    
                    # For each feature, vary it and see how the prediction changes
                    for i, feature in enumerate(numeric_features):
                        current_value = record[feature].iloc[0]
                        
                        # Create a range of values to test
                        # For each feature, create 11 values: current value plus 5 less and 5 more
                        values_range = np.linspace(current_value * 0.5, current_value * 1.5, 11)
                        
                        # Make predictions for each value
                        predictions = []
                        for value in values_range:
                            # Create a copy of the record and change just this feature
                            test_record = record.copy()
                            test_record[feature] = value
                            
                            # Preprocess the test record
                            processed_record = self._preprocess_record(test_record)
                            
                            # Get prediction from the segment model
                            pred = self._get_segment_prediction(processed_record, segment_id)
                            
                            if pred is not None:
                                predictions.append(pred)
                            else:
                                # If prediction fails, use the original prediction
                                predictions.append(prediction)
                        
                        # Plot the sensitivity curve
                        axs[i].plot(values_range, predictions, '-o', color='#2196F3')
                        
                        # Highlight the current value
                        current_idx = np.abs(values_range - current_value).argmin()
                        axs[i].plot(current_value, predictions[current_idx], 'ro', markersize=10, 
                                label=f'Current: {current_value:.2f}')
                        
                        # Draw a horizontal line for the original prediction
                        axs[i].axhline(y=prediction, color='green', linestyle='--', 
                                    label=f'Original Prediction: ${prediction:.2f}')
                        
                        # Add value annotations
                        for j, (x, y) in enumerate(zip(values_range, predictions)):
                            if j % 2 == 0:  # Label every other point for clarity
                                axs[i].annotate(f'${y:.2f}', (x, y), textcoords="offset points", 
                                            xytext=(0,10), ha='center')
                        
                        # Set labels and title
                        axs[i].set_xlabel(f'{feature} Value')
                        axs[i].set_ylabel('Prediction ($)')
                        axs[i].set_title(f'Sensitivity to {feature}', fontsize=14)
                        
                        # Add grid
                        axs[i].grid(True, linestyle='--', alpha=0.7)
                        
                        # Add legend
                        axs[i].legend(loc='upper left')
                    
                    # Overall title
                    plt.suptitle('Sensitivity Analysis: How Feature Changes Affect Prediction', fontsize=16, fontweight='bold')
                    
                    # Add explanation
                    explanation = (
                        "This analysis shows how changing each feature affects the prediction.\n"
                        "The current value is marked with a red dot, and the horizontal green line indicates the original prediction.\n"
                        "The steeper the curve, the more sensitive the prediction is to changes in that feature."
                    )
                    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                            bbox=dict(facecolor='#f0f0f0', alpha=0.6))
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=150)
                    plt.close()
    
    def _visualize_ensemble_weights(self, record, segment_id, output_dir):
        """
        Visualize how the ensemble model weights the different segment predictions
        """
        # Create meta features for the record
        meta_features = self.model._create_meta_features(record)
        
        # Get segment predictions
        segment_predictions = {}
        for col in meta_features.columns:
            if col.startswith('pred_'):
                segment_name = col.replace('pred_', '')
                segment_predictions[segment_name] = meta_features[col].iloc[0]
        
        # Filter out None or NaN values
        segment_predictions = {k: v for k, v in segment_predictions.items() 
                            if v is not None and not pd.isna(v)}
        
        # If we have the meta-model's coefficients, try to extract them
        meta_model_coefficients = None
        meta_feature_names = None
        
        try:
            if hasattr(self.model.meta_model, 'final_estimator_') and hasattr(self.model.meta_model.final_estimator_, 'coef_'):
                # Get meta-model's coefficients (works for linear meta-models like ElasticNet)
                meta_model_coefficients = self.model.meta_model.final_estimator_.coef_
                meta_feature_names = self.model.meta_features
                
                # Create visualization for the ensemble weights
                plt.figure(figsize=(12, 8))
                
                # Prepare the data
                feature_weights = []
                feature_labels = []
                
                for i, feature in enumerate(meta_feature_names):
                    if feature.startswith('pred_'):
                        weight = meta_model_coefficients[i]
                        segment_name = feature.replace('pred_', '')
                        feature_weights.append(weight)
                        feature_labels.append(segment_name)
                
                # Only proceed if we have segment weights
                if feature_weights:
                    # Sort by absolute weight
                    sorted_indices = np.argsort(np.abs(feature_weights))[::-1]
                    feature_weights = [feature_weights[i] for i in sorted_indices]
                    feature_labels = [feature_labels[i] for i in sorted_indices]
                    
                    # Limit to top 10 for readability
                    feature_weights = feature_weights[:10]
                    feature_labels = feature_labels[:10]
                    
                    # Create colors based on weight sign
                    colors = ['green' if w > 0 else 'red' for w in feature_weights]
                    
                    # Create horizontal bar chart
                    y_pos = np.arange(len(feature_labels))
                    plt.barh(y_pos, feature_weights, align='center', color=colors, alpha=0.7)
                    
                    # Add a vertical line at 0
                    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    
                    # Highlight the segment used for this record
                    for i, label in enumerate(feature_labels):
                        if label == segment_id:
                            plt.text(-max(abs(np.array(feature_weights))) * 0.15, i, '← Current segment', 
                                    ha='right', va='center', fontsize=10, fontweight='bold',
                                    bbox=dict(facecolor='yellow', alpha=0.4, boxstyle='round,pad=0.3'))
                    
                    # Add value labels
                    for i, weight in enumerate(feature_weights):
                        plt.text(weight + (0.01 if weight >= 0 else -0.01) * max(abs(np.array(feature_weights))), 
                                i, f"{weight:.3f}", ha='left' if weight >= 0 else 'right', va='center')
                    
                    # Set labels and title
                    plt.yticks(y_pos, feature_labels)
                    plt.xlabel('Weight in Ensemble Meta-Model')
                    plt.title('How the Meta-Model Weights Different Segment Predictions', fontsize=16)
                    
                    # Add grid
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Add explanation
                    explanation = (
                        "This chart shows the weights the ensemble meta-model assigns to different segment predictions.\n"
                        "Positive weights (green) increase the final prediction, while negative weights (red) decrease it.\n"
                        "The segment used for this record is highlighted, showing its importance in the ensemble."
                    )
                    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                            bbox=dict(facecolor='#f0f0f0', alpha=0.6))
                    
                    # Adjust layout
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    
                    # Save figure
                    plt.savefig(os.path.join(output_dir, 'ensemble_weights.png'), dpi=150)
                    plt.close()
        except Exception as e:
            self._log(f"Error creating ensemble weights visualization: {str(e)}")
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
        return df.iloc[[30]]  # Use the first row as sample
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

