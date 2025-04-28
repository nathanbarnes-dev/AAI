import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Add necessary paths to import the UnifiedSegmentedModel class
current_dir = Path(__file__).parent
model_paths = list(current_dir.glob("**/model.py"))
if model_paths:
    sys.path.append(str(model_paths[0].parent))
else:
    sys.path.append('./9 final model')  # Assume model.py is in this directory

try:
    from model import UnifiedSegmentedModel
    print("Successfully imported UnifiedSegmentedModel class")
except ImportError as e:
    print(f"Error importing UnifiedSegmentedModel: {e}")
    print("Please ensure model.py is in the current directory or in '9 final model' subdirectory")
    sys.exit(1)

# Create output directory for visualizations
os.makedirs('model_visualizations', exist_ok=True)

def load_model_and_data():
    """
    Load the trained model and data
    
    Returns:
        tuple: (model, full_data)
    """
    # Try to find the model file
    model_path = 'unified_segmented_model.pkl'
    if not os.path.exists(model_path):
        # Look for the model in different locations
        for potential_path in [
            './9 final model/unified_segmented_model.pkl',
            './unified_models/unified_segmented_model.pkl',
            './output/unified_segmented_model.pkl'
        ]:
            if os.path.exists(potential_path):
                model_path = potential_path
                break
    
    try:
        model = UnifiedSegmentedModel.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Try to find the data file
    data_path = 'Synthetic_Data_For_Students.csv'
    if not os.path.exists(data_path):
        # Look for the data file in different locations
        for potential_path in [
            './Synthetic_Data_For_Students.csv', 
            '../Synthetic_Data_For_Students.csv'
        ]:
            if os.path.exists(potential_path):
                data_path = potential_path
                break
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully from {data_path}")
        return model, df
    except Exception as e:
        print(f"Error loading data: {e}")
        return model, None

def visualize_model_architecture(model):
    """
    Create a visualization showing the model's architecture
    
    Args:
        model: Trained UnifiedSegmentedModel
    """
    print("\nCreating model architecture visualization...")
    
    # Get segment information
    segments = list(model.segment_models.keys())
    num_segments = len(segments)
    
    # Get key stats for each segment
    segment_stats = []
    for segment_name, model_info in model.segment_models.items():
        segment_stats.append({
            'name': segment_name,
            'features': len(model_info['features']),
            'r2': model_info['metrics']['r2'],
            'size': model_info.get('size', 0)
        })
    
    # Sort by size
    segment_stats = sorted(segment_stats, key=lambda x: x['size'], reverse=True)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot segment sizes
    plt.subplot(2, 2, 1)
    bars = plt.bar([s['name'][:15] + '...' if len(s['name']) > 15 else s['name'] for s in segment_stats], 
            [s['size'] for s in segment_stats])
    plt.title('Segment Sizes')
    plt.xticks(rotation=90)
    plt.ylabel('Number of Records')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot segment feature counts
    plt.subplot(2, 2, 2)
    plt.bar([s['name'][:15] + '...' if len(s['name']) > 15 else s['name'] for s in segment_stats], 
            [s['features'] for s in segment_stats])
    plt.title('Features Used by Each Segment')
    plt.xticks(rotation=90)
    plt.ylabel('Number of Features')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot segment R² scores
    plt.subplot(2, 2, 3)
    bars = plt.bar([s['name'][:15] + '...' if len(s['name']) > 15 else s['name'] for s in segment_stats], 
            [s['r2'] for s in segment_stats])
    plt.title('R² Scores for Each Segment Model')
    plt.xticks(rotation=90)
    plt.ylabel('R² Score')
    plt.ylim(0, 1)  # R² is typically between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create diagram of the unified model architecture
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create a simple diagram showing the architecture
    text = "UnifiedSegmentedModel Architecture:\n\n"
    text += f"Input Data → Preprocessing → Feature Engineering\n\n"
    text += f"↓\n\n"
    text += f"Segment Assignment ({len(segments)} segments)\n\n"
    text += f"↓\n\n"
    text += f"Segment-Specific Models\n"
    text += f"(XGBoost with custom features for each segment)\n\n"
    text += f"↓\n\n"
    text += f"Meta-Learner\n"
    text += f"(Combines segment predictions)\n\n"
    text += f"↓\n\n"
    text += f"Final Prediction"
    
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('model_visualizations/model_architecture.png', dpi=300)
    plt.close()
    
    print("Model architecture visualization saved.")

def visualize_segment_distribution(model, data):
    """
    Visualize how data is distributed across segments
    
    Args:
        model: Trained UnifiedSegmentedModel
        data: Full dataset
    """
    print("\nVisualizing segment distribution...")
    
    # Preprocess data
    preprocessed_data = model._preprocess_data(data)
    
    # Engineer features
    engineered_data = model._engineer_features(preprocessed_data)
    
    # Assign segments
    engineered_data['segment'] = engineered_data.apply(model._get_segment_for_record, axis=1)
    
    # Count segments
    segment_counts = engineered_data['segment'].value_counts()
    
    # Calculate percentage
    segment_percentages = segment_counts / segment_counts.sum() * 100
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot segment counts
    plt.subplot(2, 1, 1)
    bars = plt.bar(segment_counts.index, segment_counts.values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    plt.title('Distribution of Records Across Segments')
    plt.xlabel('Segment')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot segment percentages
    plt.subplot(2, 1, 2)
    plt.pie(segment_percentages, labels=segment_percentages.index, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 8})
    plt.title('Percentage of Records in Each Segment')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('model_visualizations/segment_distribution.png', dpi=300)
    plt.close()
    
    # Check if there's a dominant segment
    max_segment = segment_counts.idxmax()
    max_percentage = segment_percentages.max()
    
    if max_percentage > 50:
        print(f"Warning: Dominant segment '{max_segment}' contains {max_percentage:.1f}% of all records.")
    
    print("Segment distribution visualization saved.")
    
    # Return engineered data with segments
    return engineered_data

def visualize_feature_importance_by_segment(model):
    """
    Create heatmap showing which features are important across segments
    
    Args:
        model: Trained UnifiedSegmentedModel
    """
    print("\nVisualizing feature importance by segment...")
    
    # Collect feature importances for each segment
    all_features = set()
    segment_importances = {}
    
    for segment_name, model_info in model.segment_models.items():
        segment_model = model_info['model']
        segment_features = model_info['features']
        
        # Get feature importances if available
        try:
            if hasattr(segment_model, 'feature_importances_'):
                importances = dict(zip(segment_features, segment_model.feature_importances_))
                segment_importances[segment_name] = importances
                all_features.update(segment_features)
            elif hasattr(segment_model, 'get_booster'):
                # XGBoost model
                importance_scores = segment_model.get_booster().get_score(importance_type='gain')
                # Map feature indices to names (this might need adjustment)
                importances = {}
                for i, feature in enumerate(segment_features):
                    # XGBoost uses feature indices in importance dict
                    feature_key = f"f{i}"
                    if feature_key in importance_scores:
                        importances[feature] = importance_scores[feature_key]
                    else:
                        importances[feature] = 0
                segment_importances[segment_name] = importances
                all_features.update(segment_features)
        except Exception as e:
            print(f"Could not get importance for segment {segment_name}: {str(e)}")
    
    if not segment_importances:
        print("Could not extract feature importances from segment models.")
        return
    
    # Create a DataFrame for heatmap
    all_features = sorted(list(all_features))
    all_segments = list(segment_importances.keys())
    
    importance_matrix = np.zeros((len(all_segments), len(all_features)))
    
    for i, segment in enumerate(all_segments):
        for j, feature in enumerate(all_features):
            importance_matrix[i, j] = segment_importances[segment].get(feature, 0)
    
    # Normalize by row (segment)
    for i in range(len(all_segments)):
        row_max = importance_matrix[i].max()
        if row_max > 0:
            importance_matrix[i] = importance_matrix[i] / row_max
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(importance_matrix, cmap='viridis', yticklabels=all_segments, xticklabels=all_features)
    plt.title('Feature Importance Across Segments (Normalized)')
    plt.xlabel('Features')
    plt.ylabel('Segments')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('model_visualizations/feature_importance_heatmap.png', dpi=300)
    plt.close()
    
    # Create a summary of top features per segment
    top_features = {}
    for segment, importances in segment_importances.items():
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features[segment] = [f[0] for f in sorted_importances[:5]]  # Top 5 features
    
    # Create a bar chart showing the most common important features
    feature_frequency = {}
    for segment_top_features in top_features.values():
        for feature in segment_top_features:
            feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
    
    feature_frequency = dict(sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(list(feature_frequency.keys()), list(feature_frequency.values()))
    plt.title('Most Common Important Features Across Segments')
    plt.xlabel('Feature')
    plt.ylabel('Number of Segments')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_visualizations/common_important_features.png', dpi=300)
    plt.close()
    
    print("Feature importance visualizations saved.")

def visualize_error_analysis(model, data):
    """
    Visualize model errors by segment
    
    Args:
        model: Trained UnifiedSegmentedModel
        data: Full dataset with assigned segments
    """
    print("\nPerforming error analysis by segment...")
    
    # Only use data with actual settlement values
    data = data.dropna(subset=['SettlementValue']).copy()
    
    # Split into train and test for consistent evaluation
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Get predictions for test data
    try:
        test_predictions = model.predict(test_data)
        
        # If predictions successful, add to dataframe
        if len(test_predictions) == len(test_data):
            test_data['prediction'] = test_predictions
            test_data['error'] = test_data['prediction'] - test_data['SettlementValue']
            test_data['abs_error'] = np.abs(test_data['error'])
            test_data['percent_error'] = 100 * np.abs(test_data['error']) / test_data['SettlementValue']
            
            # Assign segments
            test_data['segment'] = test_data.apply(model._get_segment_for_record, axis=1)
            
            # Calculate error metrics by segment
            segment_metrics = {}
            for segment in test_data['segment'].unique():
                segment_data = test_data[test_data['segment'] == segment]
                if len(segment_data) > 0:
                    segment_metrics[segment] = {
                        'count': len(segment_data),
                        'mae': mean_absolute_error(segment_data['SettlementValue'], segment_data['prediction']),
                        'mse': mean_squared_error(segment_data['SettlementValue'], segment_data['prediction']),
                        'rmse': np.sqrt(mean_squared_error(segment_data['SettlementValue'], segment_data['prediction'])),
                        'r2': r2_score(segment_data['SettlementValue'], segment_data['prediction']),
                        'mean_percent_error': segment_data['percent_error'].mean()
                    }
            
            # Convert to DataFrame for easier plotting
            metrics_df = pd.DataFrame(segment_metrics).T
            metrics_df = metrics_df.sort_values('count', ascending=False)
            
            # Plot error metrics by segment
            plt.figure(figsize=(14, 10))
            
            # Plot MAE by segment
            plt.subplot(2, 2, 1)
            bars = plt.bar(metrics_df.index, metrics_df['mae'])
            plt.title('Mean Absolute Error by Segment')
            plt.xlabel('Segment')
            plt.ylabel('MAE ($)')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot RMSE by segment
            plt.subplot(2, 2, 2)
            plt.bar(metrics_df.index, metrics_df['rmse'])
            plt.title('Root Mean Squared Error by Segment')
            plt.xlabel('Segment')
            plt.ylabel('RMSE ($)')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot R² by segment
            plt.subplot(2, 2, 3)
            plt.bar(metrics_df.index, metrics_df['r2'])
            plt.title('R² Score by Segment')
            plt.xlabel('Segment')
            plt.ylabel('R² Score')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot mean percent error by segment
            plt.subplot(2, 2, 4)
            plt.bar(metrics_df.index, metrics_df['mean_percent_error'])
            plt.title('Mean Percentage Error by Segment')
            plt.xlabel('Segment')
            plt.ylabel('Mean % Error')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('model_visualizations/error_analysis_by_segment.png', dpi=300)
            plt.close()
            
            # Create error distribution plot
            plt.figure(figsize=(14, 8))
            
            # Plot histogram of errors
            plt.subplot(2, 2, 1)
            plt.hist(test_data['error'], bins=30)
            plt.title('Distribution of Errors')
            plt.xlabel('Error ($)')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            
            # Plot absolute errors against actual values
            plt.subplot(2, 2, 2)
            plt.scatter(test_data['SettlementValue'], test_data['abs_error'], alpha=0.5)
            plt.title('Absolute Error vs. Actual Value')
            plt.xlabel('Actual Settlement Value ($)')
            plt.ylabel('Absolute Error ($)')
            plt.grid(alpha=0.3)
            
            # Plot predicted vs actual
            plt.subplot(2, 2, 3)
            plt.scatter(test_data['SettlementValue'], test_data['prediction'], alpha=0.5)
            plt.plot([test_data['SettlementValue'].min(), test_data['SettlementValue'].max()], 
                     [test_data['SettlementValue'].min(), test_data['SettlementValue'].max()], 
                     'r--')
            plt.title('Predicted vs. Actual Values')
            plt.xlabel('Actual Settlement Value ($)')
            plt.ylabel('Predicted Settlement Value ($)')
            plt.grid(alpha=0.3)
            
            # Plot box plot of errors by segment (for top 5 segments by count)
            plt.subplot(2, 2, 4)
            top_segments = metrics_df.index[:5]  # Top 5 segments by count
            segment_errors = [test_data[test_data['segment'] == segment]['error'] for segment in top_segments]
            plt.boxplot(segment_errors, labels=top_segments)
            plt.title('Error Distribution by Top 5 Segments')
            plt.xlabel('Segment')
            plt.ylabel('Error ($)')
            plt.xticks(rotation=90)
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('model_visualizations/error_distribution.png', dpi=300)
            plt.close()
            
            print("Error analysis visualizations saved.")
            
            return test_data, metrics_df
        else:
            print(f"Prediction length mismatch: got {len(test_predictions)}, expected {len(test_data)}")
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
    
    return None, None

def visualize_meta_model_contribution(model, test_data_with_segments):
    """
    Visualize how the meta-model contributes to final predictions
    
    Args:
        model: Trained UnifiedSegmentedModel
        test_data_with_segments: Test data with segment assignments and predictions
    """
    print("\nVisualizing meta-model contribution...")
    
    if test_data_with_segments is None or 'prediction' not in test_data_with_segments.columns:
        print("Cannot analyze meta-model contribution without test predictions")
        return
    
    # Sample a few records for detailed analysis
    sampled_data = test_data_with_segments.sample(min(5, len(test_data_with_segments)), random_state=42)
    
    # Create a plot showing segment and meta-model predictions
    plt.figure(figsize=(15, 10))
    
    for i, (idx, record) in enumerate(sampled_data.iterrows(), 1):
        plt.subplot(len(sampled_data), 1, i)
        
        # Get the segment for this record
        segment = record['segment']
        
        # Get meta features
        try:
            meta_features = model._create_meta_features(pd.DataFrame([record]))
            
            # Try to add segment-specific predictions
            segment_predictions = {}
            for segment_name, model_info in model.segment_models.items():
                segment_model = model_info['model']
                segment_features = model_info['features']
                
                # Check if we have all required features
                missing_features = [f for f in segment_features if f not in record.index]
                if not missing_features:
                    try:
                        X_segment = pd.DataFrame([record[segment_features]])
                        pred = segment_model.predict(X_segment)[0]
                        segment_predictions[segment_name] = pred
                    except:
                        pass
            
            # Sort segments by prediction value
            sorted_segments = sorted(segment_predictions.items(), key=lambda x: x[1])
            
            # Plot segment predictions
            segment_names = [s[0] for s in sorted_segments]
            preds = [s[1] for s in sorted_segments]
            
            bars = plt.barh(segment_names, preds, alpha=0.7)
            
            # Highlight the assigned segment
            for j, bar in enumerate(bars):
                if segment_names[j] == segment:
                    bar.set_color('red')
                    bar.set_alpha(1.0)
            
            # Add the final prediction
            plt.axvline(x=record['prediction'], color='green', linestyle='--', 
                        linewidth=2, label='Final Prediction')
            
            # Add the actual value
            plt.axvline(x=record['SettlementValue'], color='blue', linestyle='-', 
                        linewidth=2, label='Actual Value')
            
            plt.title(f"Record {idx}: Segment = {segment}")
            plt.xlabel('Settlement Value Prediction ($)')
            
            # Only add legend to the first subplot
            if i == 1:
                plt.legend()
        
        except Exception as e:
            plt.text(0.5, 0.5, f"Error analyzing record: {str(e)}", 
                     ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('model_visualizations/meta_model_contribution.png', dpi=300)
    plt.close()
    
    print("Meta-model contribution visualization saved.")

def main():
    """Main function"""
    print("=== Generating Enhanced Model Visualizations ===")
    
    # Load model and data
    model, data = load_model_and_data()
    if model is None or data is None:
        print("Failed to load model or data. Exiting.")
        return
    
    # Visualize model architecture
    visualize_model_architecture(model)
    
    # Visualize segment distribution
    data_with_segments = visualize_segment_distribution(model, data)
    
    # Visualize feature importance by segment
    visualize_feature_importance_by_segment(model)
    
    # Visualize error analysis
    test_data_with_errors, segment_metrics = visualize_error_analysis(model, data)
    
    # Visualize meta-model contribution
    if test_data_with_errors is not None:
        visualize_meta_model_contribution(model, test_data_with_errors)
    
    print("\n=== Model Visualization Complete ===")
    print(f"Visualizations saved to 'model_visualizations/' directory")

if __name__ == "__main__":
    main()