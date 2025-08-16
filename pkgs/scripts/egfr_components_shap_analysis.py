import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import dill
import torch
import joblib
from lifelines import CoxPHFitter
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Import project modules
from pkgs.commons import (
    generate_data_path_latest_rep, 
    egfr_components_cox_model_path,
    egfr_components_train_data_path,
    egfr_components_test_data_path
)
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.experiments.utils import get_tv_rnn_model_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class EGFRComponentsSHAPAnalysis:
    """
    SHAP Analysis for eGFR Components Experiment
    Features: age, gender, serum_creatinine
    """
    
    def __init__(self):
        self.features = ['age', 'gender', 'serum_creatinine']
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.shap_values = {}
        self.explainers = {}
        
    def load_data(self):
        """Load training and test data for eGFR components experiment"""
        print("Loading eGFR components experiment data...")
        
        self.train_data = pd.read_csv(egfr_components_train_data_path)
        self.test_data = pd.read_csv(egfr_components_test_data_path)
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Features: {self.features}")
        
        return self.train_data, self.test_data
    
    def load_cox_model(self):
        """Load Cox Proportional Hazards model"""
        try:
            if os.path.exists(egfr_components_cox_model_path):
                with open(egfr_components_cox_model_path, 'rb') as f:
                    cox_model = dill.load(f)
                self.models['cox'] = cox_model
                print("✓ Cox model loaded successfully")
                return True
            else:
                print("✗ Cox model file not found")
                return False
        except Exception as e:
            print(f"✗ Error loading Cox model: {e}")
            return False
    
    def load_pytorch_model(self, model_path, model_name):
        """Load PyTorch models (RNN, DDH, Hazard Transformer, etc.)"""
        try:
            if os.path.exists(model_path):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.load(model_path, map_location=device, weights_only=False)
                model.eval()
                self.models[model_name] = model
                print(f"✓ {model_name} model loaded successfully")
                return True
            else:
                print(f"✗ {model_name} model file not found")
                return False
        except Exception as e:
            print(f"✗ Error loading {model_name} model: {e}")
            return False
    
    def create_surrogate_model(self, model_name):
        """
        Create a surrogate model for SHAP analysis when the original model is too complex
        """
        print(f"Creating surrogate model for {model_name}...")
        
        X_train = self.train_data[self.features].values
        X_test = self.test_data[self.features].values
        
        # Get risk scores from the original model
        risk_scores = self.get_risk_scores(model_name, X_test)
        if risk_scores is None:
            return None
            
        # Train a Random Forest to mimic the original model's behavior
        rf_surrogate = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Use both train and test data to train surrogate
        X_combined = np.vstack([X_train, X_test])
        train_scores = self.get_risk_scores(model_name, X_train)
        y_combined = np.concatenate([train_scores, risk_scores])
        
        rf_surrogate.fit(X_combined, y_combined)
        
        # Store surrogate model
        surrogate_name = f'{model_name}_surrogate'
        self.models[surrogate_name] = rf_surrogate
        
        print(f"✓ Surrogate model for {model_name} created successfully")
        return rf_surrogate
    
    def get_risk_scores(self, model_name, X):
        """Get risk scores from different model types"""
        if model_name == 'cox':
            if 'cox' in self.models:
                # Create DataFrame with proper column names for Cox model
                # X contains only our 3 meaningful features: age, gender, serum_creatinine
                df_temp = pd.DataFrame(X, columns=self.features)
                
                # The Cox model was trained with all columns, but we need to provide them
                # even if we're only analyzing the meaningful ones
                all_cox_params = self.models['cox'].params_.index.tolist()
                
                # Add the required columns that the model expects
                for param in all_cox_params:
                    if param not in df_temp.columns:
                        if param == 'Unnamed: 0':
                            df_temp[param] = range(len(df_temp))  # Simple row index
                        elif param == 'duration_in_days':
                            df_temp[param] = 1  # Set to constant for risk scoring
                        else:
                            df_temp[param] = 0
                
                # Ensure we have all columns in the right order
                df_temp = df_temp[all_cox_params]
                return -self.models['cox'].predict_partial_hazard(df_temp).values
            
        elif model_name in ['rnn_surv', 'ddh', 'hazard_transformer', 'logistic_hazard']:
            if model_name in self.models:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = self.models[model_name]
                
                with torch.no_grad():
                    try:
                        if model_name == 'rnn_surv':
                            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
                            _, risk_scores = model(X_tensor)
                            return 1 - risk_scores.squeeze().cpu().numpy()
                            
                        elif model_name == 'ddh':
                            # DDH model requires features and mask
                            batch_size = len(X)
                            max_seq_length = 1  # For SHAP, we use single time point
                            
                            # Normalize features like in the original experiment
                            features = np.zeros((batch_size, max_seq_length, len(self.features)))
                            features[:, 0, 0] = (X[:, 0] - self.train_data['age'].mean()) / self.train_data['age'].std()
                            features[:, 0, 1] = X[:, 1]  # gender, no normalization
                            features[:, 0, 2] = (X[:, 2] - self.train_data['serum_creatinine'].mean()) / self.train_data['serum_creatinine'].std()
                            
                            mask = np.ones((batch_size, max_seq_length))  # All positions are valid
                            
                            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
                            
                            hazard_preds, _ = model(features_tensor, mask_tensor)
                            # Take the last time step prediction
                            risk_scores = hazard_preds[:, -1, 0].cpu().numpy()
                            return risk_scores
                            
                        elif model_name == 'hazard_transformer':
                            # Similar to DDH but different normalization and output
                            batch_size = len(X)
                            max_seq_length = 1
                            
                            features = np.zeros((batch_size, max_seq_length, len(self.features)))
                            features[:, 0, 0] = (X[:, 0] - self.train_data['age'].mean()) / self.train_data['age'].std()
                            features[:, 0, 1] = X[:, 1]  # gender, no normalization
                            features[:, 0, 2] = (X[:, 2] - self.train_data['serum_creatinine'].mean()) / self.train_data['serum_creatinine'].std()
                            
                            mask = np.ones((batch_size, max_seq_length))
                            
                            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
                            
                            hazard_preds, _, _ = model(features_tensor, mask_tensor)
                            risk_scores = hazard_preds[:, -1, 0].cpu().numpy()
                            return risk_scores
                            
                        elif model_name == 'logistic_hazard':
                            # LogisticHazard uses normalized features directly
                            features = np.zeros((len(X), len(self.features)))
                            features[:, 0] = (X[:, 0] - self.train_data['age'].mean()) / self.train_data['age'].std()
                            features[:, 1] = X[:, 1]  # gender, no normalization
                            features[:, 2] = (X[:, 2] - self.train_data['serum_creatinine'].mean()) / self.train_data['serum_creatinine'].std()
                            
                            X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                            
                            # The saved model is actually just the neural network
                            # Get the raw outputs and convert to risk scores
                            outputs = model(X_tensor)
                            
                            # Convert logits to probabilities and then to risk scores
                            # For survival models, we typically use 1 - survival probability
                            risk_scores = torch.sigmoid(outputs).mean(dim=1).cpu().numpy()
                            
                            return risk_scores
                            
                    except Exception as e:
                        print(f"Error with {model_name}: {e}")
                        return None
        
        return None
    
    def create_cox_wrapper_function(self):
        """Create a wrapper function for Cox model that only takes meaningful features"""
        def cox_wrapper(X):
            """
            Wrapper function that takes only the meaningful features (age, gender, serum_creatinine)
            and adds the required dummy columns for the Cox model
            """
            # X is expected to be a numpy array with shape (n_samples, 3)
            # where columns are: age, gender, serum_creatinine
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Create DataFrame with meaningful features
            df_temp = pd.DataFrame(X, columns=self.features)
            
            # Add required dummy columns
            df_temp['Unnamed: 0'] = range(len(df_temp))
            df_temp['duration_in_days'] = 1  # Constant value for risk prediction
            
            # Reorder to match Cox model expectations
            all_cox_params = self.models['cox'].params_.index.tolist()
            df_temp = df_temp[all_cox_params]
            
            # Get predictions
            results = -self.models['cox'].predict_partial_hazard(df_temp).values
            return results
        
        return cox_wrapper

    def analyze_cox_model_with_shap(self):
        """Perform SHAP analysis on Cox model"""
        if 'cox' not in self.models:
            print("Cox model not available for SHAP analysis")
            return None
            
        print("\nPerforming SHAP analysis on Cox model...")
        
        X_train = self.train_data[self.features].values
        X_test = self.test_data[self.features].values
        
        # Create wrapper function that only uses meaningful features
        cox_wrapper = self.create_cox_wrapper_function()
        
        # Create SHAP explainer using the wrapper function
        explainer = shap.Explainer(cox_wrapper, X_train)
        
        # Calculate SHAP values for a sample of test data
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]
        
        print(f"Computing SHAP values for {sample_size} samples...")
        shap_values = explainer(X_sample)
        
        # Ensure feature names are properly set
        if hasattr(shap_values, 'feature_names'):
            shap_values.feature_names = self.features
        elif hasattr(shap_values, 'data') and hasattr(shap_values.data, 'columns'):
            shap_values.data.columns = self.features
        
        # For newer SHAP versions, set feature names directly
        try:
            shap_values.feature_names = self.features
        except:
            pass
        
        self.shap_values['cox'] = shap_values
        self.explainers['cox'] = explainer
        
        return shap_values
    
    def analyze_pytorch_model_with_shap(self, model_name):
        """Perform SHAP analysis on PyTorch models"""
        if model_name not in self.models:
            print(f"{model_name} model not available for SHAP analysis")
            return None
            
        print(f"\nPerforming SHAP analysis on {model_name} model...")
        
        X_train = self.train_data[self.features].values
        X_test = self.test_data[self.features].values
        
        # Create surrogate model
        surrogate = self.create_surrogate_model(model_name)
        if surrogate is None:
            return None
            
        # Create SHAP explainer
        try:
            explainer = shap.Explainer(surrogate, X_train)
        except Exception as e:
            if "singular" in str(e).lower():
                print(f"Handling singular covariance matrix for {model_name}, using TreeExplainer...")
                try:
                    # Use TreeExplainer for problematic models
                    explainer = shap.TreeExplainer(surrogate)
                except Exception as e2:
                    print(f"TreeExplainer also failed for {model_name}, adding more noise...")
                    # Add more substantial random noise to avoid singular matrix
                    noise_scale = 1e-5 if model_name == 'hazard_transformer' else 1e-6
                    X_train_noisy = X_train + np.random.normal(0, noise_scale, X_train.shape)
                    explainer = shap.Explainer(surrogate, X_train_noisy)
            else:
                print(f"Failed to create explainer for {model_name}: {e}")
                return None
        
        # Calculate SHAP values for a sample of test data
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]
        
        print(f"Computing SHAP values for {sample_size} samples...")
        shap_values = explainer(X_sample)
        
        # Ensure feature names are properly set
        if hasattr(shap_values, 'feature_names'):
            shap_values.feature_names = self.features
        elif hasattr(shap_values, 'data') and hasattr(shap_values.data, 'columns'):
            shap_values.data.columns = self.features
        
        # For newer SHAP versions, set feature names directly
        try:
            shap_values.feature_names = self.features
        except:
            pass
        
        self.shap_values[model_name] = shap_values
        self.explainers[model_name] = explainer
        
        return shap_values
    
    def create_shap_summary_plot(self, model_name):
        """Create SHAP summary plot for a specific model"""
        if model_name not in self.shap_values:
            print(f"No SHAP values available for {model_name}")
            return
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values[model_name], 
            feature_names=self.features,
            show=False,
            plot_type="violin"
        )
        plt.title(f'SHAP Summary Plot - {model_name.upper()} Model\neGFR Components Experiment')
        plt.tight_layout()
        
        # Save plot
        output_path = f"{generate_data_path_latest_rep}/shap_summary_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP summary plot saved: {output_path}")
    
    def create_shap_bar_plot(self, model_name):
        """Create SHAP bar plot showing feature importance"""
        if model_name not in self.shap_values:
            print(f"No SHAP values available for {model_name}")
            return
            
        plt.figure(figsize=(8, 6))
        # Set feature names on the SHAP values object before plotting
        shap_values = self.shap_values[model_name]
        if hasattr(shap_values, 'feature_names'):
            shap_values.feature_names = self.features
        shap.plots.bar(shap_values, show=False)
        plt.title(f'SHAP Feature Importance - {model_name.upper()} Model\neGFR Components Experiment')
        plt.tight_layout()
        
        # Save plot
        output_path = f"{generate_data_path_latest_rep}/shap_importance_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP importance plot saved: {output_path}")
    
    def create_shap_waterfall_plots(self, model_name, n_samples=5):
        """Create SHAP waterfall plots for individual predictions"""
        if model_name not in self.shap_values:
            print(f"No SHAP values available for {model_name}")
            return
            
        shap_values = self.shap_values[model_name]
        
        for i in range(min(n_samples, len(shap_values))):
            # Create a custom SHAP Explanation object with proper feature names
            single_explanation = shap_values[i]
            
            # Create a new explanation object with proper feature names
            import shap
            explanation = shap.Explanation(
                values=single_explanation.values,
                base_values=single_explanation.base_values,
                data=single_explanation.data,
                feature_names=self.features
            )
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.title(f'SHAP Waterfall Plot - {model_name.upper()} Model (Sample {i+1})\neGFR Components Experiment')
            plt.tight_layout()
            
            # Save plot
            output_path = f"{generate_data_path_latest_rep}/shap_waterfall_{model_name}_sample{i+1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ SHAP waterfall plots saved for {model_name} ({n_samples} samples)")
    
    def create_comparative_analysis(self):
        """Create comparative SHAP analysis across different models"""
        available_models = [name for name in self.shap_values.keys()]
        
        if len(available_models) < 2:
            print("Need at least 2 models for comparative analysis")
            return
            
        print(f"\nCreating comparative SHAP analysis for models: {available_models}")
        
        # Calculate mean absolute SHAP values for each model and feature
        feature_importance_df = pd.DataFrame(index=self.features)
        
        for model_name in available_models:
            mean_shap_values = np.mean(np.abs(self.shap_values[model_name].values), axis=0)
            feature_importance_df[model_name] = mean_shap_values
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SHAP Analysis Comparison - eGFR Components Experiment', fontsize=16)
        
        # Plot 1: Feature importance comparison
        ax1 = axes[0, 0]
        feature_importance_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Mean Absolute SHAP Values by Model')
        ax1.set_ylabel('Mean |SHAP Value|')
        ax1.set_xlabel('Features')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Feature importance heatmap
        ax2 = axes[0, 1]
        sns.heatmap(feature_importance_df.T, annot=True, cmap='YlOrRd', ax=ax2, fmt='.3f')
        ax2.set_title('Feature Importance Heatmap')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Models')
        
        # Plot 3: Normalized feature importance
        ax3 = axes[1, 0]
        normalized_importance = feature_importance_df.div(feature_importance_df.sum(axis=0), axis=1)
        normalized_importance.plot(kind='bar', ax=ax3, stacked=True)
        ax3.set_title('Normalized Feature Importance (Relative)')
        ax3.set_ylabel('Proportion of Importance')
        ax3.set_xlabel('Features')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Model agreement analysis
        ax4 = axes[1, 1]
        correlation_matrix = feature_importance_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4, fmt='.3f')
        ax4.set_title('Model Agreement (Correlation)')
        
        plt.tight_layout()
        
        # Save comparison plot
        output_path = f"{generate_data_path_latest_rep}/shap_comparative_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparative SHAP analysis saved: {output_path}")
        
        return feature_importance_df
    
    def generate_detailed_report(self, feature_importance_df=None):
        """Generate a detailed SHAP analysis report"""
        report = []
        report.append("=" * 80)
        report.append("SHAP ANALYSIS REPORT - eGFR COMPONENTS EXPERIMENT")
        report.append("=" * 80)
        report.append("")
        
        report.append("EXPERIMENT OVERVIEW:")
        report.append(f"  - Scenario: eGFR Components")
        report.append(f"  - Features: {', '.join(self.features)}")
        report.append(f"  - Training samples: {len(self.train_data):,}")
        report.append(f"  - Test samples: {len(self.test_data):,}")
        report.append("")
        
        report.append("MODELS ANALYZED:")
        for model_name in self.models.keys():
            if not model_name.endswith('_surrogate'):
                status = "✓" if model_name in self.shap_values else "✗"
                report.append(f"  {status} {model_name.upper()}")
        report.append("")
        
        if feature_importance_df is not None:
            report.append("FEATURE IMPORTANCE SUMMARY:")
            report.append("-" * 50)
            
            # Calculate overall importance ranking
            mean_importance = feature_importance_df.mean(axis=1).sort_values(ascending=False)
            
            for i, (feature, importance) in enumerate(mean_importance.items(), 1):
                report.append(f"{i}. {feature}: {importance:.4f} (mean absolute SHAP)")
                
                # Add interpretation
                if feature == 'serum_creatinine':
                    report.append("   → Primary biomarker for kidney function")
                elif feature == 'age':
                    report.append("   → Age-related decline in kidney function")
                elif feature == 'gender':
                    report.append("   → Gender differences in creatinine metabolism")
                report.append("")
            
            report.append("MODEL-SPECIFIC INSIGHTS:")
            report.append("-" * 50)
            
            for model_name in feature_importance_df.columns:
                report.append(f"\n{model_name.upper()} Model:")
                model_importance = feature_importance_df[model_name].sort_values(ascending=False)
                
                for feature, importance in model_importance.items():
                    percentage = (importance / model_importance.sum()) * 100
                    report.append(f"  - {feature}: {importance:.4f} ({percentage:.1f}%)")
        
        report.append("\n" + "=" * 80)
        report.append("INTERPRETATION NOTES:")
        report.append("=" * 80)
        report.append("• Higher absolute SHAP values indicate greater feature importance")
        report.append("• Positive SHAP values increase the risk prediction")
        report.append("• Negative SHAP values decrease the risk prediction")
        report.append("• eGFR components (age, gender, serum_creatinine) are used instead of")
        report.append("  the calculated eGFR value to understand the individual contributions")
        report.append("• This analysis helps identify which demographic and laboratory")
        report.append("  factors are most predictive of ESRD risk")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Run the complete SHAP analysis pipeline"""
        print("Starting comprehensive SHAP analysis for eGFR Components experiment...")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            print("Failed to load data. Exiting...")
            return
        
        # Load available models
        models_to_analyze = []
        
        # Load Cox model
        if self.load_cox_model():
            models_to_analyze.append('cox')
        
        # Load PyTorch models
        pytorch_models = {
            'rnn_surv': f"{generate_data_path_latest_rep}/egfr_components_rnn_surv_model.pt",
            'ddh': f"{generate_data_path_latest_rep}/egfr_components_ddh_model.pt",
            'hazard_transformer': f"{generate_data_path_latest_rep}/egfr_components_hazard_transformer_model.pt",
            'logistic_hazard': f"{generate_data_path_latest_rep}/egfr_components_logistic_hazard_model.pt"
        }
        
        for model_name, model_path in pytorch_models.items():
            if self.load_pytorch_model(model_path, model_name):
                models_to_analyze.append(model_name)
        
        if not models_to_analyze:
            print("No models available for analysis. Exiting...")
            return
        
        print(f"\nAnalyzing models: {models_to_analyze}")
        print("=" * 70)
        
        # Perform SHAP analysis for each model
        successful_models = []
        for model_name in models_to_analyze:
            print(f"\nAnalyzing {model_name.upper()} model...")
            
            try:
                if model_name == 'cox':
                    shap_values = self.analyze_cox_model_with_shap()
                else:
                    shap_values = self.analyze_pytorch_model_with_shap(model_name)
                
                if shap_values is not None:
                    successful_models.append(model_name)
                    # Create individual plots
                    self.create_shap_summary_plot(model_name)
                    self.create_shap_bar_plot(model_name)
                    self.create_shap_waterfall_plots(model_name, n_samples=3)
                else:
                    print(f"✗ SHAP analysis failed for {model_name}")
                    
            except Exception as e:
                print(f"✗ Error analyzing {model_name}: {str(e)}")
                continue
        
        if not successful_models:
            print("No successful SHAP analyses. Exiting...")
            return
        
        # Create comparative analysis
        if len(successful_models) > 1:
            print("\nCreating comparative analysis...")
            feature_importance_df = self.create_comparative_analysis()
        else:
            feature_importance_df = None
            print(f"\nSkipping comparative analysis (only {len(successful_models)} successful model(s))")
        
        # Generate detailed report
        print("\nGenerating detailed report...")
        report = self.generate_detailed_report(feature_importance_df)
        
        # Save report
        report_path = f"{generate_data_path_latest_rep}/egfr_components_shap_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Detailed report saved: {report_path}")
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Generated files:")
        print(f"  - Report: egfr_components_shap_analysis_report.txt")
        print(f"  - Plots: shap_*.png files in {generate_data_path_latest_rep}/")
        print(f"  - Successful analyses: {len(successful_models)} models ({', '.join(successful_models)})")
        
        return report


def main():
    """Main execution function"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(generate_data_path_latest_rep, exist_ok=True)
        
        # Initialize and run analysis
        analyzer = EGFRComponentsSHAPAnalysis()
        report = analyzer.run_complete_analysis()
        
        if report:
            print("\n" + "=" * 70)
            print("ANALYSIS SUMMARY")
            print("=" * 70)
            print("✓ SHAP analysis completed successfully")
            print("✓ All plots and reports generated")
            print("✓ Check the generated_data/rep1/ directory for outputs")
        
    except Exception as e:
        print(f"An error occurred during SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
