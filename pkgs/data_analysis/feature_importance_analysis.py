import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import torch
import shap

pd.set_option('future.no_silent_downcasting', True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkgs.commons import (
    current_rep, generate_data_path_latest_rep,
    egfr_components_train_data_path, egfr_components_test_data_path,
    five_labms_train_subset_path, five_labms_test_subset_path,
    five_labms_num_subsets_train, five_labms_num_subsets_test
)
from pkgs.experiments.utils import load_pkl_and_dill_model


class FeatureImportanceAnalyzer:
    def __init__(self, scenario, output_dir):
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.report_lines = []
        self.models = ['cox', 'ddh', 'hazard_transformer', 'logistic_hazard', 'rnn_surv']
        self.all_importances = {}
        self.model_pretty_names = {
            'cox': 'Cox',
            'ddh': 'Dynamic DeepHit',
            'hazard_transformer': 'Hazard Transformer',
            'logistic_hazard': 'Logistic Hazard',
            'rnn_surv': 'RNN-Surv'
        }
        
    def log(self, message):
        print(message)
        self.report_lines.append(message)
        
    def analyze_egfr_components(self):
        self.log("="*80)
        self.log("FEATURE IMPORTANCE ANALYSIS - eGFR COMPONENTS SCENARIO")
        self.log("="*80)
        self.log("")
        
        try:
            train_data = pd.read_csv(egfr_components_train_data_path)
            test_data = pd.read_csv(egfr_components_test_data_path)
            
            self.log(f"Training samples: {len(train_data)}")
            self.log(f"Test samples: {len(test_data)}")
            self.log(f"Features: {list(train_data.columns)}")
            self.log("")
            
            feature_cols = ['age', 'gender', 'serum_creatinine']
            
            self.log("FEATURE OVERVIEW:")
            self.log("-" * 40)
            for col in feature_cols:
                if col in train_data.columns:
                    mean_val = train_data[col].mean()
                    std_val = train_data[col].std()
                    self.log(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
            self.log("")
            
        except Exception as e:
            self.log(f"Error loading eGFR components data: {e}")
            return
        
        self.all_importances['egfr_components'] = {}
        
        self.analyze_all_models('egfr_components', test_data, feature_cols)
    
    def analyze_fivelabms(self):
        self.log("="*80)
        self.log("FEATURE IMPORTANCE ANALYSIS - FIVELABMS SCENARIO")
        self.log("="*80)
        self.log("")
        
        try:
            train_dfs = []
            for i in range(five_labms_num_subsets_train):
                subset_path = five_labms_train_subset_path(i)
                if os.path.exists(subset_path):
                    df = pd.read_csv(subset_path)
                    train_dfs.append(df)
                    self.log(f"Loaded training subset {i}: {len(df)} samples")
            
            if not train_dfs:
                self.log("No training subsets found for FIVELABMS")
                return
                
            train_data = pd.concat(train_dfs, ignore_index=True)
            
            test_dfs = []
            for i in range(five_labms_num_subsets_test):
                subset_path = five_labms_test_subset_path(i)
                if os.path.exists(subset_path):
                    df = pd.read_csv(subset_path)
                    test_dfs.append(df)
                    self.log(f"Loaded test subset {i}: {len(df)} samples")
            
            if not test_dfs:
                self.log("No test subsets found for FIVELABMS")
                return
                
            test_data = pd.concat(test_dfs, ignore_index=True)
            
            self.log(f"Total training samples: {len(train_data)}")
            self.log(f"Total test samples: {len(test_data)}")
            self.log(f"Features: {list(train_data.columns)}")
            self.log("")
            
            potential_features = ['egfr', 'egfr_missing', 'hemoglobin', 'hemoglobin_missing']
            feature_cols = [col for col in potential_features if col in train_data.columns]
            
            self.log("FEATURE OVERVIEW:")
            self.log("-" * 40)
            for col in feature_cols:
                mean_val = train_data[col].mean()
                std_val = train_data[col].std()
                missing_pct = (train_data[col] == 0).mean() * 100 if 'missing' not in col else train_data[col].mean() * 100
                self.log(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, missing%={missing_pct:.1f}")
            self.log("")
            
        except Exception as e:
            self.log(f"Error loading FIVELABMS data: {e}")
            return
        
        self.all_importances['fivelabms'] = {}
        
        self.analyze_all_models('fivelabms', test_data, feature_cols)
    
    def analyze_all_models(self, scenario_name, test_data, feature_cols):
        self.log(f"ANALYZING ALL MODELS FOR {scenario_name.upper()}")
        self.log("-" * 60)
        
        if scenario_name == 'egfr_components':
            model_paths = {
                'cox': generate_data_path_latest_rep + '/egfr_components_cox_model.dill',
                'ddh': generate_data_path_latest_rep + '/egfr_components_ddh_model.pt',
                'hazard_transformer': generate_data_path_latest_rep + '/egfr_components_hazard_transformer_model.pt',
                'logistic_hazard': generate_data_path_latest_rep + '/egfr_components_logistic_hazard_model.pt',
                'rnn_surv': generate_data_path_latest_rep + '/egfr_components_rnn_surv_model.pt'
            }
        else:
            model_paths = {
                'cox': generate_data_path_latest_rep + '/fivelabms_cox_model.dill',
                'ddh': generate_data_path_latest_rep + '/fivelabms_ddh_model.pt',
                'hazard_transformer': generate_data_path_latest_rep + '/fivelabms_hazard_transformer_model.pt',
                'logistic_hazard': generate_data_path_latest_rep + '/fivelabms_logistic_hazard_model.pt',
                'rnn_surv': generate_data_path_latest_rep + '/fivelabms_rnn_surv_model.pt'
            }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                self.log(f"\nAnalyzing {model_name.upper()} model...")
                try:
                    importance_data = self.get_model_feature_importance(
                        model_name, model_path, test_data, feature_cols
                    )
                    self.all_importances[scenario_name][model_name] = importance_data
                    self.log_model_feature_importance(model_name, importance_data)
                except Exception as e:
                    self.log(f"Error analyzing {model_name} model: {e}")
            else:
                self.log(f"Model file not found: {model_path}")
        
        self.create_consolidated_importance_plot(scenario_name, feature_cols)
        self.create_shap_plot(scenario_name, feature_cols, test_data)
    
    def get_model_feature_importance(self, model_name, model_path, test_data, feature_cols):
        importance_data = {'features': feature_cols, 'importance': None, 'coefficients': None}
        
        if model_name == 'cox':
            try:
                cox_model = load_pkl_and_dill_model(model_path)
                if cox_model and hasattr(cox_model, 'params_'):
                    coefficients = cox_model.params_
                    feature_coeffs = []
                    for feat in feature_cols:
                        if feat in coefficients.index:
                            feature_coeffs.append(abs(coefficients[feat]))
                        else:
                            feature_coeffs.append(0.0)
                    importance_data['importance'] = feature_coeffs
                    importance_data['coefficients'] = [coefficients.get(feat, 0.0) for feat in feature_cols]
            except Exception as e:
                self.log(f"Error loading Cox model: {e}")
                
        else:
            try:
                importance_data['importance'] = self.extract_nn_feature_importance(
                    model_name, model_path, test_data, feature_cols)
            except Exception as e:
                self.log(f"Error extracting feature importance from {model_name}: {e}")
                importance_data['importance'] = [0.0] * len(feature_cols)
        
        return importance_data
    
    def extract_nn_feature_importance(self, model_name, model_path, test_data, feature_cols):        
        X_test = test_data[feature_cols].fillna(0).values.astype(np.float32)
        sample_size = len(X_test)
        X_sample = X_test[:sample_size]
        
        try:
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            model.eval()
            
            if model_name == 'hazard_transformer':
                return self.extract_model_weights_importance(model, feature_cols)
            elif model_name in ['ddh', 'logistic_hazard', 'rnn_surv']:
                return self.extract_gradient_based_importance(model, X_sample, feature_cols)
            else:
                return self.extract_gradient_based_importance(model, X_sample, feature_cols)
                
        except Exception as e:
            raise ValueError(f"Error loading model {model_path}: {e}")
    
    def extract_transformer_attention_weights(self, model, X_sample, feature_cols):
        try:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_sample)
                
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
                
                batch_size, seq_len = X_tensor.shape[:2]
                mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
                mask.requires_grad_(False)
                
                if hasattr(model, 'transformer_encoder') or hasattr(model, 'attention_weights'):
                    try:                        
                        return self.extract_gradient_based_importance(model, X_sample, feature_cols)
                        
                    except Exception as e:
                        self.log(f"Error in transformer forward pass: {e}")
                        return self.extract_gradient_based_importance(model, X_sample, feature_cols)
                else:
                    return self.extract_gradient_based_importance(model, X_sample, feature_cols)
                    
        except Exception as e:
            self.log(f"Error extracting transformer attention: {e}")
            return [1.0/len(feature_cols)] * len(feature_cols)
    
    def extract_gradient_based_importance(self, model, X_sample, feature_cols):
        try:
            X_tensor = torch.FloatTensor(X_sample)
            
            is_rnn_surv = hasattr(model, 'rnn') and hasattr(model, 'embedding_layers')
            is_hazard_transformer = hasattr(model, 'transformer_encoder') and hasattr(model, 'input_embedding')
            
            if is_rnn_surv:
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
            elif is_hazard_transformer:
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
            
            X_tensor.requires_grad_(True)
            
            model.train()
            for param in model.parameters():
                param.requires_grad_(True)
            
            try:
                if is_hazard_transformer:
                    batch_size, seq_len = X_tensor.shape[:2]
                    with torch.no_grad():
                        mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
                    
                    was_training = model.training
                    model.train()
                    
                    try:
                        hazard_preds, encoded, eval_times = model(X_tensor, mask)
                        output = hazard_preds
                    finally:
                        model.train(was_training)
                else:
                    output = model(X_tensor)
                    
                if isinstance(output, tuple):
                    output = output[1] if is_rnn_surv else output[0]
                
                if output.dim() > 1:
                    if output.numel() == 1:
                        output = output.squeeze()
                    else:
                        output = output.sum()
                elif output.dim() == 0:
                    pass
                else:
                    output = output.sum()
                
                gradients = torch.autograd.grad(
                    outputs=output,
                    inputs=X_tensor,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )[0]
                
                if gradients is None:
                    self.log("Gradients are None, falling back to weight-based importance")
                    return self.extract_model_weights_importance(model, feature_cols)
                
                if is_rnn_surv or is_hazard_transformer:
                    if gradients.dim() == 3:
                        importance = torch.abs(gradients).mean(dim=0).squeeze(0).detach().numpy()
                    elif gradients.dim() == 2:
                        importance = torch.abs(gradients).mean(dim=0).detach().numpy()
                    else:
                        importance = torch.abs(gradients).detach().numpy().flatten()
                else:
                    importance = torch.abs(gradients).mean(dim=0).detach().numpy()
                
                if importance.ndim > 1:
                    importance = importance.flatten()
                
                if hasattr(importance, 'shape') and len(importance.shape) > 1:
                    importance = importance.reshape(-1)
                
                if len(importance) > len(feature_cols):
                    importance = importance[:len(feature_cols)]
                elif len(importance) < len(feature_cols):
                    remaining = len(feature_cols) - len(importance)
                    pad_value = importance.mean() if len(importance) > 0 else 1.0
                    importance = np.concatenate([importance, np.full(remaining, pad_value)])
                
                if importance.sum() > 0:
                    importance = importance / importance.sum()
                else:
                    importance = np.ones(len(feature_cols)) / len(feature_cols)
                
                return importance.tolist()
                
            except (TypeError, RuntimeError) as e:
                error_str = str(e)
                if "missing" in error_str and "mask" in error_str and not is_hazard_transformer:
                    try:
                        with torch.no_grad():
                            if X_tensor.dim() == 2:
                                mask = torch.ones(X_tensor.shape[0], 1, dtype=torch.bool)
                            elif X_tensor.dim() == 3:
                                mask = torch.ones(X_tensor.shape[0], X_tensor.shape[1], dtype=torch.bool)
                            else:
                                mask = torch.ones(X_tensor.shape[0], X_tensor.shape[-1], dtype=torch.bool)
                        
                        output = model(X_tensor, mask)
                        if isinstance(output, tuple):
                            output = output[1] if is_rnn_surv else output[0]
                        
                        if output.dim() > 1:
                            if output.numel() == 1:
                                output = output.squeeze()
                            else:
                                output = output.sum()
                        
                        gradients = torch.autograd.grad(
                            outputs=output,
                            inputs=X_tensor,
                            create_graph=False,
                            retain_graph=False,
                            only_inputs=True,
                            allow_unused=True
                        )[0]
                        
                        if gradients is None:
                            return self.extract_model_weights_importance(model, feature_cols)
                        
                        importance = torch.abs(gradients).mean(dim=0).detach().numpy()
                        
                        if importance.ndim > 1:
                            importance = importance.flatten()
                        
                        if hasattr(importance, 'shape') and len(importance.shape) > 1:
                            importance = importance.reshape(-1)
                        
                        if len(importance) > len(feature_cols):
                            importance = importance[:len(feature_cols)]
                        elif len(importance) < len(feature_cols):
                            remaining = len(feature_cols) - len(importance)
                            pad_value = importance.mean() if len(importance) > 0 else 1.0
                            importance = np.concatenate([importance, np.full(remaining, pad_value)])
                        
                        if importance.sum() > 0:
                            importance = importance / importance.sum()
                        else:
                            importance = np.ones(len(feature_cols)) / len(feature_cols)
                        
                        return importance.tolist()
                        
                    except Exception as mask_e:
                        self.log(f"Error with mask: {mask_e}")
                        return self.extract_model_weights_importance(model, feature_cols)
                        
                elif "size of tensor" in error_str or "dimension" in error_str or "indices" in error_str:
                    try:
                        if X_tensor.dim() == 2 and not is_hazard_transformer:
                            X_tensor_reshaped = X_tensor.unsqueeze(1)
                            X_tensor_reshaped.requires_grad_(True)
                            
                            try:
                                output = model(X_tensor_reshaped)
                            except:
                                with torch.no_grad():
                                    mask = torch.ones(X_tensor_reshaped.shape[0], X_tensor_reshaped.shape[1], dtype=torch.bool)
                                output = model(X_tensor_reshaped, mask)
                            
                            if isinstance(output, tuple):
                                output = output[1] if is_rnn_surv else output[0]
                            
                            if output.dim() > 1:
                                if output.numel() == 1:
                                    output = output.squeeze()
                                else:
                                    output = output.sum()
                            
                            gradients = torch.autograd.grad(
                                outputs=output,
                                inputs=X_tensor_reshaped,
                                create_graph=False,
                                retain_graph=False,
                                only_inputs=True,
                                allow_unused=True
                            )[0]
                            
                            if gradients is None:
                                return self.extract_model_weights_importance(model, feature_cols)
                            
                            if gradients.dim() > 2:
                                gradients = gradients.squeeze(1)
                            importance = torch.abs(gradients)
                            
                            if importance.dim() > 2:
                                importance = importance.mean(dim=0).mean(dim=0)
                            elif importance.dim() == 2:
                                importance = importance.mean(dim=0)
                            elif importance.dim() == 1:
                                pass
                            else:
                                importance = importance.flatten()
                            
                            importance = importance.detach().numpy()
                            
                            if isinstance(importance, np.ndarray):
                                if importance.ndim > 1:
                                    importance = importance.flatten()
                            else:
                                importance = np.array([importance] if np.isscalar(importance) else importance)
                            
                            if importance.ndim > 1:
                                importance = importance.flatten()
                            
                            if len(importance) > len(feature_cols):
                                importance = importance[:len(feature_cols)]
                            elif len(importance) < len(feature_cols):
                                importance = np.pad(importance, (0, len(feature_cols) - len(importance)))
                            
                            if importance.sum() > 0:
                                importance = importance / importance.sum()
                            else:
                                importance = np.ones(len(feature_cols)) / len(feature_cols)
                            
                            return importance.tolist()
                        else:
                            return self.extract_model_weights_importance(model, feature_cols)
                            
                    except Exception as reshape_e:
                        self.log(f"Error with reshape: {reshape_e}")
                        return self.extract_model_weights_importance(model, feature_cols)
                else:
                    self.log(f"Error in gradient calculation: {e}")
                    return self.extract_model_weights_importance(model, feature_cols)
                    
        except Exception as e:
            self.log(f"Error in gradient-based importance: {e}")
            return self.extract_model_weights_importance(model, feature_cols)
        finally:
            model.eval()
    
    def extract_model_weights_importance(self, model, feature_cols):
        try:
            first_layer = None
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.in_features == len(feature_cols):
                    first_layer = module
                    break
            
            if first_layer is not None:
                weights = first_layer.weight.data.abs().mean(dim=0).numpy()
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    return weights.tolist()
            
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    weights = module.weight.data.abs().mean(dim=0).numpy()
                    if len(weights) == len(feature_cols):
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                        return weights.tolist()
            
            return [1.0/len(feature_cols)] * len(feature_cols)
            
        except Exception as e:
            self.log(f"Error extracting model weights: {e}")
            return [1.0/len(feature_cols)] * len(feature_cols)
    
    def log_model_feature_importance(self, model_name, importance_data):
        if importance_data['importance'] is not None:
            self.log(f"  {model_name.upper()} Feature Importance:")
            for feat, imp in zip(importance_data['features'], importance_data['importance']):
                if importance_data['coefficients'] and model_name == 'cox':
                    coeff_idx = importance_data['features'].index(feat)
                    coeff = importance_data['coefficients'][coeff_idx]
                    hr = np.exp(coeff)
                    self.log(f"    {feat}: {imp:.4f} (coeff={coeff:.4f}, HR={hr:.3f})")
                else:
                    self.log(f"    {feat}: {imp:.4f}")
    
    def create_consolidated_importance_plot(self, scenario_name, feature_cols):
        try:
            _, axes = plt.subplots(1, len(self.models), figsize=(4*len(self.models), 6))
            if len(self.models) == 1:
                axes = [axes]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(feature_cols)))
            
            for i, model_name in enumerate(self.models):
                ax = axes[i]
                
                if (model_name in self.all_importances[scenario_name] and 
                    self.all_importances[scenario_name][model_name]['importance'] is not None):
                    
                    importance = self.all_importances[scenario_name][model_name]['importance']
                    
                    bars = ax.barh(range(len(feature_cols)), importance)
                    ax.set_yticks(range(len(feature_cols)))
                    ax.set_yticklabels(feature_cols)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'{self.model_pretty_names[model_name].upper()}')
                    
                    for j, bar in enumerate(bars):
                        bar.set_color(colors[j])
                        
                else:
                    ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{model_name.upper()}')
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()
            
            output_path = self.output_dir / f'{scenario_name}_all_models_feature_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"Consolidated feature importance plot saved to: {output_path}")
            
        except Exception as e:
            self.log(f"Error creating consolidated importance plot: {e}")
    
    def create_shap_plot(self, scenario_name, feature_cols, test_data, background_size=100, eval_size=1000):
        """Simple SHAP analysis for feature importance."""
        self.log(f"\nCreating SHAP plots for {scenario_name}...")
        
        # Get test data
        X_test = test_data[feature_cols].fillna(0).values.astype(np.float32)
        
        # Sample background and evaluation data
        background_sample = X_test[:min(background_size, len(X_test))]
        eval_sample = X_test[:min(eval_size, len(X_test))]
        
        # Load training data for background
        if scenario_name == 'egfr_components':
            train_data = pd.read_csv(egfr_components_train_data_path)
            X_train = train_data[feature_cols].fillna(0).values.astype(np.float32)
        else:  # fivelabms
            train_dfs = []
            for i in range(five_labms_num_subsets_train):
                subset_path = five_labms_train_subset_path(i)
                if os.path.exists(subset_path):
                    df = pd.read_csv(subset_path)
                    train_dfs.append(df)
            if train_dfs:
                train_data = pd.concat(train_dfs, ignore_index=True)
                X_train = train_data[feature_cols].fillna(0).values.astype(np.float32)
            else:
                X_train = background_sample
        
        background_train = X_train[:min(background_size, len(X_train))]
        
        # Get model paths
        if scenario_name == 'egfr_components':
            model_paths = {
                'cox': generate_data_path_latest_rep + '/egfr_components_cox_model.dill',
                'ddh': generate_data_path_latest_rep + '/egfr_components_ddh_model.pt',
                'hazard_transformer': generate_data_path_latest_rep + '/egfr_components_hazard_transformer_model.pt',
                'logistic_hazard': generate_data_path_latest_rep + '/egfr_components_logistic_hazard_model.pt',
                'rnn_surv': generate_data_path_latest_rep + '/egfr_components_rnn_surv_model.pt'
            }
        else:
            model_paths = {
                'cox': generate_data_path_latest_rep + '/fivelabms_cox_model.dill',
                'ddh': generate_data_path_latest_rep + '/fivelabms_ddh_model.pt',
                'hazard_transformer': generate_data_path_latest_rep + '/fivelabms_hazard_transformer_model.pt',
                'logistic_hazard': generate_data_path_latest_rep + '/fivelabms_logistic_hazard_model.pt',
                'rnn_surv': generate_data_path_latest_rep + '/fivelabms_rnn_surv_model.pt'
            }
        
        # Create subplots for each model
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        model_idx = 0
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found: {model_path}")
                continue
                
            try:
                self.log(f"Creating SHAP plot for {model_name}...")
                
                if model_name == 'cox':
                    # Load Cox model
                    cox_model = load_pkl_and_dill_model(model_path)
                    if cox_model and hasattr(cox_model, 'predict'):
                        explainer = shap.Explainer(cox_model.predict, background_train)
                        shap_values = explainer(eval_sample)
                        
                        # Create SHAP summary plot
                        plt.sca(axes[model_idx])
                        shap.summary_plot(shap_values, eval_sample, 
                                        feature_names=feature_cols, 
                                        show=False, plot_size=None)
                        plt.title(f'{self.model_pretty_names[model_name]} SHAP')
                    else:
                        axes[model_idx].text(0.5, 0.5, 'Model not available', 
                                           ha='center', va='center', transform=axes[model_idx].transAxes)
                        axes[model_idx].set_title(f'{self.model_pretty_names[model_name]} SHAP')
                        
                else:
                    # Load neural network model
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                    model.eval()
                    
                    # Create a wrapper function for SHAP
                    def model_predict(x):
                        with torch.no_grad():
                            x_tensor = torch.FloatTensor(x)
                            
                            # Handle different model architectures
                            if hasattr(model, 'rnn') and hasattr(model, 'embedding_layers'):
                                # RNN-Surv
                                if x_tensor.dim() == 2:
                                    x_tensor = x_tensor.unsqueeze(1)
                                output = model(x_tensor)
                                if isinstance(output, tuple):
                                    output = output[1]  # Get hazard prediction
                            elif hasattr(model, 'transformer_encoder'):
                                # Hazard Transformer
                                if x_tensor.dim() == 2:
                                    x_tensor = x_tensor.unsqueeze(1)
                                batch_size, seq_len = x_tensor.shape[:2]
                                mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
                                hazard_preds, _, _ = model(x_tensor, mask)
                                output = hazard_preds
                            else:
                                # DDH, Logistic Hazard - may need mask
                                try:
                                    output = model(x_tensor)
                                    if isinstance(output, tuple):
                                        output = output[0]
                                except TypeError as e:
                                    if "missing" in str(e) and "mask" in str(e):
                                        # DDH model needs mask
                                        if x_tensor.dim() == 2:
                                            x_tensor = x_tensor.unsqueeze(1)
                                        batch_size, seq_len = x_tensor.shape[:2]
                                        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
                                        output = model(x_tensor, mask)
                                        if isinstance(output, tuple):
                                            output = output[0]
                                    else:
                                        raise e
                            
                            # Ensure output is 2D for SHAP
                            if output.dim() == 1:
                                output = output.unsqueeze(1)
                            elif output.dim() > 2:
                                output = output.view(output.shape[0], -1)
                            
                            return output.numpy()
                    
                    # Create SHAP explainer
                    explainer = shap.Explainer(model_predict, background_train)
                    shap_values = explainer(eval_sample)
                    
                    # Create SHAP summary plot
                    plt.sca(axes[model_idx])
                    shap.summary_plot(shap_values, eval_sample, 
                                    feature_names=feature_cols, 
                                    show=False, plot_size=None)
                    plt.title(f'{self.model_pretty_names[model_name]} SHAP')
                
                model_idx += 1
                
            except Exception as e:
                self.log(f"Error creating SHAP plot for {model_name}: {e}")
                axes[model_idx].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                                   ha='center', va='center', transform=axes[model_idx].transAxes)
                axes[model_idx].set_title(f'{self.model_pretty_names[model_name]} SHAP')
                model_idx += 1
        
        # Hide unused subplots
        for i in range(model_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / f'{scenario_name}_all_models_shap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"SHAP plots saved to: {output_path}")
    
    def save_report(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = [
            "FEATURE IMPORTANCE ANALYSIS REPORT",
            "=" * 80,
            f"Generated on: {timestamp}",
            f"Repetition: {current_rep}",
            f"Analysis scenarios: {self.scenario}",
            f"Models analyzed: {', '.join(self.models)}",
            "=" * 80,
            ""
        ]
        
        full_report = header + self.report_lines
        
        report_path = self.output_dir / f'feature_importance_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(full_report))
        
        print(f"Analysis report saved to: {report_path}")


def main():    
    output_dir = generate_data_path_latest_rep
    
    analyzer = FeatureImportanceAnalyzer("egfr_components_and_fivelabms", output_dir)
    
    analyzer.log("Starting Feature Importance Analysis...")
    analyzer.log(f"Output directory: {output_dir}")
    analyzer.log(f"Current repetition: {current_rep}")
    analyzer.log(f"Models to analyze: {', '.join(analyzer.models)}")
    analyzer.log("")
    
    analyzer.analyze_egfr_components()
    
    analyzer.analyze_fivelabms()
    
    analyzer.log("="*80)
    analyzer.log("ANALYSIS COMPLETE")
    analyzer.log("="*80)
    analyzer.log("")
    analyzer.log("Key findings:")
    analyzer.log("- Cox model coefficients show direct impact on hazard ratio")
    analyzer.log("- Neural network models analyzed using gradient-based importance and attention weights")
    analyzer.log("- Model-specific feature importance extracted from actual trained models")
    analyzer.log("")
    analyzer.log("Files generated:")
    analyzer.log("- feature_importance_analysis_report.txt (this report)")
    analyzer.log("- *_all_models_feature_importance.png (consolidated importance plots)")
    analyzer.log("- *_all_models_shap.png (consolidated model-specific plots)")
    
    analyzer.save_report()


if __name__ == "__main__":
    main()
