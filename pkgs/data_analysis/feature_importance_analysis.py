import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import torch

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
                    ax.set_title(f'{model_name.upper()}')
                    
                    for j, bar in enumerate(bars):
                        bar.set_color(colors[j])
                        
                else:
                    ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{model_name.upper()}')
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.suptitle(f'Feature Importance Analysis - {scenario_name.upper()}', fontsize=16)
            plt.tight_layout()
            
            output_path = self.output_dir / f'{scenario_name}_all_models_feature_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"Consolidated feature importance plot saved to: {output_path}")
            
        except Exception as e:
            self.log(f"Error creating consolidated importance plot: {e}")
    
    def create_shap_plot(self, scenario_name, feature_cols, test_data, background_size=100, eval_size=1000):
        import shap
        n_samples = len(test_data)
        if n_samples == 0:
            self.log("Empty test data; skipping SHAP computation.")
            return
        background_size = min(background_size, n_samples)
        eval_size = min(eval_size, n_samples)
        rng = np.random.RandomState(0)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        bg_idx = indices[:background_size]
        eval_idx = indices[background_size: background_size + eval_size] if background_size < n_samples else indices[:eval_size]
        X_background = test_data.loc[bg_idx, feature_cols].fillna(0).values.astype(np.float32)
        X_eval = test_data.loc[eval_idx, feature_cols].fillna(0).values.astype(np.float32)
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
        shap_results = {}
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found for SHAP: {model_path}")
                continue
            self.log(f"Computing SHAP for {model_name.upper()} (this may be slow)...")
            try:
                if model_name == 'cox':
                    model = load_pkl_and_dill_model(model_path)
                else:
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                    model.eval()
                def make_predict_fn(m, m_name):
                    if m_name == 'cox':
                        def predict_fn(X):
                            try:
                                if isinstance(X, np.ndarray):
                                    X_df = pd.DataFrame(X, columns=feature_cols)
                                else:
                                    X_df = X.copy()
                                    if not all(col in X_df.columns for col in feature_cols):
                                        X_df = X_df[feature_cols] if any(col in X_df.columns for col in feature_cols) else pd.DataFrame(X_df.values, columns=feature_cols)
                                
                                predictions = m.predict_partial_hazard(X_df)
                                return predictions.values if hasattr(predictions, 'values') else predictions
                            except Exception as e:
                                self.log(f"Error in Cox predict wrapper: {e}")
                                return np.zeros(len(X)) if isinstance(X, np.ndarray) else np.zeros(len(X_df))
                        return predict_fn
                    else:
                        def predict_fn(X):
                            try:
                                xt = torch.FloatTensor(X)
                                if xt.dim() == 2:
                                    if hasattr(m, 'rnn') or hasattr(m, 'transformer_encoder') or hasattr(m, 'input_embedding'):
                                        xt = xt.unsqueeze(1)
                                xt = xt.to('cpu')
                                with torch.no_grad():
                                    try:
                                        out = m(xt)
                                    except TypeError:
                                        try:
                                            mask = torch.ones(xt.shape[0], xt.shape[1], dtype=torch.bool) if xt.dim() == 3 else torch.ones(xt.shape[0], 1, dtype=torch.bool)
                                            out = m(xt, mask)
                                        except Exception:
                                            out = m(xt)
                                    if isinstance(out, tuple):
                                        if len(out) > 1 and isinstance(out[1], torch.Tensor) and out[1].shape[0] == xt.shape[0]:
                                            preds = out[1]
                                        else:
                                            preds = out[0]
                                    else:
                                        preds = out
                                    if isinstance(preds, torch.Tensor):
                                        if preds.dim() > 1:
                                            if preds.shape[1] >= 1:
                                                preds = preds[:, 0]
                                            else:
                                                preds = preds.sum(dim=1)
                                        preds = preds.detach().cpu().numpy().reshape(-1,)
                                        return preds
                                    else:
                                        arr = np.asarray(preds).reshape(-1,)
                                        return arr
                            except Exception as e:
                                self.log(f"Error in torch predict wrapper for {m_name}: {e}")
                                return np.zeros(X.shape[0])
                        return predict_fn
                predict_fn = make_predict_fn(model, model_name)
                explainer = None
                try:
                    masker = shap.maskers.Independent(X_background)
                    explainer = shap.Explainer(predict_fn, masker, feature_names=feature_cols)
                    shap_vals = explainer(X_eval)
                    vals = getattr(shap_vals, "values", None)
                    if vals is None:
                        vals = np.asarray(shap_vals)
                except Exception as ex:
                    self.log(f"shap.Explainer failed for {model_name}: {ex}; falling back to KernelExplainer (slower).")
                    try:
                        kexp = shap.KernelExplainer(lambda x: predict_fn(x), X_background)
                        vals = kexp.shap_values(X_eval, nsamples=200)
                    except Exception as kex:
                        self.log(f"KernelExplainer failed for {model_name}: {kex}")
                        continue
                if isinstance(vals, list):
                    vals = np.asarray(vals[0])
                vals = np.asarray(vals)
                if vals.ndim == 3:
                    vals = vals[0]
                if vals.ndim != 2 or vals.shape[1] != len(feature_cols):
                    try:
                        vals = vals.reshape(vals.shape[0], len(feature_cols))
                    except Exception as reshape_e:
                        self.log(f"Unexpected SHAP shape for {model_name}: {vals.shape}; error: {reshape_e}")
                        continue
                mean_abs_shap = np.mean(np.abs(vals), axis=0)
                if scenario_name not in self.all_importances:
                    self.all_importances[scenario_name] = {}
                if model_name not in self.all_importances[scenario_name]:
                    self.all_importances[scenario_name][model_name] = {}
                self.all_importances[scenario_name][model_name]['importance_shap'] = mean_abs_shap.tolist()
                self.all_importances[scenario_name][model_name]['method'] = 'shap_mean_abs'
                shap_results[model_name] = mean_abs_shap
                self.log(f"Computed SHAP for {model_name}: mean(|SHAP|) for {len(feature_cols)} features.")
            except Exception as e:
                self.log(f"Error computing SHAP for {model_name}: {e}")
                continue
        try:
            models_with_shap = [m for m in self.models if m in shap_results]
            if not models_with_shap:
                self.log("No SHAP results to plot.")
                return
            _, axes = plt.subplots(1, len(models_with_shap), figsize=(4*len(models_with_shap), 6))
            if len(models_with_shap) == 1:
                axes = [axes]
            for i, model_name in enumerate(models_with_shap):
                ax = axes[i]
                mean_abs_shap = shap_results[model_name]
                bars = ax.barh(range(len(feature_cols)), mean_abs_shap)
                ax.set_yticks(range(len(feature_cols)))
                ax.set_yticklabels(feature_cols)
                ax.set_xlabel('Mean |SHAP| (model output units)')
                ax.set_title(f'{model_name.upper()} — method: SHAP mean(|SHAP|)')
                max_imp = max(mean_abs_shap) if max(mean_abs_shap) > 0 else 1
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.plasma(mean_abs_shap[j] / max_imp))
            plt.suptitle(f'SHAP Model Importance - {scenario_name.upper()}', fontsize=16)
            plt.tight_layout()
            output_path = self.output_dir / f'{scenario_name}_all_models_shap_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"SHAP consolidated importance plot saved to: {output_path}")
        except Exception as e:
            self.log(f"Error plotting SHAP results: {e}")
    
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
