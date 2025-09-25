# model.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold

import warnings
warnings.filterwarnings('ignore')

# --- Import Models ---
try:
    import xgboost as xgb
except ImportError:
    pass
try:
    import lightgbm as lgb
except ImportError:
    pass
try:
    import catboost as cb
except ImportError:
    pass


class ScreenTimeModel:
    """
    A wrapper for training, tuning, and evaluating multi-label classification models.
    Uses MultiOutputClassifier for SHAP compatibility.
    """
    def __init__(self, model_type='xgboost', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.target_names = None
        # --- ADDED: Attributes to hold all three ensemble components ---
        self.xgb_component = None
        self.lgbm_component = None
        self.cat_component = None

        if model_type not in ['xgboost', 'lightgbm', 'ensemble', 'catboost']:
            raise ValueError("Unsupported model_type. Choose from: xgboost, lightgbm, ensemble, catboost")

    def train(self, X_train, y_train, target_names):
        """
        Builds a pipeline, defines hyperparameter search space, and trains the model
        using RandomizedSearchCV with K-Fold cross-validation.
        """
        print(f"🚀 Training and tuning {self.model_type} model with Pipeline and RandomizedSearchCV...")
        self.target_names = target_names

        total_neg = np.sum(y_train == 0)
        total_pos = np.sum(y_train == 1)
        scale_pos_weight_value = total_neg / total_pos
        print(f"⚖️ Calculated scale_pos_weight for imbalance: {scale_pos_weight_value:.2f}")

        xgb_base = xgb.XGBClassifier(objective='binary:logistic', random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        lgbm_base = lgb.LGBMClassifier(objective='binary', random_state=self.random_state, verbose=-1)
        catboost_base = cb.CatBoostClassifier(random_state=self.random_state, verbose=0)
        
        moc_xgb = MultiOutputClassifier(estimator=xgb_base)
        moc_lgbm = MultiOutputClassifier(estimator=lgbm_base)
        moc_catboost = MultiOutputClassifier(estimator=catboost_base)
        
        param_dist_xgb = {
            'estimator__scale_pos_weight': [scale_pos_weight_value, scale_pos_weight_value * 0.5, scale_pos_weight_value * 1.5],
            'estimator__n_estimators': [100, 200, 300],
            'estimator__learning_rate': [0.05, 0.1, 0.2],
            'estimator__max_depth': [3, 5, 7],
        }
        param_dist_lgbm = {
            'estimator__is_unbalance': [True],
            'estimator__n_estimators': [100, 200, 300],
            'estimator__learning_rate': [0.05, 0.1, 0.2],
            'estimator__num_leaves': [20, 31, 40],
        }
        param_dist_catboost = {
            'estimator__scale_pos_weight': [scale_pos_weight_value, scale_pos_weight_value * 0.5, scale_pos_weight_value * 1.5],
            'estimator__iterations': [200, 300, 500],
            'estimator__learning_rate': [0.05, 0.1],
            'estimator__depth': [4, 6, 8],
        }

        if self.model_type == 'xgboost':
            estimator = moc_xgb
            param_dist = param_dist_xgb
        elif self.model_type == 'lightgbm':
            estimator = moc_lgbm
            param_dist = param_dist_lgbm
        elif self.model_type == 'catboost':
            estimator = moc_catboost
            param_dist = param_dist_catboost
        elif self.model_type == 'ensemble':
            print("Ensemble selected. Training tuned XGBoost, LightGBM, and CatBoost models first (this will take longer)...")
            self.xgb_component = self._tune_single_model(X_train, y_train, moc_xgb, param_dist_xgb, return_pipeline=True)
            self.lgbm_component = self._tune_single_model(X_train, y_train, moc_lgbm, param_dist_lgbm, return_pipeline=True)
            # --- ADDED: Tune CatBoost as well for the ensemble ---
            self.cat_component = self._tune_single_model(X_train, y_train, moc_catboost, param_dist_catboost, return_pipeline=True)

            print("🤝 Assembling the Soft-Voting Ensemble...")
            xgb_tuned_moc = self.xgb_component.named_steps['model']
            lgbm_tuned_moc = self.lgbm_component.named_steps['model']
            cat_tuned_moc = self.cat_component.named_steps['model']

            # --- CHANGED: Add CatBoost to the VotingClassifier ---
            ensemble_model = VotingClassifier(
                estimators=[
                    ('xgb', xgb_tuned_moc.estimator), 
                    ('lgbm', lgbm_tuned_moc.estimator),
                    ('cat', cat_tuned_moc.estimator)
                ],
                voting='soft'
            )
            final_model = MultiOutputClassifier(estimator=ensemble_model)
            pipeline = Pipeline([('scaler', self.xgb_component.named_steps['scaler']), ('model', final_model)])
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            print("✅ Ensemble training complete.")
            return

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions={'model__' + k: v for k, v in param_dist.items()},
            n_iter=10,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        random_search.fit(X_train, y_train)
        
        print("\n" + "="*60)
        print(f"🏆 Best F1-Macro Score from CV: {random_search.best_score_:.4f}")
        print("🔍 Best Hyperparameters Found:")
        print(random_search.best_params_)
        print("="*60 + "\n")

        self.model = random_search.best_estimator_
        print("✅ Training and tuning complete.")

    def _tune_single_model(self, X_train, y_train, estimator, param_dist, return_pipeline=False):
        """Helper function to tune a single model."""
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(pipeline, {'model__' + k: v for k, v in param_dist.items()}, n_iter=10, cv=cv, scoring='f1_macro', n_jobs=-1, random_state=self.random_state, verbose=0)
        search.fit(X_train, y_train)
        print(f"Tuned {estimator.estimator.__class__.__name__} with score: {search.best_score_:.4f}")
        return search.best_estimator_ if return_pipeline else search.best_estimator_.named_steps['model']

    def evaluate(self, X_test, y_test):
        """Evaluates the final tuned model on the hold-out test set."""
        print("🔍 Evaluating final model on the test set...")
        y_pred = self.model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        hamming = hamming_loss(y_test, y_pred)
        print("\n" + "="*60)
        print("Final Multi-Label Classification Report (Test Set)")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=self.target_names, zero_division=0))
        print(f"Hamming Loss: {hamming:.4f} (Lower is better)")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print("="*60)
        return {'hamming_loss': hamming, 'f1_macro': f1_macro}

    def save_model(self, output_dir='models'):
        """Saves the trained pipeline and component models to files."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{self.model_type}_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"✅ Model pipeline saved to {model_path}")
        
        if self.model_type == 'ensemble':
            xgb_path = os.path.join(output_dir, 'ensemble_xgb_component.joblib')
            lgbm_path = os.path.join(output_dir, 'ensemble_lgbm_component.joblib')
            # --- ADDED: Save the CatBoost component as well ---
            cat_path = os.path.join(output_dir, 'ensemble_cat_component.joblib')
            joblib.dump(self.xgb_component, xgb_path)
            joblib.dump(self.lgbm_component, lgbm_path)
            joblib.dump(self.cat_component, cat_path)
            print(f"✅ Ensemble component models saved.")
            
        return model_path