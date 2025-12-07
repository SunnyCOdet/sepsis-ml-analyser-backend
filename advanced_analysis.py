import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def perform_analysis(filepath):
    """
    Comprehensive analysis with multiple advanced ML models
    """
    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        df = pd.read_csv(filepath, encoding='latin1')

    # Normalize column names
    df.columns = df.columns.str.strip()
    
    # Column mapping
    col_map = {
        'Lactate': 'Lactate. 0hr',
        'Albumin': 'Albumin 0 hr',
        'CRP': 'CRP 0hr',
        'NLR': 'NLR 0hr',
        'PCT': 'PCT 0hr',
        'APACHE': 'APACHE II score',
        'Outcome': 'Outcomes of the patient'
    }
    
    # Verify columns exist
    missing_cols = [c for c in col_map.values() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Extract relevant data
    data = df[list(col_map.values())].copy()
    data.columns = col_map.keys()
    
    # Clean Outcome column
    def clean_outcome(val):
        s = str(val).lower().strip()
        if 'expired' in s: return 1
        if 'relived' in s: return 0
        return np.nan

    data['Outcome'] = data['Outcome'].apply(clean_outcome)
    data = data.dropna(subset=['Outcome'])

    # Handle non-numeric data
    def clean_numeric(val):
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        s = s.replace('>', '').replace('<', '').replace(',', '')
        try:
            return float(s)
        except:
            return np.nan

    feature_cols = ['Lactate', 'Albumin', 'CRP', 'NLR', 'PCT', 'APACHE']
    for col in feature_cols:
        data[col] = data[col].apply(clean_numeric)

    # Calculate LAR
    data['LAR'] = data['Lactate'] / data['Albumin'].replace(0, np.nan)
    
    # Use KNN Imputer for better handling of missing values
    model_features = ['LAR', 'PCT', 'CRP', 'NLR', 'APACHE']
    
    # Try KNN imputation first, fall back to mean if not enough data
    try:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        data[model_features] = knn_imputer.fit_transform(data[model_features])
    except:
        # Fallback to mean imputation
        mean_imputer = SimpleImputer(strategy='mean')
        data[model_features] = mean_imputer.fit_transform(data[model_features])

    results = {}
    
    # Correlation Analysis
    corr_matrix = data[['LAR', 'PCT', 'Outcome']].corr()
    results['correlation'] = {
        'LAR_PCT': corr_matrix.loc['LAR', 'PCT'],
        'LAR_Outcome': corr_matrix.loc['LAR', 'Outcome'],
        'PCT_Outcome': corr_matrix.loc['PCT', 'Outcome']
    }

    # Model configurations
    models_config = [
        {'name': 'LAR', 'features': ['LAR']},
        {'name': 'PCT', 'features': ['PCT']},
        {'name': 'LAR+CRP+NLR', 'features': ['LAR', 'CRP', 'NLR']},
        {'name': 'LAR+CRP+NLR+APACHE II', 'features': ['LAR', 'CRP', 'NLR', 'APACHE']},
        {'name': 'All Features (Comprehensive)', 'features': model_features}
    ]
    
    model_results = []
    y = data['Outcome']
    
    for config in models_config:
        X = data[config['features']]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply SMOTE if class imbalance exists
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        except:
            X_resampled, y_resampled = X_scaled, y
        
        # Logistic Regression (baseline)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_resampled, y_resampled)
        
        # Predictions
        y_pred = clf.predict(X_scaled)
        y_prob = clf.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(X_scaled)), scoring='roc_auc')
        
        model_results.append({
            'name': config['name'],
            'metrics': {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'AUC': roc_auc,
                'Accuracy': accuracy,
                'CV_AUC_Mean': cv_scores.mean(),
                'CV_AUC_Std': cv_scores.std()
            },
            'roc_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        })
        
    results['models'] = model_results
    return results


def predict_patient(filepath, patient_data):
    """
    Advanced ensemble prediction with multiple models and confidence scoring
    """
    # Load and prepare training data
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath, encoding='latin1')
        
    df.columns = df.columns.str.strip()
    col_map = {
        'Lactate': 'Lactate. 0hr',
        'Albumin': 'Albumin 0 hr',
        'CRP': 'CRP 0hr',
        'NLR': 'NLR 0hr',
        'PCT': 'PCT 0hr',
        'APACHE': 'APACHE II score',
        'Outcome': 'Outcomes of the patient'
    }
    
    # Clean training data
    data = df[list(col_map.values())].copy()
    data.columns = col_map.keys()
    
    def clean_outcome(val):
        s = str(val).lower().strip()
        if 'expired' in s: return 1
        if 'relived' in s: return 0
        return np.nan

    data['Outcome'] = data['Outcome'].apply(clean_outcome)
    data = data.dropna(subset=['Outcome'])

    def clean_numeric(val):
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        s = s.replace('>', '').replace('<', '').replace(',', '')
        try:
            return float(s)
        except:
            return np.nan

    feature_cols = ['Lactate', 'Albumin', 'CRP', 'NLR', 'PCT', 'APACHE']
    for col in feature_cols:
        data[col] = data[col].apply(clean_numeric)

    data['LAR'] = data['Lactate'] / data['Albumin'].replace(0, np.nan)
    
    # Prepare features
    model_features = ['LAR', 'PCT', 'CRP', 'NLR', 'APACHE']
    
    # KNN Imputer for training data
    try:
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_train = imputer.fit_transform(data[model_features])
    except:
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(data[model_features])
    
    y_train = data['Outcome']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply SMOTE for balanced training
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    except:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Train multiple models for ensemble
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, max_depth=5, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, depth=5, verbose=0),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train_balanced, y_train_balanced)
            trained_models[name] = model
        except Exception as e:
            print(f"Warning: {name} failed to train: {e}")
    
    # Prepare Patient Data
    p_lactate = float(patient_data.get('Lactate') or np.nan)
    p_albumin = float(patient_data.get('Albumin') or np.nan)
    
    # Calculate LAR with proper null handling
    if pd.notna(p_lactate) and pd.notna(p_albumin) and p_albumin != 0:
        p_lar = p_lactate / p_albumin
    else:
        p_lar = np.nan
    
    p_features = pd.DataFrame([{
        'LAR': p_lar,
        'PCT': float(patient_data.get('PCT') or np.nan),
        'CRP': float(patient_data.get('CRP') or np.nan),
        'NLR': float(patient_data.get('NLR') or np.nan),
        'APACHE': float(patient_data.get('APACHE') or np.nan)
    }])
    
    # Impute patient data using training imputer
    X_patient = imputer.transform(p_features)
    X_patient_scaled = scaler.transform(X_patient)
    
    # Get predictions from all models
    predictions = []
    probabilities = []
    model_details = []
    
    for name, model in trained_models.items():
        try:
            prob = model.predict_proba(X_patient_scaled)[0][1]
            pred = model.predict(X_patient_scaled)[0]
            predictions.append(pred)
            probabilities.append(prob)
            model_details.append({
                'model': name,
                'probability': float(prob),
                'prediction': 'Expired' if pred == 1 else 'Survived'
            })
        except Exception as e:
            print(f"Warning: {name} prediction failed: {e}")
    
    # Ensemble prediction (weighted voting)
    ensemble_prob = np.mean(probabilities)
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
    
    # Calculate confidence (inverse of standard deviation)
    prob_std = np.std(probabilities)
    confidence = max(0, min(100, (1 - prob_std) * 100))
    
    # Risk stratification
    if ensemble_prob < 0.3:
        risk_level = "LOW RISK"
        risk_color = "green"
        recommendation = "Continue standard monitoring. Patient shows favorable indicators."
    elif ensemble_prob < 0.6:
        risk_level = "MODERATE RISK"
        risk_color = "orange"
        recommendation = "Enhanced monitoring recommended. Consider preventive interventions."
    else:
        risk_level = "HIGH RISK"
        risk_color = "red"
        recommendation = "URGENT: Immediate intensive care recommended. High mortality risk detected."
    
    # Calculate feature importance (using Random Forest if available)
    feature_importance = {}
    if 'Random Forest' in trained_models:
        rf_model = trained_models['Random Forest']
        importances = rf_model.feature_importances_
        for i, feature in enumerate(model_features):
            feature_importance[feature] = float(importances[i])
    
    return {
        'ensemble_probability_expired': float(ensemble_prob),
        'ensemble_prediction': 'Expired' if ensemble_pred == 1 else 'Survived',
        'confidence_score': float(confidence),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'clinical_recommendation': recommendation,
        'model_agreement': f"{sum(predictions)}/{len(predictions)} models predict mortality",
        'individual_models': model_details,
        'calculated_features': {
            'LAR': float(p_lar) if not np.isnan(p_lar) else None,
            'Lactate': float(p_lactate) if not np.isnan(p_lactate) else None,
            'Albumin': float(p_albumin) if not np.isnan(p_albumin) else None,
            'PCT': float(patient_data.get('PCT')) if patient_data.get('PCT') else None,
            'CRP': float(patient_data.get('CRP')) if patient_data.get('CRP') else None,
            'NLR': float(patient_data.get('NLR')) if patient_data.get('NLR') else None,
            'APACHE': float(patient_data.get('APACHE')) if patient_data.get('APACHE') else None
        },
        'feature_importance': feature_importance,
        'probability_survived': float(1 - ensemble_prob),
        'num_models_used': len(trained_models)
    }
