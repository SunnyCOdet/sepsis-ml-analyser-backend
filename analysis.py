import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer

def perform_analysis(filepath):
    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        # Try reading with different encoding if default fails
        df = pd.read_csv(filepath, encoding='latin1')

    # Column Mapping based on the provided CSV structure
    # "Lactate. 0hr", "Albumin 0 hr", "CRP 0hr", "NLR 0hr", "PCT 0hr", "APACHE II score", "Outcomes of the patient"
    
    # Normalize column names to handle potential whitespace issues
    df.columns = df.columns.str.strip()
    
    # Required columns mapping
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
    
    # Rename for easier access
    data.columns = col_map.keys()
    
    # Clean Outcome column
    # Assuming 'Expired' is positive class (1) and 'Relived' is negative class (0)
    # Check unique values to be sure
    # data['Outcome'] = data['Outcome'].apply(lambda x: 1 if str(x).lower().strip() == 'expired' else 0)
    # Better approach: map known values, drop unknown
    def clean_outcome(val):
        s = str(val).lower().strip()
        if 'expired' in s: return 1
        if 'relived' in s: return 0
        return np.nan

    data['Outcome'] = data['Outcome'].apply(clean_outcome)
    data = data.dropna(subset=['Outcome']) # Drop rows without valid outcome

    # Handle non-numeric data in feature columns (e.g. ">100", "<6")
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
    # Avoid division by zero
    data['LAR'] = data['Lactate'] / data['Albumin'].replace(0, np.nan)
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    # We need to impute LAR, CRP, NLR, PCT, APACHE
    # Note: LAR depends on Lactate and Albumin. If we impute Lactate/Albumin first, we can recalc LAR, 
    # or just impute LAR directly. User asked: "If data is not available take the mean value of the available data"
    # I will impute all features used in models.
    
    model_features = ['LAR', 'PCT', 'CRP', 'NLR', 'APACHE']
    data[model_features] = imputer.fit_transform(data[model_features])

    results = {}
    
    # Correlation Analysis
    # R value correlation coefficient for Lactate albumin ratio with procalcitonin with mortality
    # Correlation matrix
    corr_matrix = data[['LAR', 'PCT', 'Outcome']].corr()
    results['correlation'] = {
        'LAR_PCT': corr_matrix.loc['LAR', 'PCT'],
        'LAR_Outcome': corr_matrix.loc['LAR', 'Outcome'],
        'PCT_Outcome': corr_matrix.loc['PCT', 'Outcome']
    }

    # Models to build
    # 1. LAR
    # 2. PCT
    # 3. LAR + CRP + NLR
    # 4. LAR + CRP + NLR + APACHE 2 SCORE
    
    models_config = [
        {'name': 'LAR', 'features': ['LAR']},
        {'name': 'PCT', 'features': ['PCT']},
        {'name': 'LAR+CRP+NLR', 'features': ['LAR', 'CRP', 'NLR']},
        {'name': 'LAR+CRP+NLR+APACHE II', 'features': ['LAR', 'CRP', 'NLR', 'APACHE']}
    ]
    
    model_results = []
    
    y = data['Outcome']
    
    for config in models_config:
        X = data[config['features']]
        
        # Logistic Regression
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X, y)
        
        # Predictions
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        model_results.append({
            'name': config['name'],
            'metrics': {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'AUC': roc_auc
            },
            'roc_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        })
        
    results['models'] = model_results
    
    return results

def predict_patient(filepath, patient_data):
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
    
    # Imputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(data[model_features])
    y_train = data['Outcome']
    
    # Train Model (using the most comprehensive set)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Prepare Patient Data
    # Calculate LAR for patient
    p_lactate = float(patient_data.get('Lactate') or 0)
    p_albumin = float(patient_data.get('Albumin') or 0)
    p_lar = p_lactate / p_albumin if p_albumin != 0 else np.nan
    
    p_features = pd.DataFrame([{
        'LAR': p_lar,
        'PCT': float(patient_data.get('PCT') or np.nan),
        'CRP': float(patient_data.get('CRP') or np.nan),
        'NLR': float(patient_data.get('NLR') or np.nan),
        'APACHE': float(patient_data.get('APACHE') or np.nan)
    }])
    
    # Impute patient data using training imputer
    X_patient = imputer.transform(p_features)
    
    # Predict
    prob = clf.predict_proba(X_patient)[0][1]
    prediction = clf.predict(X_patient)[0]
    
    return {
        'probability_expired': prob,
        'prediction': 'Expired' if prediction == 1 else 'Relived',
        'calculated_features': {
            'LAR': p_lar if not np.isnan(p_lar) else None
        }
    }
