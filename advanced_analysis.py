import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import statsmodels.api as sm


BIOMARKERS = ["Lactate", "Albumin", "LAR", "PCT", "CRP", "NLR", "APACHE"]


def _normalize_column(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _map_columns(df):
    norm_cols = {col: _normalize_column(col) for col in df.columns}

    candidates = {
        "Lactate": ["lactate0hr", "lactate0", "lactate"],
        "Albumin": ["albumin0hr", "albumin0", "albumin"],
        "CRP": ["crp0hr", "crp", "creactiveprotein"],
        "NLR": ["nlr0hr", "nlr", "neutrophillymphocyteratio"],
        "PCT": ["pct0hr", "pct", "procalcitonin"],
        "APACHE": [
            "apacheiiscore",
            "apacheii",
            "apachescore",
            "apache2score",
            "apache2",
        ],
        "Outcome": [
            "outcomesofthepatient",
            "outcome",
            "mortality",
            "survivalstatus",
            "death",
        ],
        "Culture": ["culturesensitivity", "culture", "cultureresult", "bloodculture"],
    }

    mapped = {}
    for target, opts in candidates.items():
        found = None
        for opt in opts:
            for col, norm in norm_cols.items():
                if norm == opt or opt in norm:
                    found = col
                    break
            if found:
                break
        if found:
            mapped[target] = found

    required = ["Lactate", "Albumin", "CRP", "NLR", "PCT", "Outcome"]
    missing = [key for key in required if key not in mapped]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return mapped


def _clean_numeric(val):
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if val is None:
        return np.nan
    s = str(val).strip().lower()
    if s in {"", "nan", "na", "n/a", "none"}:
        return np.nan
    s = s.replace(",", "").replace(">", "").replace("<", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def _clean_outcome(val):
    if isinstance(val, (int, float, np.number)):
        if pd.isna(val):
            return np.nan
        if int(val) in (0, 1):
            return int(val)
    s = str(val).lower().strip()
    if any(
        k in s
        for k in [
            "expired",
            "dead",
            "death",
            "non-survivor",
            "nonsurvivor",
            "non survivor",
        ]
    ):
        return 1
    if any(k in s for k in ["survivor", "survived", "alive", "relived", "discharged"]):
        return 0
    return np.nan


def _clean_culture(val):
    if val is None:
        return np.nan
    s = str(val).lower().strip()
    if s == "" or s in {"nan", "na", "n/a", "none"}:
        return np.nan
    if any(k in s for k in ["pos", "positive", "growth", "detected"]):
        return "Positive"
    if any(k in s for k in ["neg", "negative", "no growth", "not detected"]):
        return "Negative"
    return np.nan


def _shapiro_p(values):
    values = values.dropna()
    if len(values) < 3:
        return np.nan
    if len(values) > 5000:
        values = values.sample(5000, random_state=42)
    try:
        return float(stats.shapiro(values).pvalue)
    except Exception:
        return np.nan


def _iqr(values):
    return float(values.quantile(0.75) - values.quantile(0.25))


def _mode(values):
    modes = values.mode(dropna=True)
    if len(modes) == 0:
        return np.nan
    return float(modes.iloc[0])


def _effect_size_cohen_d(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / pooled)


def _effect_size_rank_biserial(u, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1 - (2 * u) / (n1 * n2))


def _prepare_data(filepath):
    if filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        try:
            df = pd.read_csv(filepath, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(
                filepath,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                encoding="latin1",
            )

    df.columns = df.columns.str.strip()
    col_map = _map_columns(df)

    base_keys = ["Lactate", "Albumin", "CRP", "NLR", "PCT", "Outcome"]
    cols = [col_map[key] for key in base_keys]
    data = df[cols].copy()
    data.columns = base_keys
    if "APACHE" in col_map:
        data["APACHE"] = df[col_map["APACHE"]]
    else:
        data["APACHE"] = np.nan

    if "Culture" in col_map:
        data["Culture"] = df[col_map["Culture"]].apply(_clean_culture)

    for col in ["Lactate", "Albumin", "CRP", "NLR", "PCT", "APACHE"]:
        data[col] = data[col].apply(_clean_numeric)

    data["Outcome"] = data["Outcome"].apply(_clean_outcome)
    data = data.dropna(subset=["Outcome"]).copy()

    data["LAR"] = data["Lactate"] / data["Albumin"].replace(0, np.nan)
    return data


def _impute_data(data):
    imputation = {}
    distribution = {}

    for col in BIOMARKERS:
        series = data[col]
        if series.dropna().empty:
            data[col] = series.fillna(0)
            imputation[col] = {"method": "missing_default", "value": 0.0}
            distribution[col] = {"p_value": np.nan, "classification": "Missing"}
            continue
        unique_count = series.dropna().nunique()
        p_value = _shapiro_p(series)
        is_categorical = unique_count <= 5
        if is_categorical:
            method = "mode"
            fill_value = _mode(series)
            classification = "Categorical"
        else:
            if not np.isnan(p_value) and p_value > 0.05:
                method = "mean"
                fill_value = float(series.mean())
                classification = "Normal"
            else:
                method = "median"
                fill_value = float(series.median())
                classification = "Non-normal"

        data[col] = series.fillna(fill_value)
        imputation[col] = {"method": method, "value": fill_value}
        distribution[col] = {"p_value": p_value, "classification": classification}

    return data, imputation, distribution


def _descriptive_stats(data):
    results = {"overall": {}, "survivors": {}, "non_survivors": {}}
    groups = {
        "overall": data,
        "survivors": data[data["Outcome"] == 0],
        "non_survivors": data[data["Outcome"] == 1],
    }

    for group_name, group_df in groups.items():
        for col in BIOMARKERS:
            series = group_df[col].dropna()
            if series.empty:
                stats_result = {
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "mode": np.nan,
                    "std": np.nan,
                    "iqr": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            else:
                stats_result = {
                    "n": int(series.count()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "mode": _mode(series),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "iqr": _iqr(series),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            results[group_name][col] = stats_result

    return results


def _group_comparison(data, distribution):
    results = {}
    survivors = data[data["Outcome"] == 0]
    non_survivors = data[data["Outcome"] == 1]

    for col in BIOMARKERS:
        x1 = survivors[col].dropna().values
        x2 = non_survivors[col].dropna().values
        if len(x1) == 0 or len(x2) == 0:
            results[col] = {
                "test": None,
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
            }
            continue

        is_normal = distribution[col]["classification"] == "Normal"
        if is_normal:
            stat, p_value = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
            effect = _effect_size_cohen_d(x1, x2)
            test_name = "Independent t-test"
        else:
            stat, p_value = stats.mannwhitneyu(x1, x2, alternative="two-sided")
            effect = _effect_size_rank_biserial(stat, len(x1), len(x2))
            test_name = "Mann-Whitney U"

        results[col] = {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": float(effect) if effect is not None else np.nan,
        }

    return results


def _correlation_analysis(data):
    results = {}
    for col in BIOMARKERS:
        r, p = stats.spearmanr(data[col], data["Outcome"], nan_policy="omit")
        results[col] = {"r": float(r), "p_value": float(p)}

    corr_matrix = data[BIOMARKERS].corr(method="spearman")
    matrix = {
        "labels": BIOMARKERS,
        "values": corr_matrix.fillna(0).values.tolist(),
    }

    return results, matrix


def _logit_univariate(data):
    results = []
    y = data["Outcome"]

    for col in BIOMARKERS:
        X = sm.add_constant(data[[col]])
        try:
            model = sm.Logit(y, X).fit(disp=0)
            beta = model.params[col]
            se = model.bse[col]
            p_value = model.pvalues[col]
            or_value = np.exp(beta)
            ci_low = np.exp(beta - 1.96 * se)
            ci_high = np.exp(beta + 1.96 * se)
            results.append(
                {
                    "biomarker": col,
                    "beta": float(beta),
                    "odds_ratio": float(or_value),
                    "ci_95": [float(ci_low), float(ci_high)],
                    "p_value": float(p_value),
                }
            )
        except Exception:
            results.append(
                {
                    "biomarker": col,
                    "beta": np.nan,
                    "odds_ratio": np.nan,
                    "ci_95": [np.nan, np.nan],
                    "p_value": np.nan,
                }
            )

    return results


def _vif_scores(df):
    scores = {}
    if df.shape[1] < 2:
        return {df.columns[0]: 1.0}

    for col in df.columns:
        X = df.drop(columns=[col])
        y = df[col]
        try:
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            if r2 >= 1:
                scores[col] = np.inf
            else:
                scores[col] = float(1 / (1 - r2))
        except Exception:
            scores[col] = np.nan

    return scores


def _logit_multivariable(data, models_config):
    results = []
    y = data["Outcome"]

    for config in models_config:
        features = config["features"]
        X = sm.add_constant(data[features])
        try:
            model = sm.Logit(y, X).fit(disp=0)
            coefs = model.params.drop("const")
            ses = model.bse.drop("const")
            pvals = model.pvalues.drop("const")
            odds = np.exp(coefs)
            ci_low = np.exp(coefs - 1.96 * ses)
            ci_high = np.exp(coefs + 1.96 * ses)
            vif = _vif_scores(data[features])
            results.append(
                {
                    "name": config["name"],
                    "features": features,
                    "coefficients": {k: float(v) for k, v in coefs.items()},
                    "odds_ratios": {k: float(v) for k, v in odds.items()},
                    "ci_95": {
                        k: [float(ci_low[k]), float(ci_high[k])] for k in coefs.index
                    },
                    "p_values": {k: float(pvals[k]) for k in coefs.index},
                    "aic": float(model.aic),
                    "vif": {k: float(vif[k]) for k in vif},
                }
            )
        except Exception:
            results.append(
                {
                    "name": config["name"],
                    "features": features,
                    "coefficients": {},
                    "odds_ratios": {},
                    "ci_95": {},
                    "p_values": {},
                    "aic": np.nan,
                    "vif": {},
                }
            )

    return results


def _roc_and_metrics(data, models_config):
    results = []
    y = data["Outcome"]

    for config in models_config:
        features = config["features"]
        X = data[features]
        try:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
        except Exception:
            probs = np.full(len(y), np.nan)

        if np.isnan(probs).any():
            fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
            roc_auc = np.nan
            cutoff = np.nan
            sensitivity = specificity = accuracy = ppv = npv = f1 = np.nan
        else:
            fpr, tpr, thresholds = roc_curve(y, probs)
            roc_auc = auc(fpr, tpr)
            youden = tpr - fpr
            idx = int(np.argmax(youden)) if len(youden) else 0
            cutoff = float(thresholds[idx]) if len(thresholds) else np.nan

            preds = (probs >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
            ppv = tp / (tp + fp) if (tp + fp) else 0
            npv = tn / (tn + fn) if (tn + fn) else 0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0

        results.append(
            {
                "name": config["name"],
                "features": features,
                "auc": float(roc_auc),
                "optimal_cutoff": cutoff,
                "roc_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "diagnostic_metrics": {
                    "sensitivity": float(sensitivity),
                    "specificity": float(specificity),
                    "accuracy": float(accuracy),
                    "ppv": float(ppv),
                    "npv": float(npv),
                    "f1": float(f1),
                },
                "probabilities": probs.tolist(),
            }
        )

    return results


def _hosmer_lemeshow(y, probs, groups=10):
    data = pd.DataFrame({"y": y, "probs": probs})
    try:
        data["bin"] = pd.qcut(data["probs"], q=groups, duplicates="drop")
    except Exception:
        return {"statistic": np.nan, "p_value": np.nan}

    grouped = data.groupby("bin", observed=True)
    obs = grouped["y"].sum()
    exp = grouped["probs"].sum()
    n = grouped.size()
    denom = exp * (1 - exp / n)
    denom = denom.replace(0, np.nan)
    hl = (((obs - exp) ** 2) / denom).sum()
    dof = max(len(obs) - 2, 1)
    p_value = 1 - stats.chi2.cdf(hl, dof)
    return {"statistic": float(hl), "p_value": float(p_value)}


def _calibration_curve(y, probs, groups=10):
    data = pd.DataFrame({"y": y, "probs": probs})
    data["bin"] = pd.qcut(data["probs"], q=groups, duplicates="drop")
    grouped = data.groupby("bin", observed=True)
    mean_pred = grouped["probs"].mean().tolist()
    mean_obs = grouped["y"].mean().tolist()
    return {"predicted": mean_pred, "observed": mean_obs}


def _internal_validation(data, models_config, bootstrap_iters=1000):
    y = data["Outcome"].values
    validation = {"kfold": {}, "bootstrap": {}}

    for config in models_config:
        X = data[config["features"]].values

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        aucs = []
        sens = []
        specs = []
        accs = []

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], probs))
            preds = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[test_idx], preds).ravel()
            sens.append(tp / (tp + fn) if (tp + fn) else 0)
            specs.append(tn / (tn + fp) if (tn + fp) else 0)
            accs.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0)

        validation["kfold"][config["name"]] = {
            "auc_mean": float(np.mean(aucs)),
            "sensitivity_mean": float(np.mean(sens)),
            "specificity_mean": float(np.mean(specs)),
            "accuracy_mean": float(np.mean(accs)),
        }

        rng = np.random.default_rng(42)
        auc_boot = []
        optimism = []
        clf_full = LogisticRegression(max_iter=1000)
        clf_full.fit(X, y)
        full_auc = roc_auc_score(y, clf_full.predict_proba(X)[:, 1])

        for _ in range(bootstrap_iters):
            idx = rng.integers(0, len(y), len(y))
            X_b = X[idx]
            y_b = y[idx]
            try:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_b, y_b)
                auc_b = roc_auc_score(y_b, clf.predict_proba(X_b)[:, 1])
                auc_t = roc_auc_score(y, clf.predict_proba(X)[:, 1])
                auc_boot.append(auc_b)
                optimism.append(auc_b - auc_t)
            except Exception:
                continue

        auc_boot = np.array(auc_boot)
        optimism = np.array(optimism)
        if len(auc_boot) == 0:
            validation["bootstrap"][config["name"]] = {
                "mean_auc": np.nan,
                "standard_error": np.nan,
                "ci_95": [np.nan, np.nan],
                "optimism_corrected_auc": np.nan,
            }
        else:
            mean_auc = float(np.mean(auc_boot))
            se = float(np.std(auc_boot, ddof=1))
            ci_low, ci_high = np.percentile(auc_boot, [2.5, 97.5])
            optimism_mean = float(np.mean(optimism))
            validation["bootstrap"][config["name"]] = {
                "mean_auc": mean_auc,
                "standard_error": se,
                "ci_95": [float(ci_low), float(ci_high)],
                "optimism_corrected_auc": float(full_auc - optimism_mean),
            }

    return validation


def _culture_sensitivity(data):
    if "Culture" not in data.columns:
        return {"available": False}

    subset = data.dropna(subset=["Culture"])
    pos = subset[subset["Culture"] == "Positive"]["LAR"]
    neg = subset[subset["Culture"] == "Negative"]["LAR"]

    if len(pos) == 0 or len(neg) == 0:
        return {"available": True, "error": "Insufficient culture groups"}

    stat, p_value = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "available": True,
        "test": "Mann-Whitney U",
        "median": {
            "culture_positive": float(pos.median()),
            "culture_negative": float(neg.median()),
        },
        "iqr": {
            "culture_positive": _iqr(pos),
            "culture_negative": _iqr(neg),
        },
        "p_value": float(p_value),
        "n": {"culture_positive": int(len(pos)), "culture_negative": int(len(neg))},
    }


def perform_analysis(filepath):
    data = _prepare_data(filepath)
    data, imputation, distribution = _impute_data(data)

    descriptive = _descriptive_stats(data)
    comparisons = _group_comparison(data, distribution)
    correlations, corr_matrix = _correlation_analysis(data)
    univariate = _logit_univariate(data)

    models_config = [
        {"name": "LAR", "features": ["LAR"]},
        {"name": "LAR+CRP", "features": ["LAR", "CRP"]},
        {"name": "LAR+CRP+NLR", "features": ["LAR", "CRP", "NLR"]},
        {"name": "LAR+CRP+NLR+APACHE II", "features": ["LAR", "CRP", "NLR", "APACHE"]},
    ]

    multivariable = _logit_multivariable(data, models_config)
    roc_models = _roc_and_metrics(data, models_config)

    biomarkers_config = [{"name": b, "features": [b]} for b in BIOMARKERS]
    roc_biomarkers = _roc_and_metrics(data, biomarkers_config)

    calibration = []
    for model in roc_models:
        probs = np.array(model["probabilities"])
        y = data["Outcome"].values
        calibration.append(
            {
                "name": model["name"],
                "hosmer_lemeshow": _hosmer_lemeshow(y, probs),
                "calibration_curve": _calibration_curve(y, probs),
            }
        )

    model_comparison = {
        "best_model": max(roc_models, key=lambda x: x["auc"])["name"],
        "auc_by_model": {m["name"]: m["auc"] for m in roc_models},
        "aic_by_model": {
            m["name"]: next(
                (mm["aic"] for mm in multivariable if mm["name"] == m["name"]), np.nan
            )
            for m in roc_models
        },
    }

    internal_validation = _internal_validation(data, models_config)
    culture = _culture_sensitivity(data)

    results = {
        "summary": {
            "n_total": int(len(data)),
            "n_survivors": int((data["Outcome"] == 0).sum()),
            "n_non_survivors": int((data["Outcome"] == 1).sum()),
        },
        "imputation": imputation,
        "distribution": distribution,
        "descriptive_stats": descriptive,
        "group_comparison": comparisons,
        "correlation": correlations,
        "correlation_matrix": corr_matrix,
        "univariate_logistic": univariate,
        "multivariable_logistic": multivariable,
        "roc_biomarkers": roc_biomarkers,
        "roc_models": roc_models,
        "calibration": calibration,
        "diagnostic_metrics": {m["name"]: m["diagnostic_metrics"] for m in roc_models},
        "model_comparison": model_comparison,
        "internal_validation": internal_validation,
        "culture_sensitivity": culture,
    }

    return results


def predict_patient(filepath, patient_data):
    data = _prepare_data(filepath)
    data, _, _ = _impute_data(data)

    model_features = ["LAR", "PCT", "CRP", "NLR", "APACHE"]
    X_train = data[model_features]
    y_train = data["Outcome"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    except Exception:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, max_depth=10
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=5
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            if len(y_train[y_train == 1]) > 0
            else 1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, random_state=42, max_depth=5, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100, random_state=42, depth=5, verbose=0
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
    }

    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train_balanced, y_train_balanced)
            trained_models[name] = model
        except Exception:
            continue

    p_lactate = _clean_numeric(patient_data.get("Lactate"))
    p_albumin = _clean_numeric(patient_data.get("Albumin"))

    if pd.notna(p_lactate) and pd.notna(p_albumin) and p_albumin != 0:
        p_lar = p_lactate / p_albumin
    else:
        p_lar = np.nan

    p_features = pd.DataFrame(
        [
            {
                "LAR": p_lar,
                "PCT": _clean_numeric(patient_data.get("PCT")),
                "CRP": _clean_numeric(patient_data.get("CRP")),
                "NLR": _clean_numeric(patient_data.get("NLR")),
                "APACHE": _clean_numeric(patient_data.get("APACHE")),
            }
        ]
    )

    p_features = p_features.fillna(X_train.mean())
    X_patient_scaled = scaler.transform(p_features)

    predictions = []
    probabilities = []
    model_details = []

    for name, model in trained_models.items():
        try:
            prob = model.predict_proba(X_patient_scaled)[0][1]
            pred = model.predict(X_patient_scaled)[0]
            predictions.append(pred)
            probabilities.append(prob)
            model_details.append(
                {
                    "model": name,
                    "probability": float(prob),
                    "prediction": "Expired" if pred == 1 else "Survived",
                }
            )
        except Exception:
            continue

    ensemble_prob = float(np.mean(probabilities)) if probabilities else 0.0
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

    prob_std = float(np.std(probabilities)) if probabilities else 0.0
    confidence = max(0.0, min(100.0, (1 - prob_std) * 100))

    if ensemble_prob < 0.3:
        risk_level = "LOW RISK"
        risk_color = "green"
        recommendation = (
            "Continue standard monitoring. Patient shows favorable indicators."
        )
    elif ensemble_prob < 0.6:
        risk_level = "MODERATE RISK"
        risk_color = "orange"
        recommendation = (
            "Enhanced monitoring recommended. Consider preventive interventions."
        )
    else:
        risk_level = "HIGH RISK"
        risk_color = "red"
        recommendation = "URGENT: Immediate intensive care recommended. High mortality risk detected."

    feature_importance = {}
    if "Random Forest" in trained_models:
        importances = trained_models["Random Forest"].feature_importances_
        for i, feature in enumerate(model_features):
            feature_importance[feature] = float(importances[i])

    return {
        "ensemble_probability_expired": float(ensemble_prob),
        "ensemble_prediction": "Expired" if ensemble_pred == 1 else "Survived",
        "confidence_score": float(confidence),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "clinical_recommendation": recommendation,
        "model_agreement": f"{sum(predictions)}/{len(predictions)} models predict mortality"
        if predictions
        else "0/0 models predict mortality",
        "individual_models": model_details,
        "calculated_features": {
            "LAR": float(p_lar) if not np.isnan(p_lar) else None,
            "Lactate": float(p_lactate) if not np.isnan(p_lactate) else None,
            "Albumin": float(p_albumin) if not np.isnan(p_albumin) else None,
            "PCT": _clean_numeric(patient_data.get("PCT"))
            if patient_data.get("PCT")
            else None,
            "CRP": _clean_numeric(patient_data.get("CRP"))
            if patient_data.get("CRP")
            else None,
            "NLR": _clean_numeric(patient_data.get("NLR"))
            if patient_data.get("NLR")
            else None,
            "APACHE": _clean_numeric(patient_data.get("APACHE"))
            if patient_data.get("APACHE")
            else None,
        },
        "feature_importance": feature_importance,
        "probability_survived": float(1 - ensemble_prob),
        "num_models_used": len(trained_models),
    }
