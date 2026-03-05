"""
Assignment 5: Scikit Learn - Classification
Breast Cancer dataset (built into scikit-learn)

Goal:
- Train 3 different classification models
- Tune each model using a simple validation split
- Evaluate final chosen models on a held-out test set
- Pick the best modeel based on test metrics

Models:
1) Logistic Regression (scaled)
2) K-Nearest Neighbors (scaled)
3) Decision Tree (no scaling needed)
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

#---------------------------
#Settings
#---------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20                  #Final hold-out test set
VAL_SIZE = 0.20                   #Validation split from training set

#Hyperparameters to try (kept small and reasonable)
LR_C_VALUES = [0.1, 1.0, 5.0, 10.0]
KNN_K_VALUES = [3, 5, 7, 9, 11]
DT_MAX_DEPTH_VALUES = [None, 3, 5, 7, 9]
DT_MIN_SAMPLES_SPLIT_VALUES = [2, 5, 10]

def safe_roc_auc(y_true, y_proba):
    """ROC AUC requires probability scores and both classes present"""
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return float("nan")
    
def evaluate_with_proba(model, X, y):
    """
    Returns (f1, roc_auc) on given data.
    Assumes model is already fit and supports predict_proba.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, zero_division=0)
    auc = safe_roc_auc(y, y_proba)
    return f1, auc

def pick_best_by_validation(candidates):
    """
    candidates: list of dicts with keys:
     - name
     - model
     - val_f1
     - val_auc
    
    Chooses best primarily by val_f1, secondarily by val_auc
    """
    best = candidates[0]
    for c in candidates[1:]:
        if c["val_f1"] > best["val_f1"]:
            best = c
        elif c["val_f1"] == best["val_f1"]:
            #tie-breaker: ROC AUC (ignore nan by treating as -1)
            c_auc = c["val_auc"] if c["val_auc"] == c["val_auc"] else -1
            b_auc = best["val_auc"] if best["val_auc"] == best["val_auc"] else -1
            if c_auc > b_auc:
                best = c
    return best

def print_final_results(model_name, model, X_test, y_test):
    """Print test metrics and supports"""
    y_pred = model.predict(X_test)

    #Probabilities for ROC AUC if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = safe_roc_auc(y_test, y_proba)
    else:
        auc = float("nan")
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=====================================")
    print(f"FINAL TEST RESULTS: {model_name}")
    print("=======================================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc == auc:
        print(f"ROC AUC  : {auc:.4f}")
    else:
        print("ROC AUC  : N/A")
    print("Confusion Matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def main():
    print("\nBreast Cancer Classification Results")
    print("-------------------------------------")

    #Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    #Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    #Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    #Standardize for LR and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #------------------------------
    # 1) Logistic Regression tuning
    #------------------------------
    lr_candidates = []
    for c in LR_C_VALUES:
        lr = LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(X_train_scaled, y_train)
        val_f1, val_auc = evaluate_with_proba(lr, X_val_scaled, y_val)
        lr_candidates.append({
            "name": f"Logistic Regression (C={c})",
            "model": lr,
            "val_f1": val_f1,
            "val_auc": val_auc
        })
    
    best_lr = pick_best_by_validation(lr_candidates)
    print(f"\nBest LR on validation: {best_lr['name']} | val F1={best_lr['val_f1']:.4f} | val AUC={best_lr['val_auc']:.4f}")

    #Retrain best LR on train+val then test
    best_lr_model = LogisticRegression(
        C=float(best_lr["name"].split("C=")[1].rstrip(")")),
        max_iter=5000,
        random_state=RANDOM_STATE
    )
    best_lr_model.fit(scaler.fit_transform(X_trainval), y_trainval) #Refit scaler with trainval
    #Need scaled test with the refit scaler:
    X_test_scaled_final = scaler.transform(X_test)

    #-------------------------------------
    # 2) KNN tuning
    #-------------------------------------
    #Refit scaler for KNN using train only(already fit above on X_train)
    scaler_knn = StandardScaler()
    X_train_scaled_knn = scaler_knn.fit_transform(X_train)
    X_val_scaled_knn = scaler_knn.transform(X_val)

    knn_candidates = []
    for k in KNN_K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled_knn, y_train)
        val_f1, val_auc = evaluate_with_proba(knn, X_val_scaled_knn, y_val)
        knn_candidates.append({
            "name": f"KNN (k={k})",
            "model": knn,
            "val_f1": val_f1,
            "val_auc": val_auc
        })

    best_knn = pick_best_by_validation(knn_candidates)
    print(f"Best KNN on validation: {best_knn['name']} | val F1={best_knn['val_f1']:.4f} | val AUC={best_knn['val_auc']:.4f}")

    #Retrain best KNN on train+val then test
    best_k = int(best_knn["name"].split("k=")[1].rstrip(")"))
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    scaler_knn_final = StandardScaler()
    X_trainval_scaled_knn = scaler_knn_final.fit_transform(X_trainval)
    X_test_scaled_knn_final = scaler_knn_final.transform(X_test)
    knn_final.fit(X_trainval_scaled_knn, y_trainval)

    #-----------------------------------------
    # 3) Decision Tree Tuning
    #-----------------------------------------
    dt_candidates = []
    for depth in DT_MAX_DEPTH_VALUES:
        for mss in DT_MIN_SAMPLES_SPLIT_VALUES:
            dt = DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=depth,
                min_samples_split=mss
            )
            dt.fit(X_train, y_train)

            #Decision Tree has predict_proba so we can use ROC AUC
            val_f1, val_auc = evaluate_with_proba(dt, X_val, y_val)
            dt_candidates.append({
                "name": f"Decision Tree (max_depth={depth}, min_samples_split={mss})",
                "model": dt,
                "val_f1": val_f1,
                "val_auc": val_auc
            })
    
    best_dt = pick_best_by_validation(dt_candidates)
    print(f"Best DT on validation: {best_dt['name']} | val F1={best_dt['val_f1']:.4f} | val AUC={best_dt['val_auc']:.4f}")

    #Retrain best DT on train+val then test
    #Parse params back out (simple parsing)
    name_dt = best_dt["name"]
    #Max_depth portion
    depth_str = name_dt.split("max_depth=")[1].split(",")[0]
    depth_final = None if depth_str == "None" else int(depth_str)
    mss_final = int(name_dt.split("min_samples_split=")[1].rstrip(")"))

    dt_final = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=depth_final,
        min_samples_split=mss_final
    )
    dt_final.fit(X_trainval, y_trainval)

    #------------------------------
    #Final test evaluation
    #------------------------------
    lr_test = print_final_results("Logistic Regression (tuned)", best_lr_model, X_test_scaled_final, y_test)
    knn_test = print_final_results(f"KNN (tuned, k={best_k})", knn_final, X_test_scaled_knn_final, y_test)
    dt_test = print_final_results("Decision Tree (tuned)", dt_final, X_test, y_test)

    #Pick best on test (F1 primary, AUC secondary)
    all_final = [
        ("Logistic Regression (tuned)", lr_test),
        (f"KNN (tuned, k={best_k})", knn_test),
        ("Decision Tree (tuned)", dt_test),
    ]

    best_name, best_metrics = all_final[0]
    for name, m in all_final[1:]:
        if m["f1"] > best_metrics["f1"]:
            best_name, best_metrics = name, m
        elif m["f1"] == best_metrics["f1"]:
            a1 = m["roc_auc"] if m["roc_auc"] == m["roc_auc"] else -1
            a2 = best_metrics["roc_auc"] if best_metrics["roc_auc"] == best_metrics["roc_auc"] else -1
            if a1 > a2:
                best_name, best_metrics = name, m
    
    print("\n=========================================================")
    print("BEST MODEL (based on TEST F1, then ROC AUC)")
    print("=========================================================")
    print(f"Best model: {best_name}")
    print(f"F1-score: {best_metrics['f1']:.4f}")
    if best_metrics["roc_auc"] == best_metrics["roc_auc"]:
        print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
    else:
        print("ROC AUC: N/A")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()




     

