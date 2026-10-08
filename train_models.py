import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

def train_and_save_models():
    print("🔄 Loading dataset...")
    df = pd.read_csv('Heart.csv', na_values=['NA', '?', ''])
    df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
    
    X = df.drop(columns=['AHD', 'HD'])
    y = df['AHD']
    
    num_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']
    cat_features = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=SEED, probability=True),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=SEED)
    }
    
    os.makedirs('models', exist_ok=True)
    metrics_dict = {}
    
    for name, model in models.items():
        print(f"🚀 Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        metrics['PR-AUC'] = auc(rec, prec)
        
        metrics_dict[name] = metrics
        
        # Save model pipeline
        joblib.dump(pipeline, f'models/{name.replace(" ", "_")}_pipeline.pkl')
        print(f"✅ {name} trained and saved. ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("🎉 All models trained and saved successfully!")

if __name__ == "__main__":
    if not os.path.exists('models/model_metrics.json'):
        train_and_save_models()
    else:
        print("✅ Models already exist. Skipping training.")