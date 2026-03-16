import mlflow
import mlflow.sklearn
import warnings
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import joblib
import scipy.sparse
import shutil
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_experiment("/Shared/Ticket_Classification_Lab3")

print("Loading precomputed features from Volumes...")
X_train_tfidf = scipy.sparse.load_npz("/Volumes/workspace/default/data/features/X_train_tfidf.npz")
X_test_tfidf = scipy.sparse.load_npz("/Volumes/workspace/default/data/features/X_test_tfidf.npz")
y_train = joblib.load("/Volumes/workspace/default/data/features/y_train.joblib")
y_test = joblib.load("/Volumes/workspace/default/data/features/y_test.joblib")

vectorizer_path = "/Volumes/workspace/default/data/features/vectorizer.joblib"
classes_path = "/Volumes/workspace/default/data/features/classes.joblib"

# Copy vectorizer and classes to local /tmp for MLflow logging artifacts
shutil.copy(vectorizer_path, "/tmp/vectorizer.joblib")
shutil.copy(classes_path, "/tmp/classes.joblib")

class TicketClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.model = joblib.load(context.artifacts["model"])
        self.class_names = joblib.load(context.artifacts["classes"])
    
    def predict(self, context, model_input):
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]
        X_tfidf = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)
        return pd.DataFrame({
            "prediction": predictions,
            "confidence": [float(max(p)) for p in probabilities]
        })

signature = ModelSignature(inputs=Schema([ColSpec("string", "text")]), outputs=Schema([ColSpec("string", "prediction"), ColSpec("double", "confidence")]))

with mlflow.start_run(run_name="Improved_LightGBM") as run:
    print("Training Improved LightGBM Model on precomputed features with optimized parameters...")
    # Best params from Optuna Trial 8:
    best_params = {
        'lambda_l1': 0.002847079345119484, 
        'lambda_l2': 5.3032664431945745e-06, 
        'num_leaves': 214, 
        'feature_fraction': 0.9228619968869629, 
        'bagging_fraction': 0.6786612089714443, 
        'bagging_freq': 1, 
        'min_child_samples': 11, 
        'learning_rate': 0.03291686506077265, 
        'n_estimators': 261,
        'class_weight': 'balanced',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1
    }
    
    model_lgb = lgb.LGBMClassifier(**best_params)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
        model_lgb.fit(X_train_tfidf, y_train)
        y_pred = model_lgb.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')
    
    mlflow.log_params({"model_type": "LightGBM"})
    mlflow.log_params(best_params)
    mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
    
    joblib.dump(model_lgb, "/tmp/model_lgb.joblib")
    artifacts = {"vectorizer": "/tmp/vectorizer.joblib", "model": "/tmp/model_lgb.joblib", "classes": "/tmp/classes.joblib"}
    
    mlflow.pyfunc.log_model(
        artifact_path="ticket_classifier",
        python_model=TicketClassifierWrapper(),
        artifacts=artifacts,
        registered_model_name="workspace.default.lab3_ticket_classifier",
        signature=signature
    )
    
    # Save run ID to local disk first, then copy to Volumes
    local_run_id_path = "/tmp/improved_run_id.txt"
    volume_run_id_path = "/Volumes/workspace/default/data/improved_run_id.txt"
    with open(local_run_id_path, "w") as f:
        f.write(run.info.run_id)
    shutil.copyfile(local_run_id_path, volume_run_id_path)
        
print(f"Improved F1: {f1:.4f} logged.")
