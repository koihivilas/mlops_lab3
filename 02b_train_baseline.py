import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
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
# (MLflow sometimes struggles with /Volumes/ paths directly for artifacts)
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


with mlflow.start_run(run_name="Baseline_LogReg") as run:
    print("Training Baseline Logistic Regression on precomputed features...")
    
    # 1. Initialize with best parameters found via GridSearchCV
    # Best params: {'C': 10.0, 'class_weight': None}
    model_lr = LogisticRegression(C=10.0, class_weight=None, max_iter=1000)
    
    # 2. Fit Model
    print("Fitting model...")
    model_lr.fit(X_train_tfidf, y_train)
    
    # 3. Evaluate
    y_pred = model_lr.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_params({"model_type": "LogisticRegression", "C": 10.0, "class_weight": "None"})
    mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
    
    joblib.dump(model_lr, "/tmp/model_lr.joblib")
    artifacts = {"vectorizer": "/tmp/vectorizer.joblib", "model": "/tmp/model_lr.joblib", "classes": "/tmp/classes.joblib"}
    
    mlflow.pyfunc.log_model(
        artifact_path="ticket_classifier",
        python_model=TicketClassifierWrapper(),
        artifacts=artifacts,
        registered_model_name="workspace.default.lab3_ticket_classifier",
        signature=signature
    )
    
    # Save run ID to local disk first, then copy to Volumes
    local_run_id_path = "/tmp/baseline_run_id.txt"
    volume_run_id_path = "/Volumes/workspace/default/data/baseline_run_id.txt"
    with open(local_run_id_path, "w") as f:
        f.write(run.info.run_id)
    shutil.copyfile(local_run_id_path, volume_run_id_path)
        
print(f"Baseline F1: {f1:.4f} logged.")
