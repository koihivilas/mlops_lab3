import mlflow
import os

client = mlflow.tracking.MlflowClient()
model_name = "workspace.default.lab3_ticket_classifier"

# 1. Read the Run IDs from previous tasks
try:
    with open("/Volumes/workspace/default/data/baseline_run_id.txt", "r") as f:
        baseline_run_id = f.read().strip()
    with open("/Volumes/workspace/default/data/improved_run_id.txt", "r") as f:
        improved_run_id = f.read().strip()
except Exception as e:
    print("Could not read run IDs. Ensure previous tasks completed successfully.")
    raise e

# 2. Get metrics for both runs
baseline_run = client.get_run(baseline_run_id)
improved_run = client.get_run(improved_run_id)

baseline_f1 = baseline_run.data.metrics.get("f1_weighted", 0)
improved_f1 = improved_run.data.metrics.get("f1_weighted", 0)

print(f"Baseline F1: {baseline_f1:.4f}")
print(f"Improved F1: {improved_f1:.4f}")

PROMOTION_THRESHOLD = 0.01  # Need at least 1% improvement to promote challenger model

if improved_f1 >= baseline_f1 + PROMOTION_THRESHOLD:
    best_run_id = improved_run_id
    print(f"Improved model wins by {improved_f1 - baseline_f1:.4f} (>= threshold {PROMOTION_THRESHOLD})")
else:
    print(f"Improved model did NOT beat baseline by threshold. Keeping baseline.")
    best_run_id = baseline_run_id

# 3. Find the registered model version associated with the best run
versions = client.search_model_versions(f"name='{model_name}'")

best_version = None
for v in versions:
    if v.run_id == best_run_id:
        best_version = v.version
        break

# 4. Promote to Production alias
if best_version:
    client.set_registered_model_alias(name=model_name, alias="production", version=best_version)
    print(f"SUCCESS: Model version {best_version} is now tagged as 'production'.")
else:
    print("WARNING: Could not find registered model version for the best run.")
