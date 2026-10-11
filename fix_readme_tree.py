import re

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

correct_tree = '''## Project structure
fraud-detection-mvp/
├── src/
│ ├── data_generator.py # synthetic transaction stream (Kafka producer)
│ ├── feature_engineering.py # batch feature computation (training)
│ ├── schema.py # shared constants (no Polars dependency)
│ ├── predictor.py # online feature computation + ONNX inference
│ ├── train.py # LightGBM training, ONNX + native export
│ ├── consumer.py # Kafka -> API bridge
│ ├── alerts.py # fraud alert delivery (file/console/Slack)
│ ├── drift_report.py # Evidently drift report
│ ├── explain.py # standalone SHAP CLI
│ └── apply_feedback.py # joins feedback into a retraining dataset
├── api/main.py # FastAPI app: /predict, /feedback, dashboard reads
├── frontend/ # React + Vite dashboard
├── docker/ # Dockerfile.api (lean), Dockerfile.stream
├── k8s/ # Kubernetes manifests
├── tests/ # pytest suite
└── .github/workflows/ci.yml

pattern = re.compile(r"## Project structure\n.*?```", re.DOTALL)
new_content, n = pattern.subn(correct_tree, content, count=1)

if n == 0:
    print("ERROR: could not find the 'Project structure' section -- nothing changed.")
else:
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Fixed ({n} replacement made).")
