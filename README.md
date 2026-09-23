# X-ray Lung Classifier

This project is a proof of concept for classifying chest X-ray images as `NORMAL` or `PNEUMONIA` with PyTorch. It is intended for research and portfolio use only, not for clinical diagnosis.

## Pipeline

```text
Amazon S3 -> data ingestion -> image transforms -> PyTorch training
		  -> evaluation and quality gate -> BentoML -> Docker -> Amazon ECR
```

The training pipeline downloads data from S3, creates reproducible artifacts, trains the CNN, evaluates accuracy, precision, recall, F1-score, and a confusion matrix, and pushes a BentoML image only when the configured accuracy threshold is met. The current default threshold is 80%.

The primary serving path is BentoML in `xray/ml/model/model_service.py`. The FastAPI application in `app.py` remains available as a local fallback and loads `MODEL_PATH` when set, otherwise the newest pipeline-produced model artifact.

## Local usage

Install the pinned dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

Run the training pipeline after configuring AWS credentials and access to the project S3 bucket:

```bash
python train.py
```

The local fallback API can be started with:

```bash
uvicorn app:app --reload
```

For BentoML serving, the pipeline saves `xray_model` and the service is defined in `bentofile.yaml`.

## Dataset

The dataset contains `NORMAL` and `PNEUMONIA` image folders for training and testing. The data is used as a research proof of concept; dataset provenance and clinical performance should be independently verified before any real-world use.

## Technology

- Python and PyTorch
- BentoML and Docker
- Amazon S3 and Amazon ECR
- GitHub Actions with AWS OIDC authentication
