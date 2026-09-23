# X-ray Lung Classifier

An end-to-end computer vision and MLOps project that classifies chest X-ray images into `NORMAL` and `PNEUMONIA` classes.

> **Important:** This project is a research and portfolio proof of concept. It is not a medical device and must not be used as a substitute for professional clinical diagnosis.

## Overview

The project combines a PyTorch CNN with a modular training pipeline, Amazon S3 data ingestion, automated evaluation, BentoML model serving, Docker containerization, Amazon ECR, and GitHub Actions.

The latest local training run achieved **95.22% test accuracy** on `1,171` test images. Accuracy alone is not sufficient for medical use, so the evaluation pipeline also records precision, recall, F1-score, and a confusion matrix.

Evaluation metrics:  accuracy=95.22% precision=96.51% recall=96.96% f1=96.73% confusion_matrix=[[286, 30], [26, 829]]

## Architecture

https://user-images.githubusercontent.com/71321529/216753362-aeb34400-d21d-4b21-b2ce-63b86a47b594.jpg

```text
Amazon S3
	|
	v
Data ingestion -> Data transformation -> PyTorch training
									  |
									  v
						 Evaluation and quality gate
									  |
									  v
					   BentoML -> Docker -> Amazon ECR
									  |
									  v
						 AWS deployment or local serving
```

The model is pushed only when it passes the configured minimum accuracy threshold, currently `80%`.

## Pipeline Components

The training pipeline is divided into independent components. Each component receives a typed artifact or configuration object and returns the output required by the next stage.

### 1. Data Ingestion

**File:** `xray/components/data_ingestion.py`

- Connects to the configured Amazon S3 bucket through `S3Operation`.
- Synchronizes the `data/` folder into a timestamped local artifact directory.
- Creates a `DataIngestionArtifact` containing the train and test paths.
- Keeps cloud-storage logic separate from the rest of the ML pipeline.

### 2. Data Transformation

**File:** `xray/components/data_transformation.py`

- Loads images using `torchvision.datasets.ImageFolder`.
- Applies training augmentation: resize, center crop, color jitter, horizontal flip, and random rotation.
- Applies deterministic test preprocessing without augmentation.
- Creates PyTorch `DataLoader` objects with batch size `16`.
- Shuffles training data while keeping test data order deterministic.
- Saves the transform objects with `joblib` for reproducibility.

### 3. Model Training

**File:** `xray/components/model_training.py`

- Creates the custom CNN defined in `xray/ml/model/arch.py`.
- Trains with negative log-likelihood loss and SGD optimization.
- Uses a `StepLR` learning-rate scheduler.
- Reports batch loss, training accuracy, test loss, and test accuracy after each epoch.
- Saves the PyTorch state dictionary to the timestamped `artifacts/` directory.
- Registers the trained model in the BentoML model store as `xray_model`.

### 4. Model Evaluation

**File:** `xray/components/model_evaluation.py`

- Loads the saved model weights with CPU/GPU device mapping.
- Runs inference on the deterministic test loader without gradients.
- Calculates accuracy, precision, recall, and F1-score.
- Generates a binary confusion matrix in `[NORMAL, PNEUMONIA]` order.
- Returns all metrics through a `ModelEvaluationArtifact`.

### 5. Model Quality Gate

**File:** `xray/pipeline/train_pipeline.py`

- Compares the evaluation accuracy with `ModelEvaluationConfig.minimum_accuracy`.
- Stops the pipeline when the model is below the configured threshold.
- Calls the model pusher only after the quality gate passes.
- Prevents an underperforming model from being containerized or published.

### 6. Model Pusher

**File:** `xray/components/model_pusher.py`

- Builds a Bento from `bentofile.yaml`.
- Captures the exact generated Bento tag instead of assuming a `latest` tag.
- Containerizes the Bento using Docker.
- Resolves the AWS account ID through STS instead of hardcoding a registry account.
- Authenticates with Amazon ECR and pushes `xray_bento_image:latest`.

### 7. Serving Layer

**BentoML:** `xray/ml/model/model_service.py`

- Loads the bundled `xray_model` artifact.
- Applies the same resize and normalization preprocessing used during testing.
- Returns the predicted class label from the BentoML API.

**FastAPI fallback:** `app.py`

- Provides a simple local `/predict` endpoint.
- Loads `MODEL_PATH` when explicitly configured.
- Otherwise selects the newest pipeline-generated `model.pt` artifact.

## Project Structure

```text
.
├── app.py                         # Local FastAPI fallback API
├── train.py                       # Training pipeline entry point
├── bentofile.yaml                 # BentoML build and container config
├── requirements.txt               # Verified runtime dependencies
├── .github/workflows/             # CI/CD and continuous-training workflow
├── docs/                          # Architecture and project documentation
├── scripts/                       # Runner and environment scripts
├── notebook/                      # Experiments and investigations
├── flowcharts/                    # Design and pipeline diagrams
├── xray/
│   ├── components/                # Ingestion, transformation, training, evaluation
│   ├── entity/                    # Pipeline artifacts and configuration
│   ├── ml/model/                  # CNN architecture and BentoML service
│   └── pipeline/                  # End-to-end training orchestration
├── data/                          # Local dataset, generated and ignored
├── artifacts/                     # Timestamped outputs, generated and ignored
└── logs/                          # Runtime logs, generated and ignored
```

The generated directories are kept at the repository root because the pipeline writes to these paths. They are excluded from version control; the source layout and documentation remain clean without changing runtime behavior.

## Technology

- Python 3.10+ for local development
- PyTorch and Torchvision
- FastAPI for the local fallback API
- BentoML 1.4.39 for model packaging and serving
- Docker for containerization
- Amazon S3 for training data
- Amazon ECR for container images
- GitHub Actions with AWS OIDC authentication

## Setup

Create and activate the virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The training pipeline expects AWS credentials with access to the configured S3 bucket. Verify the AWS identity before running it:

```powershell
aws sts get-caller-identity
```

## Train And Package

Run the complete pipeline:

```powershell
python train.py
```

The pipeline performs these steps:

1. Downloads the dataset from Amazon S3.
2. Applies separate training and test transforms.
3. Trains the PyTorch CNN.
4. Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
5. Stops before deployment if accuracy is below the quality threshold.
6. Builds the BentoML service and Docker image.
7. Logs in to Amazon ECR and pushes the image when AWS access and the repository are available.

The ECR repository must exist before the push step:

```powershell
aws ecr create-repository --repository-name xray_bento_image --region us-east-1
```

## Run The Container Locally

The pipeline produces the local image `xray_bento_image:latest`:

```powershell
docker run --rm -p 3000:3000 xray_bento_image:latest
```

The BentoML service is then available at `http://localhost:3000`.

## Local FastAPI Fallback

The fallback API loads `MODEL_PATH` when provided. Otherwise it selects the newest pipeline-generated model artifact:

```powershell
uvicorn app:app --reload
```

Send an image to the prediction endpoint:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\xray.jpg"
```

Example response:

```json
{
  "prediction_index": 1,
  "prediction_label": "Pneumonia"
}
```

## Deployment Workflow

The intended deployment flow is:

```text
GitHub Actions
	-> AWS OIDC role
	-> EC2 training runner
	-> S3 dataset
	-> BentoML and Docker
	-> Amazon ECR
```

The workflow uses short-lived AWS credentials through GitHub OIDC. Configure the required repository secrets, including `AWS_ROLE_ARN`, AWS region, EC2 runner settings, and the GitHub runner token, before enabling continuous training.

## Dataset Layout

The dataset follows the `ImageFolder` layout:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
	├── NORMAL/
	└── PNEUMONIA/
```

## Limitations

- The model is a proof of concept and has not been clinically validated.
- Reported metrics depend on the dataset split and should not be interpreted as clinical performance.
- The current architecture is a custom CNN; transfer learning and calibration are possible future improvements.
- Docker images include the PyTorch runtime and can be large, especially when CUDA dependencies are installed.

More detail is available in [docs/architecture.md](docs/architecture.md).
