# System Architecture

## Training flow

```text
Amazon S3
    |
    v
Data ingestion -> Data transformation -> Model training
                                      |
                                      v
                         Model evaluation and quality gate
                                      |
                         accuracy >= configured threshold
                                      |
                                      v
                       BentoML -> Docker -> Amazon ECR
```

## Runtime components

- `train.py` starts the end-to-end training pipeline.
- `xray/components/` contains the pipeline stages.
- `xray/entity/` defines stage configuration and artifacts.
- `xray/ml/model/` contains the CNN architecture and BentoML service.
- `app.py` provides a local FastAPI fallback.
- `bentofile.yaml` defines the Bento build, model inclusion, and container runtime.

## Generated directories

These directories are runtime outputs and are intentionally excluded from source control:

- `artifacts/` stores timestamped pipeline outputs and trained weights.
- `logs/` stores pipeline logs.
- `data/` stores local copies of the dataset.

The directory names remain unchanged because the pipeline configuration writes to them directly.
