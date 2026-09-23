import sys
from typing import Tuple

import torch
from torch.nn import Module, NLLLoss
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelEvaluation:
   def __init__(
      self,
      data_transformation_artifact: DataTransformationArtifact,
      model_evaluation_config: ModelEvaluationConfig,
      model_trainer_artifact: ModelTrainerArtifact,
   ):

      self.data_transformation_artifact = data_transformation_artifact

      self.model_evaluation_config = model_evaluation_config

      self.model_trainer_artifact = model_trainer_artifact

   def configuration(self) -> Tuple[DataLoader, Module, Module]:
      logging.info("Entered the configuration method of Model evaluation class")

      try:
         test_dataloader: DataLoader = (
               self.data_transformation_artifact.transformed_test_object
         )

         model: Module = Net()
         model.load_state_dict(
            torch.load(
               self.model_trainer_artifact.trained_model_path,
               map_location=self.model_evaluation_config.device,
               weights_only=True,
            )
         )

         model.to(self.model_evaluation_config.device)

         cost: Module = NLLLoss()

         model.eval()

         logging.info("Exited the configuration method of Model evaluation class")

         return test_dataloader, model, cost

      except Exception as e:
         raise XRayException(e, sys)

   def test_net(self) -> tuple[float, float, float, float, list[list[int]]]:
      logging.info("Entered the test_net method of Model evaluation class")

      try:
         test_dataloader, model, cost = self.configuration()
         true_labels: list[int] = []
         predicted_labels: list[int] = []
         total_loss = 0.0

         with torch.no_grad():
               for images, labels in test_dataloader:
                  images = images.to(self.model_evaluation_config.device)
                  labels = labels.to(self.model_evaluation_config.device)
                  output = model(images)
                  loss = cost(output, labels)
                  predictions = torch.argmax(output, dim=1)

                  true_labels.extend(labels.cpu().tolist())
                  predicted_labels.extend(predictions.cpu().tolist())
                  total_loss += loss.item() * labels.size(0)

         if not true_labels:
            raise ValueError("The evaluation dataset is empty")

         true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in zip(true_labels, predicted_labels))
         false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in zip(true_labels, predicted_labels))
         false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in zip(true_labels, predicted_labels))
         true_negative = sum(actual == 0 and predicted == 0 for actual, predicted in zip(true_labels, predicted_labels))

         accuracy = sum(actual == predicted for actual, predicted in zip(true_labels, predicted_labels)) / len(true_labels) * 100
         precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
         recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
         f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
         confusion_matrix = [[true_negative, false_positive], [false_negative, true_positive]]

         logging.info(
            "Evaluation metrics: loss=%.4f accuracy=%.2f%% precision=%.2f%% recall=%.2f%% f1=%.2f%% confusion_matrix=%s",
            total_loss / len(true_labels), accuracy, precision * 100, recall * 100,
            f1_score * 100, confusion_matrix,
         )
         logging.info("Exited the test_net method of Model evaluation class")

         return accuracy, precision * 100, recall * 100, f1_score * 100, confusion_matrix

      except Exception as e:
         raise XRayException(e, sys)

   def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
      logging.info(
         "Entered the initiate_model_evaluation method of Model evaluation class"
      )

      try:
         accuracy, precision, recall, f1_score, confusion_matrix = self.test_net()

         model_evaluation_artifact: ModelEvaluationArtifact = (
               ModelEvaluationArtifact(
                  model_accuracy=accuracy,
                  model_precision=precision,
                  model_recall=recall,
                  model_f1_score=f1_score,
                  confusion_matrix=confusion_matrix,
               )
         )

         logging.info(
               "Exited the initiate_model_evaluation method of Model evaluation class"
         )

         return model_evaluation_artifact

      except Exception as e:
         raise XRayException(e, sys)