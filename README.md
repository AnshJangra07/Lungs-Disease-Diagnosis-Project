#  Xray Lung Classifier

## Problem statement
Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.
Our task is to create a API whichs predict whether the given images are penumonia or not.

## Solution Proposed
The solution proposed for the above problem is that we have used Computer vision to solve the above problem to classify the data. We have used the Pytorch
framework to solve the above problem also we have have created our custom CNN network with the help of pytorch. Then we have created a API which takes in the images and predicts wheter a person is having Pneumonia or not. Then we have dockerized the application and deployed the model on AWS cloud.


![xray_arch](https://user-images.githubusercontent.com/71321529/216753362-aeb34400-d21d-4b21-b2ce-63b86a47b594.jpg)

## Dataset used
The dataset was shared by Apollo diagnostic center for research purpose. So we hvae created a POC with the given data.

## Tech Stack Used
1. Python 
2. FastAPI 
3. Pytorch
4. Docker
5. AWS
6. Azure

## Infrastructure required
1. AWS S3
2. AWS App Runner
3. Github Actions
