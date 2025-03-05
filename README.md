# Alzheimer's Disease and Healthy Aging Prediction

## Overview

This project utilizes machine learning techniques to predict the likelihood of subjective cognitive decline or memory loss among older adults. The dataset is sourced from the **Behavioral Risk Factor Surveillance System (BRFSS)**, collected from 2015 to 2022 via health-related telephone surveys.

## Team Members

- **Nicole (Cody) Peterson**
- **Diana Roemer**

## Dataset

The dataset consists of **284,142** observations, with key features including:

- **Demographics**: Age, sex, and race/ethnicity.
- **Health Indicators**: Overall health, physical health, and mental health.
- **Behavioral Risk Factors**: Smoking, alcohol use, and binge drinking.
- **Cognitive Health**: Subjective cognitive decline and memory loss.
- **Caregiving Variables**: Information on caregiving for individuals with cognitive impairment.
- **Geographic Data**: State and territory locations with latitude and longitude.

## Project Objectives

The primary objective is to **construct and evaluate machine learning models** that predict cognitive decline using various health-related and behavioral factors. The models will:

- Preprocess and clean data for analysis.
- Select relevant features.
- Apply supervised learning algorithms, including **artificial neural networks**.
- Evaluate model performance and identify key predictors.

## Methods

The project follows a structured **data science pipeline**:

1. **Data Cleaning**
   - Remove missing values from crucial fields.
   - Focus on key predictors and response variables.
2. **Feature Selection**
   - Include demographic, health, and behavioral factors.
3. **Machine Learning Models**
   - Utilize **Artificial Neural Networks (ANNs)**.
   - Compare performance with models like **K-Nearest Neighbors (kNN)**.
4. **Performance Evaluation**
   - Metrics: Mean Squared Error (MSE), R-squared, and Accuracy.

## Key Variables

### **Response Variables (Cognitive Health Outcomes)**

- **Frequent mental distress**
- **Functional difficulties due to cognitive decline**
- **Need for assistance with daily activities due to cognitive decline**
- **Worsening cognitive decline over time**

### **Predictor Variables (Health Risk Factors)**

- **Binge drinking within the past 30 days**
- **Smoking history**
- **High blood pressure diagnosis**
- **Obesity (BMI > 30)**
- **Medication for high blood pressure**

## Clustering Analysis

To uncover patterns among health risk factors, we applied **K-Means clustering**:

- Standardized all predictor variables.
- Used the **Elbow Method** to determine optimal clusters.
- Grouped individuals into clusters based on health risk behaviors.

## Results

- Created structured datasets for model training.
- Identified relationships between health behaviors and cognitive decline.
- Built a predictive model to assess individual risk factors.

## Acknowledgments

- **Centers for Disease Control and Prevention (CDC)** for providing the dataset.
- **Machine Learning Libraries Used**: Scikit-learn, TensorFlow, Pandas, Matplotlib.

## Future Work

- **Enhance feature selection** using domain-specific knowledge.
- **Expand models** to include additional deep learning architectures.
- **Integrate real-time health data** for more accurate predictions.
