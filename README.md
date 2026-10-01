# 💼 AI-Based Hiring Prediction System

An end-to-end Machine Learning project that predicts a candidate's recruiter decision based on candidate attributes such as experience, education, skills, certifications, salary expectations, and project experience.

The project covers the complete ML workflow — from data preprocessing and feature engineering to model training, hyperparameter tuning, evaluation, and deployment through a Streamlit web application.

> **Note:** This project is developed for educational and demonstration purposes. It should not be used as an autonomous hiring decision system or as the sole basis for real-world employment decisions.

---

## 📌 Project Overview

The **AI-Based Hiring Prediction System** uses supervised machine learning to classify candidates into two categories:

- ✅ **Hire**
- ❌ **Reject**

The system analyzes candidate-related attributes and uses a trained classification model to generate a predicted recruiter decision along with prediction probabilities.

The project also includes an interactive **Streamlit web application** where users can enter candidate information and receive a model prediction.

---

## 🎯 Objectives

The main objectives of this project are:

- Perform data cleaning and preprocessing.
- Handle missing values.
- Convert categorical data into machine-readable features.
- Encode candidates' multiple skills.
- Train and compare different classification algorithms.
- Tune model hyperparameters using GridSearchCV.
- Evaluate models using multiple classification metrics.
- Validate the model using cross-validation.
- Save the final trained model as a reusable pipeline.
- Build an interactive Streamlit application.
- Demonstrate an end-to-end machine learning workflow.

---

## 📊 Dataset

The dataset contains **1,000 candidate records** and includes the following attributes:

| Feature | Description |
|---|---|
| Resume_ID | Unique candidate/resume identifier |
| Name | Candidate name |
| Skills | Candidate's technical skills |
| Experience (Years) | Years of professional experience |
| Education | Candidate's educational qualification |
| Certifications | Professional certifications |
| Job Role | Target job role |
| Recruiter Decision | Target variable |
| Salary Expectation ($) | Expected salary |
| Projects Count | Number of projects |
| AI Score (0-100) | Candidate AI score |

### Target Distribution

The target variable contains:

- **812 Hire**
- **188 Reject**

The target was encoded as:

```text
Hire   → 1
Reject → 0