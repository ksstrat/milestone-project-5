# Mildew Detection in Cherry Leaves

A predictive analytics dashboard built with **Streamlit**, **Python**, and **TensorFlow**, aiming to:
1. Visually differentiate healthy vs. mildew-infected cherry leaves.
2. Predict whether a given cherry leaf is healthy or infected using an ML model.

The project delivers a complete CRISP-DM workflow — from data exploration and model development to deployment as an interactive dashboard.

---

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
A **fictitious business scenario** was created where predictive analytics can be applied to a real workplace situation.
In this case, *Farmy & Foods* aims to detect powdery mildew on cherry leaves using Machine Learning to automate manual inspections and maintain crop quality.

The dataset contains over 4,000 images showing healthy and mildew-infected cherry leaves collected from the company’s crop fields.

---

## Business Requirements

The client, *Farmy & Foods*, faces the challenge that their cherry plantations are increasingly affected by powdery mildew, a fungal disease. The current manual inspection process takes approximately 30 minutes per tree and is not scalable for thousands of trees across multiple farms.

Machine Learning offers a way to detect mildew instantly from leaf images, enabling automated disease control and efficiency improvements.

| ID | Requirement | Description |
|----|--------------|-------------|
| BR1 | Visual Study | Conduct a visual study to differentiate healthy and mildew-infected leaves. |
| BR2 | Prediction | Predict if a cherry leaf is healthy or infected. |

Success Metric: **Model Accuracy ≥ 97% on Test Set**

---

## Pages Overview

| Page | Purpose |
|------|----------|
| Project Summary | Dataset and business overview |
| Visual Study (BR1) | Visual differentiations and EDA plots |
| Prediction (BR2) | Upload images and predict leaf health |
| Hypotheses | List and validation of project hypotheses |
| Technical | Model architecture, metrics, and artifacts |

---

## Hypotheses and Validation

- Hypothesis 1: Healthy leaves and mildew-infected leaves show distinguishable texture and color differences.
- Hypothesis 2: Reducing image size to 100x100 maintains accuracy above 97%.
- Hypothesis 3: Model maintains high predictive performance on unseen test data and remains robust under common image variations such as lighting, color tone, and orientation changes.

Each hypothesis will be validated using exploratory data analysis and model evaluation metrics.

---

## Business Requirements and Analytical Approach

Below is the mapping between the defined business requirements and their analytical or ML implementation:

| Business Requirement | Data Visualisation / ML Task |
|----------------------|------------------------------|
| BR1 | Image montage, average image comparison, difference heatmaps |
| BR2 | Convolutional Neural Network (CNN) classifier predicting Healthy / Mildew |

---

## Machine Learning Business Case

| Element | Description |
|----------|-------------|
| **Goal** | Build a binary image classification model that detects powdery mildew on cherry leaves. |
| **Learning Method** | Supervised Learning (Convolutional Neural Network). |
| **Inputs** | RGB image (resized 100x100). |
| **Outputs** | Label: `Healthy` or `Mildew`. |
| **Success Metric** | Accuracy ≥ 97% on test data. |
| **Model Output Relevance** | Enables automated, scalable leaf inspection to maintain crop quality. |

---

## Dashboard Design

| Page | Description |
|------|--------------|
| Project Summary | Introduction, dataset context, business case |
| Visual Study | Class averages, difference images, montages |
| Prediction | File uploader, predictions, probabilities |
| Hypotheses | Data-backed hypothesis validation |
| Technical | Model metrics, confusion matrix, learning curves |

---

## Deployment

### Heroku Deployment Steps
1. Log in to [Heroku](https://www.heroku.com/) and create a new App.
2. Under the **Deploy** tab, choose **GitHub** as the deployment method.
3. Connect your repository and branch.
4. Click **Deploy Branch**.
5. Once deployment succeeds, click **Open App** to access the dashboard.

If the Heroku slug exceeds 500 MB, exclude unnecessary files using `.slugignore`.

**Live App:** _To be added after deployment_

---

## Main Data Analysis and Machine Learning Libraries

| Library | Purpose |
|----------|----------|
| `pandas`, `numpy` | Data manipulation and processing |
| `matplotlib`, `seaborn`, `plotly` | Data visualization |
| `scikit-learn` | Preprocessing, evaluation, pipelines |
| `tensorflow`, `keras` | CNN modeling |
| `streamlit` | Dashboard development |

---

## Notes
- Forked from the official **Code Institute template** for the Cherry Leaves project.
- Python version: `3.12.1` (matches Heroku runtime).
- Deployment target: **Heroku**.

---

## Project Status
Base structure completed (5 pages, navigation, local run successful).
Next steps: Data collection, EDA, model training, and deployment.

---

## Credits

### Content
- Original dataset and context from Code Institute.
- Project scaffold and guidance based on CI walkthrough materials.

### Media
- Dataset images sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).

---

## Acknowledgements
Special thanks to Code Institute for providing the educational framework and dataset foundation for this project.