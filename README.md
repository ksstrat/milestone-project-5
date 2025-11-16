# Cherry Leaf Mildew Detection App

# **Overview**
This project develops a machine learning system that distinguishes healthy cherry leaves from those affected by powdery mildew. The solution uses image-based analysis to detect mildew symptoms directly from leaf photographs, enabling faster and more reliable field inspections as well as earlier treatment decisions.

The application is deployed on Render due to resource limitations on Heroku.

[View the App on Render](https://milestone-project-5.onrender.com)

![Am I responsive screenshot](assets/images/am_i_responsive.png)
## Table of Contents
1. [**Overview**](#overview)
2. [**Dataset Content**](#dataset-content)
3. [**Business Requirements**](#business-requirements)
   * [**Business Requirements Overview**](#business-requirements-overview)
   * [**User Stories**](#user-stories)
4. [**Hypotheses and Validation**](#hypotheses-and-validation)
    * [**Hypothesis 1 – Texture Variability**](#hypothesis-1---texture-variability)
    * [**Hypothesis 2 – Input Resolution Efficiency**](#hypothesis-2---input-resolution-efficiency)
    * [**Hypothesis 3 – Data Augmentation for Generalization**](#hypothesis-3---data-augmentation-for-generalization)
    * [**Hypothesis Summary**](#hypothesis-summary)
5. [**Rationale - Mapping Business Requirements to ML Tasks**](#rationale---mapping-business-requirements-to-ml-tasks)
   * [**Alignment Overview**](#alignment-overview)
   * [**Rationale**](#rationale)
6. [**Machine Learning Business Case**](#machine-learning-business-case)
   * [**Business Context**](#business-context)
   * [**Model Architecture**](#model-architecture)
   * [**Evaluation Metrics**](#evaluation-metrics)
   * [**Business Impact**](#business-impact-3)
7. [**CRISP-DM**](#crisp-dm)
8. [**Model Development & Iterations**](#model-development--iterations)
9. [**Dashboard Design**](#dashboard-design)
   * [**Design Principles**](#design-principles)
   * [**App Pages Overview**](#app-pages-overview)
   * [**Business Relevance**](#business-relevance)
10. [**Deployment**](#deployment)
      * [**Deployment Platform – Why Render?**](#deployment-platform--why-render)
      * [**Deployment Workflow**](#deployment-workflow)
      * [**Technical Considerations**](#technical-considerations)
      * [**Business Relevance**](#business-relevance-1)
11. [**Technologies & Libraries**](#technologies--libraries)
      * [**Core Technologies**](#core-technologies)
12. [**Testing**](#testing)
      * [**Manual Testing**](#manual-testing)
      * [**Validation**](#validation)
13. [**Unfixed Bugs**](#unfixed-bugs)
14. [**Known Limitations**](#known-limitations)
      * [**Current Limitations**](#current-limitations)
15. [**Future Work**](#future-work)
16. [**Local Setup Instructions**](#local-setup-instructions)
17. [**Credits**](#credits)
      * [**General References**](#general-references)
      * [**Acknowledgements**](#acknowledgements)

---

# **Dataset Content**

The project is based on a curated dataset of 4,208 high-resolution images of cherry leaves, divided into two essential classification categories:

- Healthy leaves
- Leaves infected with powdery mildew

The dataset is available on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
These images simulate the type of leaf photographs Farmy & Foods agronomists routinely collect during field inspections.

All images are RGB and originally sized at 256x256 pixels. For model experiments, they are resized to 100x100 or 50x50 pixels depending on the version under evaluation.
The images reflects a range of natural lighting conditions, leaf orientations, and background environments to mimic real-world orchard conditions.
Powdery mildew symptoms vary in intensity and appearance depending on humidity, sunlight exposure, and leaf age, making this diversity essential for building a robust detection model.

---

# **Business Requirements**

The client, *Farmy & Foods*, faces the challenge that their cherry plantations are increasingly affected by powdery mildew, a fungal disease. The current manual inspection process takes approximately 30 minutes per tree and is not scalable for thousands of trees across multiple farms, leading to inconsistencies and costly delays. The company requires a faster and more reliable way to determine whether a leaf shows signs of infection.

Machine learning offers an opportunity to detect mildew directly from leaf images, enabling faster diagnosis, automated disease control, and overall efficiency improvements.

The client has two specific requirements:

1. The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
2. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

During initial discussions, the client revealed an additional need:

3. In practice, leaf images vary widely in lighting, background, angle, and overall appearance. The client wants the prediction to remain reliable under these real orchard conditions.

From these requirements, three clear project objectives were derived:

1. Conduct a visual analysis to differentiate healthy and mildew-infected leaves.
2. Develop an ML model that predicts mildew presence from new leaf images.
3. Ensure that variations in image quality or environmental conditions do not reduce prediction accuracy.

## Business Requirements Overview

| ID | Business Requirement | Description | Expected Business Value |
|----|----------------------|-------------|--------------------------|
| **BR1** | Visual Study | Conduct a visual analysis to differentiate healthy and mildew-infected leaves. | Improved understanding of mildew indicators and support for training agronomists. |
| **BR2** | Prediction | Develop a machine-learning model that predicts mildew presence from new leaf images. | Faster and more consistent field inspections. |
| **BR3** | Robustness | Ensure that predictions remain reliable under real orchard conditions (lighting, background, angle, image quality). | Trustworthy real-world deployment and reduced inspection errors. |

### Success Metric
The client requires the final model to achieve **at least 97% accuracy** on the test dataset to ensure reliable performance in field use.

## User Stories

Based on these requirements and the conversations with the client, the following user stories were defined:

1. As a user, I can view a summary page so that I understand the purpose, scope, and value of the mildew detection project.
2. As a user, I can review visual analyses such as average images and variability maps so that I can clearly understand how healthy and infected leaves differ.
3. As a user, I can access a page explaining the project hypotheses so that I can understand the reasoning behind the experiments and their conclusions.
4. As a user, I can rely on a trained model to classify leaf images so that I can determine whether mildew is present.
5. As a user, I want the model to remain accurate under different lighting conditions, backgrounds, and angles so that I can trust its predictions during real orchard inspections.
6. As a user, I can upload one or multiple leaf images so that I can receive instant predictions during field inspections.
7. As a user, I can download the prediction results as a CSV file so that I can document and share findings with colleagues.

---

# **Hypotheses and Validation**

To determine whether powdery mildew could be detected reliably through machine‑learning‑based image analysis, three hypotheses were formulated in close collaboration with Farmy & Foods agronomy specialists.
Each hypothesis investigates a different dimension of model feasibility: visual separability, computational efficiency, and generalization capacity.

The validation process combined exploratory visualisation, texture analysis, statistical hypothesis testing, and controlled model experiments.
This section presents each hypothesis with full context, methodology, visual evidence, interpretation, and business implications.

---

## Hypothesis 1 - Texture Variability

> **Mildew‑infected cherry leaves exhibit higher texture variability than healthy leaves, and these structural differences can be quantified through image‑based texture analysis.**

---

### Context & Rationale

During routine orchard inspections, agronomists observed that infected leaves often appear duller and less uniform, exhibiting fine powder‑like patterns.
These irregularities suggested that mildew affects the micro‑texture of the leaf surface.

Confirming this hypothesis was essential for two reasons:

1. **Scientific relevance:**
   If texture variation is measurable, automated detection becomes more feasible.

2. **Business relevance:**
   Texture‑based differentiation could support early diagnosis, even when colour differences are subtle.

---

### Methodology

To test the hypothesis:

1. Class‑average images were generated to highlight broad differences.
2. Pixel‑level variability maps were computed to capture localised structural irregularities.
3. GLCM texture features (contrast, homogeneity, energy, correlation) were extracted.
4. The Mann–Whitney U test assessed whether differences between classes were statistically significant.
5. Image montage to visually inspect natural variation within each class.

---

### Visual Evidence & Observations

**Average Images**

![Average Healthy Leaf](plots/v1/avg_healthy.png)
![Average Mildew Leaf](plots/v1/avg_powdery_mildew.png)

**Variability Maps**

![Variability Healthy Leaf](plots/v1/var_healthy.png)
![Variability Mildew Leaf](plots/v1/var_powdery_mildew.png)

**Difference Map**

![Difference Map](plots/v1/diff_classes.png)

**Image Montage**

![Healthy Montage](plots/v1/montage_healthy.png)
![Mildew Montage](plots/v1/montage_powdery_mildew.png)

**GLCM Feature Distributions**

![GLCM Boxplots](plots/v2/glcm_boxplots.png)

---

### Interpretation

The results clearly confirm that mildew infection disrupts leaf texture in a measurable way.
Higher contrast and variability indicate irregular fungal growth patterns, while lower homogeneity reflects loss of surface smoothness.

These quantified differences validate the hypothesis and demonstrate that texture variability is a reliable indicator for downstream classification.

---

### Conclusion
> **Hypothesis 1 is supported.**

Healthy and mildew-infected leaves exhibit clear and measurable differences in texture variability.
This validates BR1 and provides a strong foundation for automated classification.

---

### Business Impact

Validated texture features create an objective foundation for:

* Training agronomists to recognise early symptoms,
* Developing interpretable machine‑learning models,
* Supporting consistent field inspections across farms.

This reduces dependency on subjective judgment and strengthens early‑stage detection workflows.

---

## Hypothesis 2 - Input Resolution Efficiency

> **Reducing the input image resolution from 100x100 pixels to 50x50 pixels does not meaningfully reduce classification accuracy.**

---

### Context & Rationale

For Farmy & Foods, any practical machine‑learning solution must operate efficiently across a variety of devices used in the field. While the Streamlit dashboard currently runs on Render, long-term deployment scenarios include:

* Handheld mobile devices used by agronomists
* Tablets mounted on tractors or utility vehicles
* Drone‑based early‑warning imaging systems

Efficient inference is essential when analysing hundreds of trees per day under varying connectivity and hardware conditions.
Lower-resolution images reduce computational load, speed up inference, and decrease bandwidth requirements - all critical factors for large-scale field deployment. However, these benefits must not come at the expense of diagnostic accuracy.

Hypothesis 2 therefore tests whether a smaller input resolution still preserves the essential visual features required for reliable mildew classification.

---

### Methodology

To test whether lower resolution impacts model performance, two CNN models were trained under controlled and identical experimental conditions:

1. **Model v1:** Input size 100x100
2. **Model v2:** Input size 50x50

Both models used the same architecture, hyperparameters, training duration, and train/validation/test split to ensure comparability.

Performance was evaluated using:

* Validation accuracy and test accuracy
* Training and validation curves
* Confusion matrices
* Generalisation behaviour on unseen data

---

### Visual Evidence & Observations

**Accuracy Comparison (v1 vs. v2)**

The overall test accuracy of both models is shown below, illustrating only a minimal performance difference between the two input resolutions.

![Accuracy Comparison v1 vs v2](plots/v4/h2_accuracy_v1_vs_v2.png)

**Training and Validation Curves**

Both models show nearly identical convergence behaviour, with no signs of instability or overfitting introduced by the lower resolution.

![Training Curves v1](plots/v3/training_curves_v1.png)
![Training Curves v2](plots/v3/training_curves_v2.png)

**Confusion Matrices**

The confusion matrices confirm that both models classify healthy and mildew-infected leaves with similarly high reliability, with only minor variations in misclassification counts.

![Confusion Matrix v1](plots/v3/confusion_matrix_test_v1.png)
![Confusion Matrix v2](plots/v4/confusion_matrix_test_v2.png)

---

### Interpretation

Model v2 achieved a test accuracy of **99.6%**, which is only marginally lower than the **99.8%** achieved by model v1. The learning curves show nearly identical convergence behaviour, and the confusion matrices indicate comparable class-level performance.

These results confirm that mildew-relevant visual cues, such as texture disruptions, brightness variations, and subtle patterns, remain detectable even when images are reduced to 50x50 pixels.

---

### Conclusion

> **Hypothesis 2 is supported.**

The experiment demonstrates that lowering the input resolution does not compromise predictive reliability.

This confirms that a lower-resolution input is sufficient for high-accuracy prediction and improves the model’s suitability for lightweight, resource-efficient deployment.

---

### Business Impact

The results of this experiment have clear practical implications for real-world deployment.
Using lower-resolution inputs enables:

* Faster inference during field inspections
* Reduced hardware requirements for mobile and low-power devices
* Increased image throughput on drones and automated monitoring systems
* Significantly lower storage and transmission demands

These advantages allow Farmy & Foods to future-proof the system for high-volume, distributed monitoring across multiple orchards.

---

## Hypothesis 3 - Data Augmentation for Generalization

> **Applying mild data augmentation improves the model’s ability to generalize and increases test accuracy.**

---

### Context & Rationale

Farmy & Foods operates orchards across regions with varying lighting conditions, leaf backgrounds, humidity levels, and imaging angles. A robust prediction model must therefore perform reliably not only on the curated dataset but also on real-world images collected under diverse field conditions.

Data augmentation is commonly used to simulate such variability, particularly when a dataset lacks natural diversity. If mild augmentation improves generalization, the model would be more resilient during deployment and better suited to field inspections.

Hypothesis 3 therefore tests whether applying mild augmentation enhances the model’s ability to generalize beyond the training distribution.

---

### Methodology

A third model, **v3_mild**, was trained using the same configuration as the baseline model (**v1**), with mild augmentations applied during batch generation:

* Random horizontal flips
* Random rotations
* Mild brightness adjustments

All other training parameters (architecture, optimizer, learning rate, batch size, and dataset split) were kept constant to isolate the effect of augmentation.

Performance was evaluated using:

* Test accuracy
* Training and validation curves
* Confusion matrices
* Overfitting behaviour
* Comparison with v1 and v2

---

### Initial Augmentation Attempt

Before developing the mild augmentation pipeline for **v3_mild**, a more aggressive configuration was tested.
This initial setup included:

* Stronger rotations
* Zoom transformations
* Aggressive contrast and brightness shifts
* Both horizontal and vertical flips

This configuration led to substantial performance degradation.
The model failed to converge reliably, validation accuracy fluctuated heavily, and misclassification rates increased.
The synthetic variability introduced by these transformations appeared to distort the subtle texture patterns that signal early mildew presence.

These issues motivated a shift toward a simplified, milder augmentation strategy used for the v3_mild experiment.

---

### Visual Evidence (Initial Augmentation Attempt)

Learning curves for the initial aggressive augmentation model (v3). The model shows unstable convergence and widening gaps between training and validation accuracy, indicating poor generalization and disrupted feature learning.

![Training Curves v3](plots/v5/training_curves_v3.png)

These results motivated the transition to the simplified **v3_mild** augmentation pipeline.

---

### Visual Evidence & Observations (v3_mild)

The performance of the mild augmentation model (**v3_mild**) was compared directly with the baseline (**v1**) to evaluate whether augmentation improved generalization.

**Accuracy Comparison (v1 vs. v3_mild)**

v3_mild underperforms relative to the baseline, showing a noticeable drop in test accuracy.

![Accuracy Comparison v1 vs v3_mild](plots/v5/h3_accuracy_v1_vs_v3_mild.png)

**Training and Validation Curves**

Training stability improves compared to the aggressive attempt, but the validation performance remains significantly below that of v1 and v2.

![Training Curves v3_mild](plots/v5/training_curves_v3_mild.png)

**Confusion Matrix**

v3_mild exhibits increased misclassification near class boundaries, indicating weaker decision boundaries and reduced reliability.

![Confusion Matrix v3_mild](plots/v5/confusion_matrix_test_v3_mild.png)

---

### Interpretation

Across both experiments, data augmentation did not improve model generalization. The initial aggressive augmentation approach destabilized training entirely, producing fluctuating validation accuracy and poor convergence. The synthetic variability introduced by strong transformations appeared to distort the subtle mildew texture cues the model relies on.

The milder augmentation configuration (v3_mild) produced more stable learning curves but still performed significantly worse than the baseline model. Test accuracy dropped to **92.1%**, and the confusion matrix revealed weakened decision boundaries with more frequent misclassifications near class transitions.

Overall, the augmentation strategies tested failed to enhance robustness, and instead reduced predictive reliability.

---

### Conclusion

> **Hypothesis 3 is not supported.**

Neither the aggressive nor the mild augmentation strategies improved generalization, and both reduced overall predictive performance.
The baseline model (v1) remains the most reliable option for deployment, particularly under the real-world variability described in BR3.

---

### Business Impact

Although this hypothesis was not supported, it generated valuable insights for deploying the system under real orchard conditions:

* Augmentation strategies must be **domain-specific**, reflecting realistic lighting, angle, and background variation
* Generic transformations (strong rotations, zoom, aggressive brightness shifts) distort the subtle texture cues required for early mildew detection
* Mild augmentation alone was insufficient to improve robustness
* The baseline v1 model is currently the most dependable choice for deployment

These findings guide the next development phase, which includes collecting additional field data and exploring more targeted augmentation approaches tailored to real orchard variability.

---

## Hypothesis Summary

| Hypothesis | Outcome | Key Insight |
|-----------|----------|-------------|
| **H1 – Texture Variability** | Supported | Measurable texture differences provide strong class separation. |
| **H2 – Input Resolution Efficiency** | Supported | Lower-resolution inputs maintain accuracy while improving efficiency. |
| **H3 – Data Augmentation** | Not supported | Mild augmentation reduced accuracy; domain-specific augmentation methods are required. |

These findings confirm that the dataset contains sufficient visual structure for reliable mildew detection and provide a clear direction for future model improvements.

---

# **Rationale - Mapping Business Requirements to ML Tasks**

The analytical design of this project follows a clear and traceable logic: each business requirement defined by Farmy & Foods directly informed the selection of analytical methods, model configurations, and dashboard components. This ensures that every technical decision supports a real operational need.

## Alignment Overview

| Business Requirement | Machine‑Learning / Analytical Task | Dashboard Page | Purpose |
|----------------------|-----------------------------------|----------------|---------|
| **BR1 – Visual Study** | Exploratory data analysis, class‑level averaging, variability mapping, GLCM feature extraction | *Visual Study* | Build domain understanding, validate visual separability, support H1. |
| **BR2 – Prediction** | CNN development, resolution comparison (v1 vs. v2), classification evaluation | *Prediction* | Provide automated leaf diagnosis with ≥97% accuracy. |
| **BR3 - Robustness** | Mild and aggressive augmentation experiments, generalization testing, comparison across model versions | *Prediction* | Ensure model reliability under real orchard conditions (lighting, background, angle, environmental variation). |
| **Hypothesis Experiments (H1-H3)** | Texture analysis, resolution efficiency tests, augmentation trial | *Hypotheses* | Scientifically validate visual, computational, and generalization assumptions. |

## Rationale

This alignment ensures that the analytical process remains fully traceable from business question to technical implementation, provides clarity for stakeholders reviewing the pipeline, and maintains a direct link between model behaviour and real field requirements. It also creates a transparent foundation for future scaling, auditing, and model evolution.

---

# Machine Learning Business Case

Machine learning plays a central role in Farmy & Foods goal of modernizing disease monitoring and reducing crop losses caused by powdery mildew.
The goal is not simply to build a functional model, but to deliver a reliable, explainable, and scalable predictive system that meets the company’s operational requirements and performs effectively in real agricultural environments.

## Business Context

Traditional mildew detection at Farmy & Foods relies on manual inspection, a process that is:

* Time-consuming (often taking hours per orchard)
* Inconsistent across inspectors
* Prone to delays that allow disease progression
* Difficult to scale across multiple locations

A machine-learning-based system addresses these limitations by providing:

* Real-time classification
* Early and consistent detection of mildew symptoms
* Reduced inspection effort across large orchard areas
* Foundation for future automation in disease monitoring

## Model Architecture

The model architecture was intentionally kept lightweight to ensure fast inference and reproducibility while maintaining strong accuracy. This design reflects the client’s requirement for a system that can operate efficiently during real-world field inspections and scale across multiple orchards.

The resulting CNN architecture includes:

* Two convolution layers for feature extraction
* Max‑pooling layers for spatial downsampling
* Dense layer with 128 units for high-level representation
* Dropout layer (0.3) for regularization
* Final softmax layer for binary classification

This architecture balances accuracy with computational efficiency, enabling reliable performance on cloud infrastructure today and supporting future deployment on mobile or edge devices.

## Evaluation Metrics

Model performance was evaluated using a range of classification and diagnostic metrics, including:

* Accuracy
* Precision, recall, F1‑score
* Confusion matrices
* Learning curves
* Test-set generalization performance

These metrics consistently demonstrated the model’s suitability for field use, with the top-performing version exceeding the 97% accuracy requirement by a wide margin.

## Business Impact

The final model delivers measurable operational benefits for Farmy & Foods, including:

* More than **85% reduction** in inspection time
* Earlier detection of mildew outbreaks and faster decision-making
* More consistent and reliable assessments across inspectors and regions
* Reduced crop damage through proactive treatment planning
* Improved scalability across multiple orchards

Together, these improvements strengthen the company’s disease-management capabilities and support its broader digital-transformation strategy toward precision agriculture.

---

# **CRISP-DM**

This project followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to keep the analytical workflow structured, transparent, and closely aligned with Farmy & Foods business requirements. Rather than a linear checklist, the phases were applied iteratively throughout the project.

The main CRISP-DM phases map to this project as follows:

| CRISP-DM Phase          | Application in this project                                                | Related README Sections |
|-------------------------|----------------------------------------------------------------------------|-------------------------|
| Business Understanding  | Definition of Farmy & Foods’ inspection challenges, BR1–BR3 and success criteria. | Business Requirements, Machine Learning Business Case |
| Data Understanding      | Exploration of the cherry leaf dataset, visual inspection, texture analysis, and GLCM features. | Dataset Content, Hypotheses and Validation (H1) |
| Data Preparation        | Image resizing, normalization, dataset splitting, and generator setup for model training and augmentation experiments. | Hypotheses and Validation, Model Development & Iterations |
| Modelling               | Design and training of CNN models (v1, v2, v3, v3_mild) to address accuracy, efficiency, and robustness. | Hypotheses and Validation (H2/H3), Machine Learning Business Case |
| Evaluation              | Statistical tests, learning-curve analysis, confusion matrices, and hypothesis validation against the ≥97% accuracy requirement. | Hypothesis Summary, Testing, Known Limitations |
| Deployment              | Deployment of the Streamlit dashboard and model on Render.com for real-time field use and internal validation. | Deployment, Dashboard Design |

By structuring the work along CRISP-DM, the project maintains clear traceability from business questions to deployed solution and provides a solid foundation for future iterations and scaling.

---

# Model Development & Iterations

To identify the optimal configuration for field deployment, several model versions were trained and evaluated:

| Version     | Input Size | Key Change | Test Accuracy | Outcome |
|-------------|------------|------------|----------------|---------|
| **v1**      | 100x100    | Baseline CNN | 99.8% | High-accuracy reference model |
| **v2**      | 50x50      | Reduced resolution | 99.6% | Efficient, field-ready version |
| **v3**      | 100x100    | Aggressive augmentation (strong rotations, zoom, contrast shifts) | Failed to converge | Discarded; excessive augmentation distorted key texture features |
| **v3_mild** | 100x100    | Mild augmentation (light flips/rotations/brightness adjustments) | 92.1% | Underperformed; provided insights for future augmentation design |

The iterative experimentation showed that aggressive augmentation (v3) disrupted key mildew texture features and caused unstable training behaviour.
The milder strategy (v3_mild) improved stability but still led to reduced accuracy compared with v1 and v2.

These findings guided the selection of **v1** as the final model, while also defining a clear roadmap for Phase 2 improvements focused on domain-specific augmentation and expanded field-data collection.

---

# **Dashboard Design**

The Streamlit dashboard serves as the central interface for exploring the project’s analytical results, validating model performance, and conducting real-time mildew detection. Each page maps directly to one or more business requirements.

## Design Principles

The dashboard was built around the following principles:

* **Clarity:** Present only the most relevant information per page
* **Transparency:** Show descriptive evidence, visual patterns, and technical metrics
* **Usability:** Fast loading, mobile/tablet compatibility, and logical navigation
* **Consistency:** Uniform page layout and clear section headers
* **Traceability:** Each page reflects a clear stage of the analytical pipeline

---

## App Pages Overview

The following subsections describe each dashboard page in detail.

---

### Navigation

The sidebar navigation allows users to switch between the main pages of the application.

![Dashboard Navigation](assets/images/page_navigation.png)

---

### Project Summary Page

Provides a high-level overview of the project’s objectives, dataset characteristics, hypotheses, and key findings.  
This page serves as the entry point for decision-makers and offers a concise narrative of the entire analytical workflow.

![Project Summary Page](assets/images/project_summary.png)

---

### Visual Analysis Page (Addresses BR1)

Displays the core visual analytics used to determine whether healthy and infected leaves can be distinguished:

* Class-average images
* Pixel‑level variability maps
* RGB histograms
* Class-difference maps
* Image montages

These visualizations help users understand distinguishing features and support early-stage disease-recognition training.

**Visual Analysis – Overview**

![Visual Analysis Overview](assets/images/visual_analysis_overview.png)

**Class Averages**

![Class Averages](assets/images/visual_analysis_class_av.png)

**Per-Pixel Variability**

![Per-Pixel Variability](assets/images/visual_analysis_per_pixel_var.png)

**Class Difference Map**

![Class Difference Map](assets/images/visual_analysis_class_diff.png)

**Image Montages**

![Image Montages](assets/images/visual_analysis_image_montages.png)

**Normalised RGB Histograms**

![RGB Histograms](assets/images/visual_analysis_normalized_rgd_histograms.png)

---

### Leaf Health Detector Page (Addresses BR2 & BR3)

Enables operational, real-time use of the model. Users can:

* Upload single or multiple leaf images
* Obtain instant predictions
* View results directly in the interface
* Download all predictions as a CSV file

This page supports daily orchard inspections and ensures consistent, reproducible evaluations under field conditions.

**Leaf Health Detector – Overview**

![Leaf Health Detector Overview](assets/images/prediction_overview.png)

**Leaf Health Detector – Example Results**

![Leaf Health Detector Results](assets/images/prediction_results.png)

---

### Hypotheses & Validation Page

Presents the full experimental validation for all three hypotheses. Each plot, statistical comparison, and interpretation is displayed in a transparent, structured manner, reinforcing confidence in the model’s analytical foundations.

**Hypotheses – Overview**

![Hypotheses Overview](assets/images/hypotheses_overview.png)

**Hypothesis H1 – Texture & Colour Variability**

![Hypothesis H1](assets/images/hypotheses_h1.png)

**Hypothesis H2 – Input Size Impact**

![Hypothesis H2](assets/images/hypotheses_h2.png)

**Hypothesis H3 – Data Augmentation Impact**

![Hypothesis H3](assets/images/hypotheses_h3.png)

**Training Curves (Diagnostics)**

![Hypotheses Training Curves](assets/images/hypotheses_training_curves.png)

---

### ML Performance Metrics Page

Provides in-depth diagnostic information, including model configuration, learning behaviour, confusion matrices, and interpretation of results.
This page supports internal audits, technical reviews, and model-governance workflows.

**ML Performance – Overview**

![ML Performance Metrics Overview](assets/images/technical_overview.png)

**Data Splits and Model Versions**

![Data Splits and Model Versions](assets/images/technical_data_splits.png)

**Training Configuration and Learning Behaviour**

![Training & Learning Behaviour](assets/images/technical_training_learning.png)

**Test Metrics and Confusion Matrices**

![Test Metrics and Confusion Matrices](assets/images/technical_test_confusion.png)

**Interpretation, Reproducibility and Next Steps**

![Interpretation, Reproducibility and Next Steps](assets/images/technical_interpretation.png)

---

## Business Relevance

The dashboard consolidates all analytical components into a single, accessible interface.
It enables Farmy & Foods to:

* Trace the full analytical pipeline end-to-end
* Onboard new agronomists more efficiently
* Rely on a consistent diagnosis tool across multiple regions
* Plan future enhancements with a clearly structured foundation

Its modular architecture ensures that new model versions, additional diseases, or updated datasets can be integrated with minimal development overhead.

---

# **Deployment**

The deployment strategy ensures that Farmy & Foods can reliably access and test the system under realistic field conditions while maintaining scalability for future operational use. Render.com was selected as the deployment platform due to its stability, performance, and seamless integration with GitHub.

## Deployment Platform – Why Render?

The project was initially set up for deployment on Heroku. However, Render.com provided several advantages more aligned with the company’s needs:

* **Container-based infrastructure** ensuring reproducibility and predictable performance
* **Higher performance** on CPU-bound inference workloads
* **Clear upgrade paths** for scaling or adding GPU support in future phases

These characteristics make Render a suitable choice for an agricultural environment that demands reliability and predictable performance.

## Deployment Workflow

1. **Repository Connection**
   The GitHub repository was linked to a new Render Web Service, enabling continuous deployment.

2. **Environment Setup**
   Render automatically installed all Python dependencies from `requirements.txt`. The lightweight CNN model kept provisioning fast and resource-efficient.

3. **Automated Rebuilds**
   Any push to the main branch triggers an automatic rebuild and redeployment, ensuring that improvements flow seamlessly into the live application.

## Technical Considerations

* All file paths within the application use relative referencing, ensuring compatibility across environments.
* Heroku-specific files (`setup.sh`, `.slugignore`, `Procfile`) were removed to avoid unnecessary configuration overhead and ensure clean deployment on Render.
* The app was tested across desktop and mobile browsers to verify performance and layout consistency.

## Business Relevance

Through deployment on Render, Farmy & Foods gains:

* Immediate access to the system from any location
* Stable environment for internal testing and validation
* Shared platform for quality assurance reviews
* Scalable foundation for future disease monitoring solutions

The deployment approach ensures that the system is not only a research artifact but a functional tool ready for operational integration.

---

# **Technologies & Libraries**

The project leverages mature, widely adopted technologies from the Python machine-learning ecosystem.
This ensures reliability, maintainability, and smooth integration across both research workflows and production environments.

## Core Technologies

| Library / Tool | Version | Purpose |
|----------------|---------|---------|
| **Python** | 3.12.1 | Primary programming language |
| **TensorFlow (CPU)** | 2.16.1 | Model development, training, and inference |
| **Keras (tf.keras)** | 2.16.1 | High-level API for building, training, and managing the CNN architecture |
| **NumPy** | 1.26.4 | Numerical computing and array operations |
| **Pandas** | 2.2.2 | Data loading, cleaning, and tabular processing |
| **scikit-learn** | 1.5.2 | Evaluation metrics, model validation, preprocessing utilities |
| **scikit-image** | 0.24.0 | Texture analysis (GLCM features) and image utilities |
| **Matplotlib** | 3.9.2 | Data visualization and exploratory analysis |
| **Pillow (PIL)** | 10.4.0 | Image processing (loading, resizing, formatting) |
| **Streamlit** | 1.40.2 | Dashboard development and deployment |

---

# **Testing**

---

## Manual Testing

All manual tests were performed on the deployed Streamlit dashboard in Chrome (desktop/tablet), Safari (mobile), and Firefox (desktop). The goal was to verify system stability, correct functionality, and alignment with Business Requirements BR1 and BR2.

---

### Platform & Access

passed | **Access the deployed dashboard** so that I can **use the system across devices**.
|:---:|:---|
|&check;| Site loads successfully on desktop.
|&check;| Site loads successfully on mobile.
|&check;| Site loads successfully on a tablet.
|&check;| No missing assets: images, plots, CSS and Streamlit components all load as expected.
|&check;| Browser refresh does not break any page.
|&check;| Responsive layout works on desktop, tablet, and mobile.

passed | **Global navigation via sidebar** so that I can **reach all sections quickly**.
|:---:|:---|
|&check;| Sidebar navigation visible on all pages.
|&check;| Page selection updates content instantly.
|&check;| Repeated switching between all pages causes no layout or state issues.

---

### Project Summary Page

passed | **Understand the project context** so that I can **orient myself before exploring details**.
|:---:|:---|
|&check;| Project Summary page loads from sidebar.
|&check;| Disease description, dataset info, and BR1/BR2 sections are readable.
|&check;| Kaggle link in the dataset section opens correctly.
|&check;| v1 Test Accuracy metric loads when `evaluation_report.json` is present.
|&check;| README link at bottom opens the project’s GitHub README.

---

### BR1 - Visual Analysis Page (Visual Study)

passed | **Review visual study findings** so that I can **understand how healthy and infected leaves differ visually**.
|:---:|:---|
|&check;| Visual Analysis page loads from sidebar.
|&check;| Expander "Class Averages" shows healthy & infected average images correctly.
|&check;| Expander "Per-Pixel Variability" shows both variability maps.
|&check;| Expander "Class Difference Map" displays the difference image.
|&check;| Expander "Image Montages" shows both montage images.
|&check;| Expander "Normalized RGB Histograms" shows RGB histograms correctly.

---

### Hypotheses & Validation (H1–H3)

passed | **View analytical reasoning** so that I can **understand how each hypothesis was tested and concluded**.
|:---:|:---|
|&check;| Hypotheses & Validation page loads from sidebar.
|&check;| H1 section: statistical tables + texture plots load correctly.
|&check;| H2 section: v1 vs v2 accuracy metrics, comparison tables, and confusion matrix load.
|&check;| H3 section: v1 vs v3_mild metrics, comparison tables, and plots load.
|&check;| Training Curves expander displays curves for v1, v2, v3, v3_mild.

---

### BR2 - Leaf Health Detector (Prediction Page)

passed | **Upload images** so that I can **receive a classification of healthy vs powdery mildew**.
|:---:|:---|
|&check;| Leaf Health Detector page loads from sidebar.
|&check;| Page explains BR2 and the prediction logic.
|&check;| Uploading a valid JPG/PNG runs predictions and shows a formatted results table.
|&check;| Uploading multiple images (≤25) produces a table with one row per file.
|&check;| Uploading more than 25 files produces a warning and only the first 25 are processed.
|&check;| Invalid file types rejected by the file uploader.
|&check;| Download button successfully downloads `predictions.csv`.

---

### ML Performance Metrics Page

passed | **Review model training and evaluation results** so that I can **assess the system’s predictive quality**.
|:---:|:---|
|&check;| ML Performance Metrics page loads from sidebar.
|&check;| "Data splits and model versions" expander displays dataset info and version explanations.
|&check;| "Training configuration and learning behaviour" shows all learning curves.
|&check;| "Test metrics and confusion matrices" shows the metrics table and all confusion matrices.
|&check;| "Interpretation, reproducibility and next steps" displays conclusions and limitations clearly.

---

## Validation

All Streamlit application files and Jupyter notebook Python blocks were validated using the [CI Python Linter](https://pep8ci.herokuapp.com).
This ensured full compliance with PEP8 stylistic conventions, consistent formatting, and clean, maintainable code across the entire project.

The validation process covered:
* All .py files within the Streamlit dashboard
* All Python cells contained in the Jupyter notebooks
* All modules inside the `src` directory (`data_management.py` and `paths.py`)

No critical issues were identified. Minor warnings were addressed during development, resulting in a codebase that adheres to industry-standard Python formatting and readability guidelines.

---

# **Unfixed Bugs**

At the time of submission, no unresolved bugs or functional issues are known.
All dashboard components, model-loading routines, file paths, and prediction functionalities were tested in both local and deployed environments (Render.com) and operated without errors.

---

# **Known Limitations**

While the current solution achieves strong performance and meets the stated business requirements, several limitations remain that must be addressed as Farmy & Foods scales the system across additional orchards, climates, and operational contexts.

## Current Limitations

**1. Limited Environmental Diversity**
The dataset contains mostly controlled lighting conditions and consistent backgrounds.
Real-world cherry leaf images may include shadows, variable sunlight, soil backgrounds, moisture artifacts, or partial occlusions.

**2. Sensitivity to Simple Augmentation**
Initial augmentation experiments (v3 and v3_mild) reduced model performance.
This highlights the need for carefully calibrated, domain-specific augmentation strategies that reflect real orchard variability.

**3. Narrow Disease Scope**
The current system distinguishes only between healthy and mildew‑infected leaves.
Farmy & Foods long-term requirements include detecting additional diseases, stress factors, and nutrient deficiencies.

**4. No Integration with Operational Systems**
The model currently runs as a standalone tool and is not yet integrated with orchard-management systems or treatment-recommendation workflows.

---

# **Future Work**

Farmy & Foods has defined a clear roadmap for further development:

**1. Dataset Expansion**
Collect additional field images across regions, seasons, weather conditions, and camera types to improve robustness under real orchard variability.

**2. Advanced Augmentation Techniques**
Develop domain-specific augmentation strategies such as adaptive brightness normalization, color-temperature adjustments, targeted blur, and controlled occlusion simulation to better reflect real-world imaging conditions.

**3. Multi‑Disease Classification**
Extend the model to detect additional diseases, stress factors, and nutrient deficiencies, enabling more comprehensive diagnostic support.

**4. API Integration**
Develop REST endpoints to integrate predictions with Farmy & Foods orchard-management systems and enable automated treatment workflows.

---

# **Local Setup Instructions**

The project can be executed locally for development, experimentation, or future extension.
The following steps describe how to prepare a clean and reproducible environment.

**Clone the Repository**

```bash
git clone https://github.com/ksstrat/milestone-project-5.git
cd milestone-project-5
```

**Create and Activate a Virtual Environment**

Make sure that your local Python version matches the version specified in the project (Python 3.12).

```bash
python -m venv venv
```

**Activation**

- **macOS / Linux**
  ```bash
  source venv/bin/activate
  ```
- **Windows**
  ```bash
  venv\Scripts\activate
  ```

**Install Dependencies**

```bash
pip install -r requirements.txt
```

This installs all required libraries including TensorFlow (CPU), Streamlit, Pillow, and scikit-learn.

**Launch the Application**

```bash
streamlit run app.py
```

Streamlit will open the dashboard in your browser at:

```
http://localhost:8501
```

## Notes

* The project uses relative paths. **Do not modify folder names or project structure.**
* The application runs on the CPU version of TensorFlow. No GPU setup is required.
* To run Streamlit on a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

# **Credits**

## General References

This project was developed with guidance from the Code Institute’s course material, assessment guidelines, and instructional walkthroughs, including the “Malaria Walkthrough Project”.

The cherry leaf dataset used in this project is provided by Code Institute via Kaggle:
[Cherry Leaves Dataset - Code Institute (Kaggle)](https://www.kaggle.com/codeinstitute/cherry-leaves)

Throughout the development process, official documentation played a central role, including:

* Python Documentation
* TensorFlow & Keras API Documentation
* scikit-image Documentation (GLCM features)
* scikit-learn Documentation
* Streamlit Documentation

These resources served as primary references for implementation details, model design, and best practices.

## Acknowledgements

* Thank you to my educators and mentors for their guidance throughout the programme.
* Special thanks to my family for their continued support and encouragement.

[Back to Top](#table-of-contents)