

#  Customer Churn Classification (TensorFlow)

This project implements an **end-to-end Deep Learning pipeline** to predict **customer churn** using **TensorFlow/Keras**.  
It includes **data preprocessing, model training, hyperparameter tuning using Hyperband**, and an **interactive Streamlit application** for inference.

---

##  Project Overview

Customer churn refers to customers who stop using a company’s service.  
The objective of this project is to **predict whether a customer is likely to churn (Yes/No)** based on historical customer attributes.

### Why churn prediction matters
- Helps businesses proactively retain customers  
- Reduces revenue loss  
- Enables targeted retention and marketing strategies  

This is formulated as a **binary classification problem**.

---

## Repository Structure

Files and folders at the repository root:

```text
├── data/
│   └── Churn_Modelling.csv          # Raw churn dataset
│
├── notebook/
│   ├── experiments.ipynb            # Model experiments & hyperparameter tuning
│   ├── prediction.ipynb             # Model testing & predictions
│
├── churn_classification_model.h5    # Trained Keras model
├── LabelEncoder_gender.pkl          # Label encoder for Gender
├── OneHotEncoder_geo.pkl            # One-hot encoder for Geography
├── StandardScaler.pkl               # Scaler for numerical features
│
├── app.py                           # Streamlit app for interactive inference
├── kt_logs/                         # KerasTuner (Hyperband) logs (not pushed to GitHub)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

```

---

## 🛠️ Technical Stack

### Core Libraries
- Python 3.10.14 
- TensorFlow / Keras
- KerasTuner (Hyperband)
- NumPy, Pandas
- Scikit-learn
- Streamlit

### Deep Learning Components
- Fully connected (Dense) neural networks
- ReLU activation (hidden layers)
- Sigmoid activation (output layer)
- Binary Crossentropy loss
- Adam optimizer
- Dropout regularization
- EarlyStopping callback

---

## Setup & Usage

### Install dependencies

This project targets Python 3.10.14 and uses typical data science libraries. Install dependencies with:

```bash
pip install -r requirements.txt
```

If you plan to run on Apple Silicon with TensorFlow-Metal, follow Apple/TensorFlow guidance for environment setup.

### Running the Streamlit app (inference)

```bash
streamlit run app.py
```

---

## Reproduce Training & Experiments
- Use Jupyter notebooks under notebook/
    - experiments.ipynb → model training & hyperparameter tuning
    - prediction.ipynb → prediction testing & validation

### Data Preprocessing
Preprocessing used during training and required at inference:

- Encode categorical features (label encoding for `Gender`, one-hot for `Geography`)
- Scale numerical features with `StandardScaler` (saved as `StandardScaler.pkl`)

Make sure the same encoders and scaler are used in the Streamlit app to avoid data mismatch.

### Model & Training

- Model type: Fully-connected neural network (Keras Sequential/Functional)
- Output: Sigmoid activation for binary probability
- Loss: Binary crossentropy
- Optimizer: Adam (learning rate tuned)
- Regularization: Dropout (tuned) and EarlyStopping on `val_loss`

Typical training flow in the notebooks/scripts:
1. Load and preprocess data (encoders and scaler)
2. Define model-building function (for KerasTuner)
3. Run Hyperband tuner to find best hyperparameters
4. Retrain the final model with best hyperparameters and restore best weights


### Hyperparameter tuning

KerasTuner (Hyperband) is used to tune:
- Number of hidden layers (1–3)
- Units per hidden layer (e.g., 32, 64, 128)
- Dropout rate
- Learning rate

Objective: minimize validation loss (`val_loss`). Early stopping is used to avoid overfitting and reduce wasted compute.

Tuning results and trials are saved under `kt_logs/`. # not uploading over github

### Model Testing

### prediction.ipynb
- Loads trained model
- Runs predictions on sample inputs
- Verifies preprocessing and output probabilities

---

## Streamlit app

- `app.py` provides a simple UI for entering customer attributes and viewing churn probability.
- The app loads the saved encoders, scaler, and model from the repository root. Ensure those files are present before launching Streamlit.

--- 

## Notes & Next steps

- Validation loss is used as the primary selection metric; consider additional metrics (AUC, precision/recall) for business relevance.
- For production, consider exporting the model in a serialized format (SavedModel / TF Serving) and packaging preprocessing as a pipeline (e.g., sklearn Pipeline or custom wrapper).
- Add unit tests for the preprocessing pipeline and a small smoke test for the Streamlit app's prediction function.

---
