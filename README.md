## PaySim Fraud Detection Workflow

This project builds and evaluates machine learning models to detect fraudulent mobile money transactions using the PaySim dataset. The workflow demonstrates an end-to-end data science pipeline including exploratory analysis, model training, evaluation, explainability, and deployment through an interactive Streamlit application.

The application allows users to:

explore key descriptive analytics of the dataset

compare model performance across multiple algorithms

generate fraud predictions for custom transactions

view SHAP explainability visualizations for model interpretation

Project Structure
.
├── streamlit_app.py
├── Prajakta_Kurulkar_HW1_final.ipynb
├── requirements.txt
├── README.md
└── models/
├── preprocess.pkl
├── lr.pkl
├── tree.pkl
├── rf.pkl
├── xgb.pkl
├── mlp.keras
├── model_comparison.csv
├── shap_summary.png
├── shap_bar.png
├── shap_waterfall.png
└── best_params.json

The models/ folder contains the trained models, preprocessing pipeline, evaluation outputs, and visualization files used by the Streamlit application.

How to Run the Application
1. Clone the repository
git clone <your-github-repository-url>
2. Navigate to the project folder
cd <repository-folder>
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run streamlit_app.py

The application will open in your browser at:

http://localhost:8501

Notebook

The notebook Prajakta_Kurulkar_HW1_final.ipynb contains the complete data science workflow including:

exploratory data analysis
feature engineering
model training and comparison
SHAP explainability analysis
Deployed Application

The deployed Streamlit app provides an interactive interface for exploring the analysis and generating fraud predictions.

