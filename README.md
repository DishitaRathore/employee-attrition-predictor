## Employee Attrition Predictor

This project builds an end-to-end Machine Learning web application to predict employee attrition using Python and Streamlit. The model is trained on the IBM HR Analytics dataset to identify key factors that contribute to employee turnover, such as income, age, and overtime.

### Features

** Data Preprocessing: Handling categorical variables using One-Hot Encoding and cleaning dataset features.

** Machine Learning Model: Utilizes a Random Forest Classifier for high-accuracy predictions.

** Interactive Web Interface: Built with Streamlit to allow users to input employee details and get real-time predictions.

** Probability Scoring: Displays not just the prediction (Yes/No) but also the probability percentage of attrition.

** Deployment: Fully deployed and accessible via Hugging Face Spaces.

### How to Run

1. Clone the repository or download the source code.
2. Open the project folder in VS Code or your preferred IDE.
3. Run the Streamlit application using the following command in the terminal:

'''bash
streamlit run app.py
'''

### Requirements
** Python 3.x
** pandas
** scikit-learn
** streamlit
** joblib

Install dependencies using:
'''bash
pip install -r requirements.txt
'''

### Dataset
** 'WA_Fn-UseC_-HR-Employee-Attrition.csv': Contains the IBM HR Analytics data used for training and testing the model.

### Visualizations & Interface
The web application provides:
Interactive Form: specific input fields for Age, Monthly Income, Department, Job Satisfaction, and Overtime.
Prediction Output: A dynamic status message indicating "High Risk" or "Low Risk" of attrition.
Confidence Score: A percentage showing how confident the model is in its prediction.

### Observations
Overtime Impact: Employees working overtime generally show a higher correlation with attrition.
Income Levels: Lower monthly income ranges are often associated with higher turnover risk.
Job Satisfaction: Low satisfaction scores are a strong indicator of potential attrition.
Model Performance: The Random Forest algorithm provides robust performance on this tabular dataset compared to simple logistic regression.

---

Feel free to explore the code or fork the repository to add more features!

