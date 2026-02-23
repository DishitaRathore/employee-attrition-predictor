import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('employee_data.csv')

    features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'Department', 'JobSatisfaction', 'OverTime']
    X = df[features]

    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encoding the categorical features (Department and OverTime)
    X_encoded = pd.get_dummies(X, columns=['Department', 'OverTime'])

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluating the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on Test Data: {accuracy:.2f}")

    joblib.dump(model, 'attrition_model.pkl')
    joblib.dump(list(X_train.columns), 'model_columns.pkl')
    print("Success! 'attrition_model.pkl' and 'model_columns.pkl' saved.")

if __name__ == "__main__":
    main()