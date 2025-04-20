## Heart Disease Prediction System (Aortalyze)
Heart disease is one of the leading causes of death worldwide. Early prediction and diagnosis can significantly reduce the risk of fatal outcomes. This project implements a Machine Learning-based Heart Failure Prediction System that predicts whether a person is at risk of heart disease based on medical features.

## Features
Takes health-related inputs (age, cholesterol, blood pressure, etc.)
Predicts risk level (0 = No Risk, 1 = High Risk)
Uses trained ML models on real-world patient data
User-friendly Streamlit web interface
Real-time predictions without model storage

## Dataset Overview
Numerical features:
Age
RestingBP (blood pressure)
Cholesterol
MaxHR (maximum heart rate)
Oldpeak

Categorical features:
Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope

Target variable:
HeartDisease (1 = presence, 0 = absence)

## Data PreProcessing 
Label Encoding of categorical variables
Feature Scaling using StandardScaler
Train-test split (80-20 ratio)
Duplicate check and null value handling

## Model Building
Logistic Regression:-
Suitable for binary classification
Easy to interpret
Predicts probability using sigmoid function
Helps identify influential features

Random Forest Classifier (Primary Model):-
Ensemble of multiple decision trees
Captures complex patterns and nonlinear relationships
More accurate and robust
Final deployed model

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
Dataset providers
Streamlit for the web framework
Scikit-learn for machine learning tools
