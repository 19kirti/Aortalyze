## Heart Disease Prediction System (Aortalyze)
Heart disease is one of the leading causes of death worldwide. Early prediction and diagnosis can significantly reduce the risk of fatal outcomes. This project implements a Machine Learning-based Heart Failure Prediction System that predicts whether a person is at risk of heart disease based on medical features.


![Screenshot 2025-04-01 130458](https://github.com/user-attachments/assets/17866da0-2f09-4ddd-98eb-067e5d5e38e3)


## Features
1. Takes health-related inputs (age, cholesterol, blood pressure, etc.)
2. Predicts risk level (0 = No Risk, 1 = High Risk)
3. Uses trained ML models on real-world patient data
4. User-friendly Streamlit web interface
5. Real-time predictions without model storage

## Dataset Overview
Numerical features:
!. Age
2. RestingBP (blood pressure)
3. Cholesterol
4. MaxHR (maximum heart rate)
5. Oldpeak

Categorical features:
1. Sex
2. ChestPainType
3. RestingECG
4. ExerciseAngina
5. ST_Slope

Target variable:
HeartDisease (1 = presence, 0 = absence)

## Data PreProcessing 
1. Label Encoding of categorical variables
2. Feature Scaling using StandardScaler
3. Train-test split (80-20 ratio)
4. Duplicate check and null value handling

## Model Building
Logistic Regression:-
1. Suitable for binary classification
2. Easy to interpret
3. Predicts probability using sigmoid function
4. Helps identify influential features

Random Forest Classifier (Primary Model):-
1. Ensemble of multiple decision trees
2. Captures complex patterns and nonlinear relationships
3. More accurate and robust
4. Final deployed model

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
1. Dataset providers (Kaggle)
2. Streamlit for the web framework
3. Scikit-learn for machine learning tools
