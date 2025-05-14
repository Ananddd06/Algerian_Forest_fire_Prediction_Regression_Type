# Algerian Forest Fire Prediction 🌲🔥

## About the Dataset 📊

### Context

The Algerian Forest Fire dataset contains weather and environmental attributes that influence the occurrence of forest fires. The data includes factors like temperature, humidity, wind speed, and various indices from the Fire Weather Index (FWI) system. The goal is to predict whether a forest fire will occur based on these factors.

### Content

The dataset has observations of weather data and FWI components collected over the summer months of 2012. The target variable represents whether a fire occurred (`Fire`) or not (`Not Fire`).

## Attribute Information 📝

- **Date**: (DD/MM/YYYY) - Day, month ('June' to 'September'), and year (2012).
- **Weather data observations**:
  - **Temp**: Maximum temperature (°C) at noon (22 to 42).
  - **RH**: Relative Humidity (%) (21 to 90).
  - **Ws**: Wind speed (km/h) (6 to 29).
  - **Rain**: Total rainfall in mm (0 to 16.8).
- **FWI Components**:
  - **FFMC (Fine Fuel Moisture Code)**: Index from 28.6 to 92.5.
  - **DMC (Duff Moisture Code)**: Index from 1.1 to 65.9.
  - **DC (Drought Code)**: Index from 7.9 to 195.1.
  - **ISI (Initial Spread Index)**: Index from 0 to 56.1.
  - **BUI (Buildup Index)**: Index from 0.0 to 200.7.
  - **Fire**: Target variable, indicating fire occurrence (`Fire` or `Not Fire`).

## Installation 🔧

To get started with the Algerian Forest Fire prediction model, follow the steps below:

### Prerequisites:

- Python >= 3.7
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`

### Install Dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

```
/Algerian_Forest_Fire_Prediction
├── app.py                # Flask web application
├── Model_file            # Contains the saved model (.pkl)
├── index.html            # HTML template for the web form
├── requirements.txt      # List of dependencies
├── README.md             # Project overview

```

## How the Model Works 🤖

### Feature Engineering 🧑‍💻

We performed various steps of feature engineering to prepare the dataset for the model:

1. **Handling Missing Values**:

   - We replaced any missing values with the median of the respective columns.

2. **Feature Scaling**:
   - The features such as temperature, humidity, wind speed, and FWI components were scaled using `StandardScaler` to ensure the model performs optimally.

### Model Selection 🔍

The model was selected based on its ability to predict fire occurrences accurately. We used **Linear Regression** with **Ridge** and **Lasso Regularization** techniques to handle the multicollinearity between the features.

- **Ridge Regression**: Regularizes the model by penalizing large coefficients.
- **Lasso Regression**: Applies L1 regularization, which can eliminate irrelevant features by setting their coefficients to zero.

### Model Training 📚

The model was trained using the processed dataset and evaluated using cross-validation to select the best regularization parameter. After tuning the hyperparameters, we saved the trained model into a `.pkl` file.

### Saving and Loading the Model:

We saved the trained model with the scaler included in a pipeline for easier predictions during deployment.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Define and fit the model pipeline
model_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model_pipeline.fit(X_train, y_train)

# Save the trained model to disk
import joblib
joblib.dump(model_pipeline, 'scaled_model.pkl')
```

## Result Interpretation 📝

The model returns a prediction of either **Fire** or **Not Fire**. These predictions are based on the environmental conditions input by the user.

## Running the Application 💻

1. Save the trained model as `scaled_model.pkl`.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Access the app at http://127.0.0.1:5000/ in your browser.
4. Enter the necessary input data and receive a prediction about forest fire occurrence.

## Conclusion 🎯

This project provides a web-based tool to predict forest fire occurrences based on environmental conditions. The trained machine learning model uses regression techniques with regularization, and the prediction is available via a Flask web application.
