# 🌾Crop-recommendation-System
A Flask-based web application that recommends the most suitable crop to grow based on your state, season, area of cultivation, and productivity. It leverages a machine learning model (Random Forest) trained on historical agricultural data to provide reliable recommendations for farmers and agricultural planners.

## 🚀 Features

- Recommends crops using a pre-trained Random Forest classification model
- User-friendly interface for data input and prediction
- Inputs: State, Season, Area (in hectares), Productivity (tons/hectare)
- Real-time prediction with results displayed on the web page

## 📊 Dataset

The model is trained on agricultural data sourced from [data.gov.in](https://data.gov.in/), containing:

- **States** (e.g., Punjab, Maharashtra, etc.)
- **Seasons** (Kharif, Rabi)
- **Crop names**
- **Year-wise data from 1997 to 2015**
- **Area of cultivation (in hectares)**
- **Production (in tons)**

From this, **crop productivity** was calculated and used as a key feature. The dataset was cleaned and preprocessed, with string features encoded using LabelEncoder for model compatibility.

## 🛠 Technologies Used

- **Python**
- **Flask** – For building the web interface
- **Scikit-learn** – For model training and prediction
- **Pickle** – To save/load the model and label encoders
- **HTML** – For frontend design
