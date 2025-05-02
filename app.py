from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoders
model_path = 'Random_forest_model.pkl'
state_encoder_path = 'state_encoder.pkl'
season_encoder_path = 'season_encoder.pkl'
crop_encoder_path = 'crop_encoder.pkl'

# Load your files
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)
with open(state_encoder_path, 'rb') as state_file:
    state_encoder = pickle.load(state_file)
with open(season_encoder_path, 'rb') as season_file:
    season_encoder = pickle.load(season_file)
with open(crop_encoder_path, 'rb') as crop_file:
    crop_encoder = pickle.load(crop_file)

# Normalize encoder classes
state_classes = [state.strip().lower() for state in state_encoder.classes_]
season_classes = [season.strip().lower() for season in season_encoder.classes_]

# Decode for user-friendly display
state_decoder = {state.strip().lower(): state for state in state_encoder.classes_}
season_decoder = {season.strip().lower(): season for season in season_encoder.classes_}

@app.route('/')
def home():
    return render_template('index.html', states=list(state_decoder.values()), seasons=list(season_decoder.values()))

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        state_name = request.form['state'].strip().lower()
        season = request.form['season'].strip().lower()
        area = float(request.form['area'])
        crop_productivity = float(request.form['crop_productivity'])

        # Validate state and season
        if state_name not in state_classes:
            return f"Error: State '{state_name}' not recognized. Available states: {list(state_decoder.values())}"
        if season not in season_classes:
            return f"Error: Season '{season}' not recognized. Available seasons: {list(season_decoder.values())}"

        # Encode inputs
        state_encoded = state_encoder.transform([state_decoder[state_name]])[0]
        season_encoded = season_encoder.transform([season_decoder[season]])[0]

        # Prepare input for the model
        input_data = np.array([[state_encoded, season_encoded, area, crop_productivity]])

        # Predict the crop
        predicted_crop_encoded = rf_model.predict(input_data)[0]
        recommended_crop = crop_encoder.inverse_transform([predicted_crop_encoded])[0]

        return render_template('result.html', crop=recommended_crop, state=state_decoder[state_name], 
                               season=season_decoder[season].capitalize(), area=area, productivity=crop_productivity)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
