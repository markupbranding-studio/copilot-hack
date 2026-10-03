# Create a base Flask server

import csv
import pickle
from functools import lru_cache

from flask import Flask, request, jsonify

app = Flask(__name__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model from pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@lru_cache(maxsize=1)
def get_airports():
    """Load and cache the airport catalog so repeated requests do not re-read the CSV."""
    with open('airports.csv', newline='', encoding='utf-8') as airport_file:
        reader = csv.DictReader(airport_file)
        airports = [{
            'id': int(row['OriginAirportID']),
            'name': row['OriginAirportName']
        } for row in reader]
    return tuple(sorted(airports, key=lambda airport: airport['name']))

# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    # Store day_of_week as int
    day_of_week = int(request.args.get('day_of_week'))
    airport_id = int(request.args.get('airport_id'))
    prediction = model.predict_proba([[day_of_week, airport_id]])[0]
    
    # Split prediction string by space
    prediction = str(prediction).split(' ')

    # store first value from prediction as certainty, and remove the first character
    certainty = float(prediction[0][2:])

    # store second value from prediction as delay, and remove the last character
    delay = float(prediction[1][:-1])

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})

# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    return jsonify(list(get_airports()))

if __name__ == '__main__':
    app.run(debug=True)
