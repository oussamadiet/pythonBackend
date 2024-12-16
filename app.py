from flask import Flask, request, jsonify
from model.predict import make_predictions

app = Flask(__name__)

@app.route('/recommend-meal-plan', methods=['POST'])
def recommend_meal_plan():
    user_data = request.get_json()
    recommendation = make_predictions(user_data)
    return jsonify(recommendation)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('127.0.0.1', 5000, app, use_debugger=True, use_reloader=True)
