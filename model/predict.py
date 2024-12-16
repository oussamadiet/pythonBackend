import os
import pandas as pd
import joblib
import random

# Paths to the model, label encoder, and preprocessor
model_path = './model/diet_recommendation_modelyo.pkl'
label_encoder_path = './model/label_encoderyo.pkl'
preprocessor_path = './model/preprocessor.pkl'

# Load the trained model, label encoder, and preprocessor
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)
preprocessor = joblib.load(preprocessor_path)

# Load meal plan details
meal_plan_data_path = './static/Meal_Plan_Data.csv'
meal_plan_data = pd.read_csv(meal_plan_data_path)

# Global list to keep track of recently recommended plans
# We'll store a small list and avoid immediately repeating these.
recent_plans = []

def calculate_caloric_and_protein_needs(data):
    weight = data['Weight (kg)']
    activity_level = data['Activity Level'].lower()
    goal = data['Dietary Goal'].lower()

    activity_factors = {'sedentary': 1.2, 'moderate': 1.55, 'active': 1.725}
    base_calories = weight * 24
    caloric_needs = base_calories * activity_factors.get(activity_level, 1.2)
    if goal == 'muscle gain':
        caloric_needs += 500
    elif goal == 'weight loss':
        caloric_needs -= 500

    protein_needs = round(weight * 2.0 if goal == "muscle gain" else weight * 1.2 if goal == "weight loss" else weight * 1.5)
    data['caloric_needs'] = caloric_needs
    data['protein_needs'] = protein_needs
    return data

def encode_categorical_fields(data):
    data['Gender_encoded'] = 1 if data['Gender'].lower() == 'male' else 0
    activity_mapping = {'sedentary': 0, 'moderate': 1, 'active': 2}
    data['Activity_Level_encoded'] = activity_mapping.get(data['Activity Level'].lower(), 0)
    goal_mapping = {'weight loss': 0, 'maintenance': 1, 'muscle gain': 2}
    data['Dietary_Goal_encoded'] = goal_mapping.get(data['Dietary Goal'].lower(), 1)
    return data

def preprocess_input(data):
    data = calculate_caloric_and_protein_needs(data)
    data = encode_categorical_fields(data)
    input_df = pd.DataFrame([data])
    input_processed = preprocessor.transform(input_df)
    return input_processed, data['caloric_needs'], data['protein_needs']

def get_meal_plan_details(predicted_plan_name):
    plan_details = meal_plan_data[meal_plan_data['Meal Plan Name'].str.lower() == predicted_plan_name.lower()].to_dict('records')
    if plan_details:
        plan_info = plan_details[0]
        return {
            "Total Calories": plan_info['Total Calories'],
            "Protein (g)": plan_info['Protein (g)'],
            "Carbohydrates (g)": plan_info['Carbohydrates (g)'],
            "Fat (g)": plan_info['Fat (g)'],
            "Meals": {
                "Meal 1 (Breakfast)": plan_info['Meal 1 (Breakfast)'],
                "Meal 2 (Lunch)": plan_info['Meal 2 (Lunch)'],
                "Meal 3 (Dinner)": plan_info['Meal 3 (Dinner)']
            }
        }
    return None

def choose_alternative_plan(original_plan):
    # Get all unique meal plans
    all_plans = meal_plan_data['Meal Plan Name'].unique()
    # Filter out the original plan and recently used plans
    candidates = [p for p in all_plans if p.lower() != original_plan.lower() and p not in recent_plans]

    if not candidates:
        # If no candidates available, just return the original plan (fallback)
        return original_plan
    else:
        # Choose a random alternative
        return random.choice(candidates)

def make_predictions(user_data):
    input_data, caloric_needs, protein_needs = preprocess_input(user_data)
    prediction_encoded = model.predict(input_data)
    predicted_meal_plan_names = label_encoder.inverse_transform(prediction_encoded)

    recommended_plans = []

    for plan_name in predicted_meal_plan_names:
        # Check if this plan was recently used
        if plan_name in recent_plans:
            # Choose a different plan to introduce variety
            plan_name = choose_alternative_plan(plan_name)

        plan_details = get_meal_plan_details(plan_name)
        if plan_details:
            recommended_plans.append({
                "meal_plan_name": plan_name,
                "details": plan_details
            })

    # Update recent_plans list: add the chosen plan(s) and keep the list small
    for rp in recommended_plans:
        used_plan = rp["meal_plan_name"]
        # Add this plan to recent_plans
        recent_plans.append(used_plan)
    # Limit the recent_plans memory to avoid growing indefinitely (e.g. keep last 10)
    if len(recent_plans) > 10:
        recent_plans.pop(0)

    return {
        "caloric_needs": caloric_needs,
        "protein_needs": protein_needs,
        "recommended_meal_plans": recommended_plans
    }
