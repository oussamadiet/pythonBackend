import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Paths
input_user_data_path = './static/user_data.csv'
input_meal_plan_data_path = './static/Meal_Plan_Data.csv'
output_dir = './predata'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
user_data_df = pd.read_csv(input_user_data_path)
meal_plan_data_df = pd.read_csv(input_meal_plan_data_path)

# Calculate caloric needs and protein requirements
def calculate_caloric_needs(weight, activity_level, goal):
    activity_factors = {'sedentary': 1.2, 'moderate': 1.55, 'active': 1.725}
    base_calories = weight * 24
    caloric_needs = base_calories * activity_factors.get(activity_level.lower(), 1.2)
    return caloric_needs + 500 if goal.lower() == 'muscle gain' else caloric_needs - 500 if goal.lower() == 'weight loss' else caloric_needs

def calculate_protein_needs(weight, goal):
    return weight * 2.0 if goal.lower() == "muscle gain" else weight * 1.2 if goal.lower() == "weight loss" else weight * 1.5

# Add calculated columns to user data
user_data_df['caloric_needs'] = user_data_df.apply(lambda x: calculate_caloric_needs(x['Weight (kg)'], x['Activity Level'], x['Dietary Goal']), axis=1)
user_data_df['protein_needs'] = user_data_df.apply(lambda x: calculate_protein_needs(x['Weight (kg)'], x['Dietary Goal']), axis=1)

# Encode categorical variables
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_goal = LabelEncoder()
user_data_df['Gender_encoded'] = le_gender.fit_transform(user_data_df['Gender'])
user_data_df['Activity_Level_encoded'] = le_activity.fit_transform(user_data_df['Activity Level'])
user_data_df['Dietary_Goal_encoded'] = le_goal.fit_transform(user_data_df['Dietary Goal'])

# Merge user data and meal plans
combined_data = user_data_df.merge(meal_plan_data_df, how='cross')

# Filter based on caloric needs and dietary goal
combined_data = combined_data[((combined_data['Dietary Goal'].str.lower() == 'weight loss') & 
                               (combined_data['Total Calories'] <= combined_data['caloric_needs'])) |
                              ((combined_data['Dietary Goal'].str.lower() == 'muscle gain') & 
                               (combined_data['Total Calories'] >= combined_data['caloric_needs']))]

# Define features and target
features = ['Age', 'Weight (kg)', 'Height (cm)', 'Gender_encoded', 'Activity_Level_encoded', 'Dietary_Goal_encoded', 'caloric_needs', 'protein_needs']
target = 'Meal Plan Name'

# Handle class imbalance
X, y = combined_data[features], combined_data[target]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save datasets
train_data = pd.concat([X_train, y_train], axis=1)
validation_data = pd.concat([X_validation, y_validation], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
print("Preprocessing complete.")
