from flask import Flask, request, render_template, jsonify
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("=" * 80)
print("LOADING CATBOOST MODEL...")
print("=" * 80)

# Load the trained CatBoost model
model = CatBoostClassifier()
model_path = "catboost_emission_model.cbm"

if os.path.exists(model_path):
    model.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
else:
    print(f"ERROR: Model file not found at: {model_path}")


print("=" * 80)

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Define the exact features used during training (in order)
# These must match your training data exactly!
FEATURE_NAMES = [
    'VehMake',           # Categorical
    'VehCylinders',      # Numerical
    'VehFuelType',       # Categorical
    'AccHC',             # Numerical
    'AccCO',             # Numerical
    'AccCO2',            # Numerical
    'AccO2',             # Numerical
    'AccLambda',         # Numerical
    'AccRPM',            # Numerical
    'IdleHC',            # Numerical
    'IdleCO',            # Numerical
    'IdleCO2',           # Numerical
    'IdleO2',            # Numerical
    'IdleLambda',        # Numerical
    'IdleRPM',           # Numerical
    'VehicleAge',        # Numerical (derived)
    'CO_Ratio',          # Numerical (derived)
    'HC_Ratio',          # Numerical (derived)
    'Lambda_Deviation'   # Numerical (derived)
]

# Define categorical features - MUST MATCH TRAINING
# Based on error: VehCylinders was categorical during training
CATEGORICAL_FEATURES = ['VehMake', 'VehCylinders', 'VehFuelType']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_input(form_data):
    """
    Convert form data to model input format
    """
    try:
        # Extract values from form
        veh_make = [str(form_data.get('VehMake', 'TOYOTA'))]
        veh_cylinders = str(form_data.get('VehCylinders', '4'))  # Keep as string - it's categorical!
        veh_fuel_type = str(form_data.get('fuelType', 'Petrol'))
        acc_hc = float(form_data.get('AccHC', 0))
        acc_co = float(form_data.get('AccCO', 0))
        acc_co2 = float(form_data.get('AccCO2', 14.0))
        acc_o2 = float(form_data.get('AccO2', 0.5))
        acc_lambda = float(form_data.get('AccLambda', 1.0))
        acc_rpm = float(form_data.get('AccRPM', 500))
        idle_hc = float(form_data.get('IdleHC', 0))
        idle_co = float(form_data.get('IdleCO', 0))
        idle_co2 = float(form_data.get('IdleCO2', 13.5))
        idle_o2 = float(form_data.get('IdleO2', 0.8))
        idle_lambda = float(form_data.get('IdleLambda', 1.0))
        idle_rpm = float(form_data.get('IdleRPM', 100))
        vehicle_age = int(form_data.get('VehicleAge', 5))


        co_ratio = acc_co / (idle_co + 0.001) if idle_co > 0 else 0
        hc_ratio = acc_hc / (idle_hc + 0.001) if idle_hc > 0 else 0
        lambda_deviation = abs(acc_lambda - 1.0)


        features = {
            'VehMake': veh_make,
            'VehCylinders': veh_cylinders,  # Categorical string
            'VehFuelType': veh_fuel_type,
            'AccHC': acc_hc,
            'AccCO': acc_co,
            'AccCO2': acc_co2,
            'AccO2': acc_o2,
            'AccLambda': acc_lambda,
            'AccRPM': acc_rpm,
            'IdleHC': idle_hc,
            'IdleCO': idle_co,
            'IdleCO2': idle_co2,
            'IdleO2': idle_o2,
            'IdleLambda': idle_lambda,
            'IdleRPM': idle_rpm,
            'VehicleAge': vehicle_age,
            'CO_Ratio': co_ratio,
            'HC_Ratio': hc_ratio,
            'Lambda_Deviation': lambda_deviation
        }


        df = pd.DataFrame([features])


        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df, features

    except Exception as e:
        raise ValueError(f"Error preprocessing input: {str(e)}")

def get_top_factors(features, prediction_proba):

    factors = []

    acc_co = features['AccCO']
    acc_hc = features['AccHC']
    idle_co = features['IdleCO']
    vehicle_age = features['VehicleAge']
    lambda_dev = features['Lambda_Deviation']

    # Check AccCO
    if acc_co > 1.0:
        factors.append({
            'name': 'High AccCO Level',
            'impact': 'High',
            'value': f'{acc_co:.2f}% (Limit: 1.0%)',
            'status': 'danger',
            'shap_value': 0.3
        })
    elif acc_co > 0.8:
        factors.append({
            'name': 'AccCO Near Limit',
            'impact': 'Medium',
            'value': f'{acc_co:.2f}% (Limit: 1.0%)',
            'status': 'warning',
            'shap_value': 0.15
        })
    else:
        factors.append({
            'name': 'AccCO Within Limit',
            'impact': 'Positive',
            'value': f'{acc_co:.2f}% (Limit: 1.0%)',
            'status': 'success',
            'shap_value': -0.1
        })

    # Check AccHC
    if acc_hc > 200:
        factors.append({
            'name': 'High AccHC Emissions',
            'impact': 'High',
            'value': f'{acc_hc:.0f} ppm (Limit: 200 ppm)',
            'status': 'danger',
            'shap_value': 0.25
        })
    elif acc_hc > 150:
        factors.append({
            'name': 'AccHC Near Limit',
            'impact': 'Medium',
            'value': f'{acc_hc:.0f} ppm (Limit: 200 ppm)',
            'status': 'warning',
            'shap_value': 0.12
        })
    else:
        factors.append({
            'name': 'AccHC Within Limit',
            'impact': 'Positive',
            'value': f'{acc_hc:.0f} ppm (Limit: 200 ppm)',
            'status': 'success',
            'shap_value': -0.08
        })

    # Check IdleCO
    if idle_co > 0.5:
        factors.append({
            'name': 'High IdleCO Level',
            'impact': 'High',
            'value': f'{idle_co:.2f}% (Limit: 0.5%)',
            'status': 'danger',
            'shap_value': 0.2
        })
    elif idle_co > 0.3:
        factors.append({
            'name': 'IdleCO Moderate',
            'impact': 'Medium',
            'value': f'{idle_co:.2f}% (Limit: 0.5%)',
            'status': 'warning',
            'shap_value': 0.1
        })
    else:
        factors.append({
            'name': 'IdleCO Within Limit',
            'impact': 'Positive',
            'value': f'{idle_co:.2f}% (Limit: 0.5%)',
            'status': 'success',
            'shap_value': -0.05
        })

    # Check Vehicle Age
    if vehicle_age > 15:
        factors.append({
            'name': 'Old Vehicle',
            'impact': 'High',
            'value': f'{vehicle_age} years old',
            'status': 'danger',
            'shap_value': 0.2
        })
    elif vehicle_age > 10:
        factors.append({
            'name': 'Aging Vehicle',
            'impact': 'Medium',
            'value': f'{vehicle_age} years old',
            'status': 'warning',
            'shap_value': 0.1
        })
    else:
        factors.append({
            'name': 'Relatively New Vehicle',
            'impact': 'Positive',
            'value': f'{vehicle_age} years old',
            'status': 'success',
            'shap_value': -0.05
        })

    # Check Lambda Deviation
    if lambda_dev > 0.1:
        factors.append({
            'name': 'Lambda Out of Range',
            'impact': 'Medium',
            'value': f'Deviation: {lambda_dev:.2f}',
            'status': 'warning',
            'shap_value': 0.15
        })

    # Sort by impact and return top 4
    impact_order = {'High': 3, 'Medium': 2, 'Positive': 1}
    factors.sort(key=lambda x: impact_order.get(x['impact'], 0), reverse=True)

    return factors[:4]

def get_recommendation(prediction, fail_probability):

    if prediction == 0:  # Fail
        return (
            "This vehicle has a high probability of failing the emission test. "
            "We recommend maintenance before testing to avoid additional costs and delays."
        )
    elif fail_probability > 0.3:  # Borderline
        return (
            "⚡ This vehicle has moderate risk. While likely to pass, consider a pre-check "
            "to ensure all emission values are optimal."
        )
    else:  # Pass
        return (
            "✓ This vehicle is likely to pass the emission test. "
            "All major parameters appear to be within acceptable ranges."
        )

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
       # Pass empty form_data on first load
    return render_template('index.html', form_data={}, result=None)

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get form data
        form_data = request.form.to_dict()
        print(f"form_data: {form_data}")
        # Preprocess input
        df, features = preprocess_input(form_data)

        print(f"  df @@@@@@@@@@@@@: {df}")

        print(f"  features @@@@@@@@@@@@@: {features}")

        pool = Pool(
            data=df,
            cat_features=CATEGORICAL_FEATURES
        )

        # Make prediction using the Pool
        prediction = model.predict(pool, prediction_type='Class')
        prediction_proba = model.predict(pool, prediction_type='Probability')
        print(f"prediction: {prediction}")
        print(f"prediction_proba: {prediction_proba}")

        # Extract probabilities
        if len(prediction_proba.shape) > 1:
            pass_probability = prediction_proba[0][1]
            fail_probability = prediction_proba[0][0]
        else:
            fail_probability = prediction_proba[1]
            pass_probability = 1 - fail_probability

        print(f"  Raw fail_probability: {fail_probability}")
        print(f"  Raw pass_probability: {pass_probability}")

        # Determine result
        prediction_value = int(prediction[0])
        will_fail = prediction_value == 0
        result_label = "FAIL" if will_fail else "PASS"
        confidence = "High" if max(pass_probability, fail_probability) > 0.7 else "Medium"

        # Get contributing factors
        top_factors = get_top_factors(features, fail_probability)

        # Get recommendation
        recommendation = get_recommendation(prediction_value, fail_probability)

        # Prepare result data
        result = {
            'success': True,
            'prediction': prediction_value,
            'prediction_label': result_label,
            'will_fail': will_fail,
            'pass_probability': float(pass_probability),
            'fail_probability': float(fail_probability),
            'probability_percent_fail': f"{fail_probability * 100:.1f}",
            'probability_percent_pass': f"{pass_probability * 100:.1f}",
            'confidence': confidence,
            'top_factors': top_factors,
            'recommendation': recommendation
        }

        return render_template('index.html', result=result, form_data=form_data)

    except Exception as e:
        import traceback
        print("PREDICTION ERROR:")
        print(traceback.format_exc())

        error_result = {
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction. Please check your input values.'
        }
        return render_template('index.html', result=error_result, form_data=request.form.to_dict())

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(model_path),
        'message': 'Emission Prediction API is running'
    })

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING EMISSION APPLICATION")
    print("=" * 80)
    print("Access the application at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("=" * 80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)