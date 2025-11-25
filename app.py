from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['income', 'age', 'loan_amount', 'credit_score', 'employment_years']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}", "required_fields": required_fields}), 400
        
        features = np.array([[float(data['income']), float(data['age']), float(data['loan_amount']), float(data['credit_score']), float(data['employment_years'])]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            "loan_approved": bool(prediction),
            "approval_status": "Approved" if prediction == 1 else "Rejected",
            "confidence": {"rejection_probability": float(probability[0]), "approval_probability": float(probability[1])},
            "input_data": data
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
