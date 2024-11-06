from flask import Flask, jsonify, request
from flask_cors import CORS
from model import model
# from model import trainModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# trainModel()


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data in request"}), 400

    # Process data with your AI model
    # prediction = your_ai_model.predict(data)
    company = data['comp']
    days = int(data['days'])
    result = model(company, days)

    # Return the prediction as JSON
    # return jsonify({"prediction": prediction})
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
