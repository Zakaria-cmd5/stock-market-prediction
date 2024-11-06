import base64
from io import BytesIO
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


def model(inCompany, days):
    # Load data
    company = inCompany
    start = dt.datetime(2013, 1, 1)
    end = dt.datetime(2024, 1, 1)
    data = yf.download(company, start=start, end=end)

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data) - days + 1):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x: x + days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60))
    model.add(Dropout(0.2))
    model.add(Dense(units=days))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_start = dt.datetime(2024, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(company, start=test_start, end=test_end)

    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(
        total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Extract the first day predictions only
    predicted_prices_first_day = predicted_prices[:, 0]

    plt.plot(actual_prices, color='black', label=f"Actual {company} price")
    plt.plot(predicted_prices_first_day, color='green',
             label=f"Predicted {company} price")
    plt.title(f"{company} share price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} share price')
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.clf()

    # real_data = model_inputs[-prediction_days:]
    # real_data = np.reshape(real_data, (1, real_data.shape[0], 1))

    # predictions = []
    # for day in range(days):
    #     prediction = model.predict(real_data)
    #     predictions.append(prediction[0, 0])
    #     prediction_reshaped = np.reshape(prediction[0, 0], (1, 1, 1))
    #     real_data = np.append(real_data[:, 1:, :], prediction_reshaped, axis=1)

    # predictions = scaler.inverse_transform(
    #     np.array(predictions).reshape(-1, 1))

    real_data = [
        model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(
        real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

    response = {"image": image_data, "prediction": prediction.tolist()}
    return response


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    company = data['comp']
    days = data['days']
    result = model(company, days)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
