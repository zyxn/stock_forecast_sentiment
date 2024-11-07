import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import xgboost as xgb
import joblib

# Setup logging to file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs_{timestamp}.log"),
        logging.StreamHandler()
    ]
)

class StockPredictionModel:
    def __init__(self, filepath, timesteps=10, features=4, model_type="LSTM"):
        self.filepath = filepath
        self.timesteps = timesteps
        self.features = features
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self.model = None
        logging.info(f"Initialized StockPredictionModel with model type: {model_type}")

    def load_and_process_data(self):
        logging.info("Loading and processing data")
        data = pd.read_csv(self.filepath, sep=",", thousands=".", decimal=",")
        data.rename(columns={
            "Tanggal": "Date",
            "Terakhir": "Close",
            "Pembukaan": "Open",
            "Tertinggi": "High",
            "Terendah": "Low",
            "Vol.": "Volume",
            "Perubahan%": "Change%"
        }, inplace=True)
        data = data.iloc[::-1].reset_index(drop=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.set_index("Date", inplace=True)
        logging.info("Data loaded and processed successfully")
        return data.drop(["Change%", "Volume"], axis=1)

    def split_data(self, data, split_ratio=0.8):
        logging.info(f"Splitting data with a ratio of {split_ratio}")
        length_data = len(data)
        length_train = round(length_data * split_ratio)
        return data[:length_train], data[length_train:]

    def scale_data(self, train_data, validation_data):
        logging.info("Scaling data")
        scaled_train_data = pd.DataFrame(self.scaler.fit_transform(train_data), columns=train_data.columns)
        scaled_validation_data = pd.DataFrame(self.scaler.transform(validation_data), columns=validation_data.columns)
        logging.info("Data scaling complete")
        return scaled_train_data, scaled_validation_data

    def reshape_data_lstm(self, scaled_data):
        logging.info("Reshaping data for LSTM model")
        
        # Buat batch input 3D untuk LSTM (jumlah sampel, timesteps, fitur)
        data_reshaped = []
        
        # Iterasi dengan langkah timesteps agar setiap sampel mengandung urutan data dengan panjang yang sama
        for i in range(len(scaled_data) - self.timesteps + 1):
            timestep_data = scaled_data.iloc[i:i + self.timesteps].values  # Ambil timestep data dari `i`
            data_reshaped.append(timestep_data)
        
        # Konversi ke array numpy untuk memastikan kompatibilitas dengan model LSTM
        data_reshaped = np.array(data_reshaped)
        
        logging.info(f"Data reshaped successfully into shape {data_reshaped.shape}")
        return data_reshaped
    
    def prepare_y_for_lstm(self, y):
        """
        Adjust y to match the reduced length after creating X with timesteps.
        """
        return y[self.timesteps - 1:].values


    def build_model(self, input_shape=None):
        logging.info(f"Building model of type: {self.model_type}")
        if self.model_type == "LSTM":
            self.model = Sequential([
                Bidirectional(LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape)),
                Dropout(0.1),
                LSTM(128, activation='relu', return_sequences=False),
                Dense(128),
                Dense(1)
            ])
            self.model.compile(optimizer='adam', loss="mse", metrics=["mse"])
        elif self.model_type == "SVR":
            params =  {'C': 10, 'epsilon': 0.001, 'gamma': 'scale', 'kernel': 'linear'}
            self.model = SVR(**params)
        elif self.model_type == "XGBoost":
            self.model = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0.002, gamma=0.5, max_depth=1, n_estimators=699, learning_rate=0.01, random_state=42)
        logging.info("Model built successfully")
        
    def plot_results(self, y_train, y_pred_train, y_val, y_pred_val, train_index, val_index):
        logging.info("Plotting results")
        plt.figure(figsize=(10, 6))
        plt.plot(train_index, y_pred_train.squeeze(), label='Prediksi (Train)', color='red')
        plt.plot(train_index, y_train, label='Data Aktual (Train)', color='blue')
        plt.xlabel('Tanggal')
        plt.ylabel('Nilai Close')
        plt.title('Prediksi vs Data Aktual (Train)')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(val_index, y_pred_val.squeeze(), label='Prediksi (Validation)', color='red')
        plt.plot(val_index, y_val, label='Data Aktual (Validation)', color='blue')
        plt.xlabel('Tanggal')
        plt.ylabel('Nilai Close')
        plt.title('Prediksi vs Data Aktual (Validation)')
        plt.legend()
        plt.show()
        logging.info("Plotting complete")

    def train_model(self, X_train, y_train, epochs=200, batch_size=32):
        logging.info(f"Training model of type: {self.model_type}")
        if self.model_type == "LSTM":
            early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1, restore_best_weights=True)
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        else:
            self.model.fit(X_train, y_train)
        logging.info("Model training complete")

    def evaluate_model(self, X, y_true):
        logging.info("Evaluating model")
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        logging.info(f"Evaluation complete with MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")
        return mse, rmse, mae, mape

    def save_model(self, rmse_val):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = 1

        # Determine version
        model_name_base = f"{self.model_type}_model"
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        
        while True:
            model_filename = f"{model_name_base}_v{model_version}_{timestamp}_RMSE_{rmse_val:.2f}.h5"
            model_path = os.path.join(model_dir, model_filename)
            if not os.path.exists(model_path):
                break
            model_version += 1

        if self.model_type == "LSTM":
            self.model.save(model_path)
        else:
            joblib.dump(self.model, model_path)
        
        scaler_path = os.path.join(model_dir, f"{model_name_base}_scaler_v{model_version}_{timestamp}.pkl")
        joblib.dump(self.scaler, scaler_path)

        logging.info(f"Model saved as {model_path} and scaler as {scaler_path}")

    def run(self):
        # Load and preprocess data
        data = self.load_and_process_data()
        train_data, validation_data = self.split_data(data)

        # Scale data
        scaled_train_data, scaled_validation_data = self.scale_data(train_data, validation_data)

        # Prepare target variables
        y_train = train_data["Close"].values
        y_val = validation_data["Close"].values

        # Reshape data for LSTM or prepare for SVR/XGBoost
        if self.model_type == "LSTM":
            X_train = self.reshape_data_lstm(scaled_train_data[["Open", "Close", "High", "Low"]])
            X_val = self.reshape_data_lstm(scaled_validation_data[["Open", "Close", "High", "Low"]])
            y_train = self.prepare_y_for_lstm(train_data["Close"])
            y_val = self.prepare_y_for_lstm(validation_data["Close"])
            input_shape = (X_train.shape[1], self.features)
        else:
            X_train = scaled_train_data[["Open", "Close", "High", "Low"]].values
            X_val = scaled_validation_data[["Open", "Close", "High", "Low"]].values
            input_shape = None

        # Build and train model
        self.build_model(input_shape=input_shape)
        self.train_model(X_train, y_train)

        # Predict and evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)

        # Evaluate model performance
        train_metrics = self.evaluate_model(X_train, y_train)
        val_metrics = self.evaluate_model(X_val, y_val)
        rmse_val = val_metrics[1]  # Extract RMSE from validation metrics

        # Save model with versioning and RMSE information
        self.save_model(rmse_val)
        
        # Display metrics
        metrics_df = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "MAE", "MAPE"],
            "Train": train_metrics,
            "Validation": val_metrics
        })
        print(metrics_df)
        self.plot_results(y_train, y_pred_train, y_val, y_pred_val, train_data.index, validation_data.index)



if __name__ == "__main__":
    # Path to data file
    filepath = r"data/external/14tahun.csv"
    
    # Parameter model
    timesteps = 10
    features = 4
    model_type = "LSTM"  # Options: "LSTM", "SVR", "XGBoost"
    
    # Buat instance dari StockPredictionModel
    model = StockPredictionModel(filepath, timesteps=timesteps, features=features, model_type=model_type)
    
    # Jalankan proses pelatihan dan evaluasi
    model.run()