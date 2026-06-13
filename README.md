# 👥 Occupancy Sensing System

An intelligent machine learning application that predicts room occupancy based on real-time environmental parameters. Using data gathered from sensors (temperature, humidity, light levels, and carbon dioxide), the system identifies whether a room is **occupied** or **vacant**, making it ideal for smart buildings, energy-saving systems, and smart HVAC controls.

---

## 🚀 Features
* **Multi-Model Support:** Trains and compares three classification algorithms: Random Forest, Decision Tree, and K-Nearest Neighbors (KNN).
* **Interactive UI Dashboard:** A sleek, dark-themed **Streamlit web application** that allows users to interactively adjust sensor inputs via sliders and see real-time predictions.
* **Ensemble Voting:** Includes a majority-vote ensemble feature that combines the predictions of all three models for maximum robustness.
* **Deployment Ready:** Configured with container/hosting dependency files (`requirements.txt`) for immediate deployment to Streamlit Community Cloud.

---

## 📊 Environmental Dataset Parameters
The system uses the following features to detect room occupancy:
1. **Temperature:** Ambient room temperature in Celsius (°C).
2. **Humidity:** Relative humidity percentage (%).
3. **Light:** Light levels measured in Lux.
4. **CO2:** Carbon Dioxide concentration in parts per million (ppm).
5. **Humidity Ratio:** Derived ratio of water vapor in the air to dry air.

---

## 🛠️ Tech Stack & Architecture
* **Data Manipulation:** `Pandas` and `NumPy` are used to load, clean, and pre-process the sensor dataset.
* **Machine Learning:** `scikit-learn` for training and evaluating classification models.
* **Serialization:** `Pickle` for saving the trained models and encoders so they can be loaded instantly in production.
* **Web Interface:** `Streamlit` with custom CSS styling to create a premium, user-friendly frontend dashboard.

---

## 📂 File Structure
* **`VAT PROJECT _ OCCUPANCY.ipynb`**: The Jupyter Notebook used for data exploration, model training, evaluation, and model serialization.
* **`appX.py`**: The main Python script running the interactive Streamlit web dashboard.
* **`Occupancy.csv`**: The dataset containing historical sensor logs and occupancy labels.
* **`requirements.txt`**: The configuration file containing the necessary packages for hosting/deploying the app online.
* **`*.pkl`**: Pickled models (`dt_model.pkl`, `knn_model.pkl`, `rf_model.pkl`, `label_encoder.pkl`) loaded dynamically by the web app.

---

## 🏃‍♂️ How to Run Locally

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web Dashboard
Launch the Streamlit web application::
```bash
streamlit run appX.py
```
This will automatically open the application in your browser at `http://localhost:8501`.
"# ROOM_OCCUPANCY-sklearn" 
