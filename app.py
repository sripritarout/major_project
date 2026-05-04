import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Crash Severity Prediction")
st.title("🚨 Crash Severity Prediction System (LSTM Model)")

# --- 1. Load Data ---
# Load the full dataset to derive all necessary mappings and statistics
df_full = pd.read_csv("traffic_sample.csv")
df_full.columns = df_full.columns.str.strip().str.lower()

# --- 2. Data Preprocessing Setup (Replicate Training Environment) ---
# Create a copy for preprocessing to avoid modifying the original df_full
X_preproc = df_full.drop("most_severe_injury", axis=1).copy()
y_preproc = df_full["most_severe_injury"].copy()

# Store LabelEncoders for categorical columns
label_encoders = {}
categorical_cols_to_encode = X_preproc.select_dtypes(include='object').columns

for col in categorical_cols_to_encode:
    le = LabelEncoder()
    # Fit the LabelEncoder on all unique string representations of values in the column
    # This handles NaN as a separate category if present, making it robust for new inputs.
    unique_vals = X_preproc[col].astype(str).unique()
    le.fit(unique_vals)
    X_preproc[col] = le.transform(X_preproc[col].astype(str))
    label_encoders[col] = le

# All columns in X_preproc are now numeric (either original numbers or encoded categories)
all_features_numeric_cols = X_preproc.columns

# Now, fill any remaining NaNs in these numeric columns with their mean
for col in all_features_numeric_cols:
    if X_preproc[col].isnull().any():
        X_preproc[col] = X_preproc[col].fillna(X_preproc[col].mean())

# Fit MinMaxScaler on the fully processed X_preproc
scaler = MinMaxScaler()
scaler.fit(X_preproc)

# Fit LabelEncoder for the target variable 'most_severe_injury'
target_le = LabelEncoder()
# Fit on all unique non-NaN values of the target variable to map predictions back
target_le.fit(y_preproc.dropna())

# --- 3. Load LSTM Model ---
try:
    model = load_model("lstm_model.h5")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}. Make sure 'lstm_model.h5' is in the same directory.")
    st.stop()

# --- 4. Streamlit UI for Feature Selection ---

st.header("Input Crash Parameters")

# Categorical features for direct user input via selectbox
# These are chosen to be representative and manageable in the UI
user_input_categorical_features = [
    'weather_condition', 'lighting_condition', 'first_crash_type',
    'trafficway_type', 'alignment', 'roadway_surface_cond',
    'road_defect', 'prim_contributory_cause', 'crash_type', 'damage',
    'traffic_control_device'
]

# Dictionary to collect user inputs
user_inputs_collected = {}

for feature in user_input_categorical_features:
    # Get unique values from the original dataframe (string representation)
    unique_vals = df_full[feature].dropna().astype(str).unique()
    selected_value = st.selectbox(f"Select {feature.replace('_', ' ').title()}", unique_vals)
    user_inputs_collected[feature] = selected_value

# --- 5. Prepare a single input row for prediction ---

input_row_data = {}

for col in X_preproc.columns:
    if col in user_inputs_collected:
        # Use user-provided value for features selected in UI
        input_row_data[col] = user_inputs_collected[col]
    elif col in categorical_cols_to_encode:
        # For other categorical features not in UI (e.g., 'crash_date'), use mode from original df_full
        # Ensure mode is retrieved as a string to match LabelEncoder fitting
        mode_val_str = df_full[col].dropna().astype(str).mode()[0] if not df_full[col].dropna().empty else 'nan'
        input_row_data[col] = mode_val_str
    else:
        # For numerical features, use mean from original df_full
        input_row_data[col] = df_full[col].mean()

# Create a DataFrame from the prepared input data
single_prediction_input = pd.DataFrame([input_row_data])

# Ensure correct column order before transformation (crucial!)
single_prediction_input = single_prediction_input[X_preproc.columns]

# --- 6. Preprocess the input row for the model ---

# Apply LabelEncoders to the categorical columns in the single input row
for col in categorical_cols_to_encode:
    value_to_transform = single_prediction_input[col].iloc[0]
    try:
        single_prediction_input[col] = label_encoders[col].transform([value_to_transform])[0]
    except ValueError:
        # Fallback for unforeseen categories: use the encoded mode from the preprocessed data
        st.warning(f"Category '{value_to_transform}' for '{col}' not seen during training. Using encoded mode from training data.")
        single_prediction_input[col] = X_preproc[col].mode()[0]

# Ensure all columns are float type for consistency before scaling
for col in all_features_numeric_cols:
    if col in single_prediction_input.columns:
        single_prediction_input[col] = single_prediction_input[col].astype(float)

# Scale the input features
scaled_input = scaler.transform(single_prediction_input)

# Reshape for LSTM (samples, timesteps, features)
num_features = scaled_input.shape[1]
reshaped_input = scaled_input.reshape(1, 1, num_features)

# --- 7. Prediction ---
if st.button("Predict Most Severe Injury"):
    prediction_probs = model.predict(reshaped_input)
    predicted_class_index = np.argmax(prediction_probs, axis=1)[0]

    # Map the predicted class index back to the original label
    predicted_injury = target_le.inverse_transform([predicted_class_index])[0]

    st.success(f"🚑 Predicted Most Severe Injury: **{predicted_injury}**")
    st.markdown("---")
    st.subheader("Prediction Confidence:")
    confidence_df = pd.DataFrame({
        'Injury Type': target_le.inverse_transform(np.arange(len(target_le.classes_))), # Use target_le.classes_ for all possible labels
        'Confidence': prediction_probs[0]
    }).sort_values(by='Confidence', ascending=False)
    st.write(confidence_df)
