"""
Title: TRUDN (Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON) - Urban Drainage Network (UDN))
Author: Husamettin Taysi, github: htaysi
Date: 01/31/2025
Description: A Machine-learning based tool for the paper titled "Enhancing 2D Hydrodynamic Flood Models through Machine Learning and Urban Drainage Integration" 
"""


# %% Import Libraries and Prep Datasets
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Flatten, ConvLSTM2D, TimeDistributed
import sys
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
# %% Data Preprocessing

# Define the directory where datasets are located
# Users should replace this with their own path or configure it as needed
path = "data"  # Relative path to a 'data' folder in the repository

# Construct file paths (relative to path)
model_file = os.path.join(path, "trudn_model.csv")
spring_coord_file = os.path.join(path, "SpringCoords.csv")
cypress_coord_file = os.path.join(path, "CypressCoords.csv")

# Read model data
model_df = pd.read_csv(model_file)

# Remove rows containing -9999 in any column (assumed as missing data indicator)
model_df = model_df[~(model_df == -9999).any(axis=1)]

# Replace negative values with 0 to ensure all data is non-negative
model_df = model_df.map(lambda x: max(x, 0))

# Read coordinates of Spring Creek (test dataset)
spring_coord = pd.read_csv(spring_coord_file)

# Read coordinates of Cypress Creek (train dataset)
cypress_coord = pd.read_csv(cypress_coord_file)

# Remove duplicate coordinates in the model dataset based on 'X' and 'Y'
model_df = model_df.drop_duplicates(['X', 'Y'])

# Drop 'OBJECTID' column from model_df if it exists to avoid conflicts during merging
model_df = model_df.drop(columns=['OBJECTID'], errors='ignore')

# Extract Cypress Creek dataset by merging with coordinate data
cypress = pd.merge(cypress_coord, model_df, on=['X', 'Y'], how='left')

# Replace NaN values with 0 in Cypress Creek dataset
cypress.fillna(0, inplace=True)

# Extract Spring Creek dataset by merging with coordinate data
spring = pd.merge(spring_coord, model_df, on=['X', 'Y'], how='left')

# Replace NaN values with 0 in Spring Creek dataset
spring.fillna(0, inplace=True)

# %% Data Preparation for Machine Learning

# Set random seed for reproducibility
SEED = 100

def prep_data(train, test):
    """
    Prepare training, validation, and test datasets for machine learning.
    
    Parameters:
    - train (pd.DataFrame): Training dataset (e.g., Cypress Creek data)
    - test (pd.DataFrame): Test dataset (e.g., Spring Creek data)
    
    Returns:
    - Tuple containing scaled and reshaped training, validation, and test data,
      original features, and indices for coordinate retrieval.
      
      Note: X_temp_2 creates dataset for Temporal Input: Streamflow. It is not considered in our study but users can integrate it.
      To do that, users can follow a similar procedure with Temporal Input 1: TR-Emulator WSE
    """
    # Keep copies of original datasets
    train_original = train.copy()
    test_original = test.copy()

    # Define feature column groups
    columns_tremuWSE = [f'triton_f{i}' for i in range(1, 241)]  # TR-Emulator WSE outputs
    columns_Q = [f'Q_f{i}' for i in range(1, 241)]  # Streamflow 
    columns_usgs = [f'usgs_f{i}' for i in range(1, 241)]  # USGS WSE targets
    columns_const = ['lulc', 'dist_river', 'dem']  # Constant features
    columns_udn = ['pipe_size', 'flow_capacity', 'length']  # UDN features

    # Remove coordinate and ID columns from feature sets
    feat = [col for col in train.columns if col not in ['X', 'Y', 'ID']]
    train = train[feat]
    test = test[feat]

    # Extract input and target datasets
    X_temp = train[columns_tremuWSE]  # Temporal Input 1: TR-Emulator WSE
    X_temp_2 = train[columns_Q]  # Temporal Input 2: Streamflow (optional to use)
    y_temp = train[columns_usgs]  # Temporal Target: USGS WSE
    X_const = train[columns_const]  # Constant Input 1: Constant features
    X_udn = train[columns_udn]  # Constant Input 2: UDN features

    # Extract test datasets
    X_test_temp = test[columns_tremuWSE]
    X_test_temp_2 = test[columns_Q]
    y_test_temp = test[columns_usgs]
    X_test_const = test[columns_const]
    X_test_udn = test[columns_udn]

    # Split the main train dataset into train and validation sets 
    X_train, X_val = train_test_split(X_temp, test_size=0.2, random_state=SEED)
    X_train_2, X_val_2 = train_test_split(X_temp_2, test_size=0.2, random_state=SEED)
    y_train, y_val = train_test_split(y_temp, test_size=0.2, random_state=SEED)
    X_const_train, X_const_val = train_test_split(X_const, test_size=0.2, random_state=SEED)
    X_udn_train, X_udn_val = train_test_split(X_udn, test_size=0.2, random_state=SEED)

    # Store indices for later use (assigning coordinates)
    train_idx = X_train.index
    val_idx = X_val.index
    test_idx = X_test_temp.index

    # Initialize scalers
    udn_scaler = PowerTransformer(method='yeo-johnson')
    const_scaler = PowerTransformer(method='yeo-johnson')
    temp_scaler = PowerTransformer(method='yeo-johnson')
    temp_scaler_2 = PowerTransformer(method='yeo-johnson')

    # Scale training data
    X_train_const = const_scaler.fit_transform(X_const_train)
    X_train_udn = udn_scaler.fit_transform(X_udn_train)
    X_train_temp = temp_scaler.fit_transform(X_train)
    X_train_temp_2 = temp_scaler_2.fit_transform(X_train_2)

    # Scale validation data
    X_val_const = const_scaler.transform(X_const_val)
    X_val_udn = udn_scaler.transform(X_udn_val)
    X_val_temp = temp_scaler.transform(X_val)
    X_val_temp_2 = temp_scaler_2.transform(X_val_2)

    # Scale test data
    X_test_const = const_scaler.transform(X_test_const)
    X_test_udn = udn_scaler.transform(X_test_udn)
    X_test_temp = temp_scaler.transform(X_test_temp)
    X_test_temp_2 = temp_scaler_2.transform(X_test_temp_2)

    # Reshape temporal data for ML (e.g., for LSTM and ConvLSTM)
    X_train_temp = X_train_temp.reshape((len(X_train), 240, 1))
    X_train_temp_2 = X_train_temp_2.reshape((len(X_train_2), 240, 1))
    X_val_temp = X_val_temp.reshape((len(X_val), 240, 1))
    X_val_temp_2 = X_val_temp_2.reshape((len(X_val_2), 240, 1))
    X_test_temp = X_test_temp.reshape((len(X_test_temp), 240, 1))
    X_test_temp_2 = X_test_temp_2.reshape((len(X_test_temp_2), 240, 1))

    # Return all prepared data and indices
    return (X_train_temp, X_train_temp_2, X_train_const, X_train_udn, y_train,
            X_val_temp, X_val_temp_2, X_val_const, X_val_udn, y_val,
            X_test_temp, X_test_temp_2, X_test_const, X_test_udn, y_test_temp,
            X_temp, X_temp_2, X_const, X_udn, y_temp, train_idx, val_idx, test_idx)

# Example usage with preprocessed Cypress and Spring datasets
X_train_temp, X_train_temp_2, X_train_const, X_train_udn, y_train, \
X_val_temp, X_val_temp_2, X_val_const, X_val_udn, y_val, \
X_test_temp, X_test_temp_2, X_test_const, X_test_udn, y_test, \
X_temp, X_temp_2, X_const, X_udn, y_temp, train_idx, val_idx, test_idx = prep_data(cypress, spring)

# Extract coordinates and IDs for train, validation, and test sets
train_xy = cypress.loc[train_idx, ["X", "Y", "OBJECTID"]]
val_xy = cypress.loc[val_idx, ["X", "Y", "OBJECTID"]]
test_xy = spring.loc[test_idx, ["X", "Y", "OBJECTID"]]
# %% Run TRUDN Model
# This script defines and trains the TRUDN model, a hybrid neural network combining temporal (LSTM) 
# and constant/UDN features to predict water surface elevation (WSE) over 240 timesteps.

# Convert the input data to tensors
y = tf.convert_to_tensor(y_train.values.reshape(len(y_train), 240))        # Training target (USGS WSE)
y_valid = tf.convert_to_tensor(y_val.values.reshape(len(y_val), 240))     # Validation target
X_temp_train = tf.reshape(X_train_temp, shape=(X_train_temp.shape[0], 1, 1, 240, 1))      # Temporal input 1 (TR-Emulator WSE)
X_temp_val = tf.reshape(X_val_temp, shape=(X_val_temp.shape[0], 1, 1, 240, 1))            # Validation temporal input 1
X_temp_train_2 = tf.reshape(X_train_temp_2, shape=(X_train_temp_2.shape[0], 1, 1, 240, 1)) # Temporal input 2 (Streamflow Q)
X_temp_val_2 = tf.reshape(X_val_temp_2, shape=(X_val_temp_2.shape[0], 1, 1, 240, 1))      # Validation temporal input 2

# TRITON Model layers    
cons_features = 3  # Number of constant features (e.g., lulc, dist_river, dem)
udn_features = 3   # Number of UDN features (e.g., pipe_size, flow_capacity, length)
constant_input = tf.keras.Input(shape=(cons_features,), name="constants")      # Input for constant features
temporal_input = tf.keras.Input(shape=(1, 1, 240, 1), name='temporal')         # Input for temporal data (TR-Emulator WSE)
temporal_input_2 = tf.keras.Input(shape=(1, 1, 240, 1), name='temporal_2')     # Input for temporal data (Streamflow Q)
udn_input = tf.keras.Input(shape=(udn_features,), name='udn')                  # Input for UDN features

# Define a time range for the peak region 
# This is done manually based on pr simulations. 
# To understand the time range of peak flows, users can check TRITON results, USGS Water Data, 
# or use a different loss function without a peak region
peak_start = 75  # Start of peak flow region (timestep index)
peak_end = 95    # End of peak flow region (timestep index)
alpha = 0.5      # Weight for global vs. peak MSE (0.5 = equal weight)

def two_term_mse_loss(y_true, y_pred):
    """
    Custom loss function combining global MSE and MSE in a specific time window (peak region).
    
    Args:
        y_true: True values with shape (batch, 240)
        y_pred: Predicted values with shape (batch, 240)
    
    Returns:
        Weighted average of global and peak-region MSE.
    """
    # 1) Global MSE (over all 240 timesteps)
    mse_global = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)  # shape (batch,)

    # 2) Peak-region MSE (slice the timesteps from peak_start to peak_end)
    y_true_peak = y_true[:, peak_start:peak_end]
    y_pred_peak = y_pred[:, peak_start:peak_end]
    mse_peak = tf.reduce_mean(tf.square(y_true_peak - y_pred_peak), axis=-1)  # shape (batch,)

    # Combine them with alpha (e.g., alpha=0.5 gives equal weight to global and peak)
    loss_per_sample = alpha * mse_global + (1.0 - alpha) * mse_peak

    # Average across the batch
    return tf.reduce_mean(loss_per_sample)

# Temporal layers (Water Level timeseries)        
lstm_1 = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(1,1), name="lstm_1")(temporal_input)
flat = tf.keras.layers.Flatten(name="flat")(lstm_1)
reshaped_flat = tf.keras.layers.Reshape((1, flat.shape[1]))(flat)
lstm_2 = tf.keras.layers.LSTM(16, name="lstm_2", return_sequences=True)(reshaped_flat)
temp_lstm = tf.keras.layers.Dropout(0.05)(lstm_2)
temp_lstm_flat = tf.keras.layers.Flatten()(temp_lstm)

# Constant layers (DEM, n, Slope)
dense_const = tf.keras.layers.Dense(32, name="dense_const", activation='relu')(constant_input)
flat_const = tf.keras.layers.Flatten(name="flat_const")(dense_const)

# UDN layers (Pipe length, Drainage density, Distance to nodes)
dense_udn = tf.keras.layers.Dense(32, name="dense_udn", activation='relu')(udn_input)
flat_udn = tf.keras.layers.Flatten(name="flat_udn")(dense_udn)

# Concatenate temporal + constant + UDN layers
concat_1 = tf.keras.layers.concatenate([temp_lstm_flat, flat_const, flat_udn], name="concat_1")

dense_1 = tf.keras.layers.Dense(240, name="output")(concat_1)   
    
trudn = tf.keras.Model(inputs=[temporal_input, constant_input, udn_input], 
                      outputs=[dense_1])  # Note: temporal_input_2 is defined but not used yet

# Define early stopping callback to prevent overfitting
es = tf.keras.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model with custom loss and standard metrics
trudn.compile(optimizer=optimizer,
              loss=two_term_mse_loss,
              metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

# Display model summary
trudn.summary()

# Train the model
# Note: temporal_input_2 (X_temp_train_2, X_temp_val_2) is prepared but not used in this architecture
history_trudn = trudn.fit(
    {"temporal": X_temp_train, "constants": X_train_const, "udn": X_train_udn},
    y,
    epochs=75,
    batch_size=16,
    validation_data=({"temporal": X_temp_val, "constants": X_val_const, "udn": X_val_udn}, y_valid),
    callbacks=[es]
)

# Commented alternative training call from original code (kept for reference)
# history_triton = trudn.fit({"temporal": X_triton_train, "temporal_2":X_temp_train_2,  "udn": X_train_udn}, 
#                            y, 
#                            epochs=100, 
#                            batch_size=8, 
#                            validation_data=({"temporal": X_triton_val, "temporal_2":X_temp_val_2, "udn": X_val_udn}, y_valid), 
#                            callbacks=[es])

# %% Predictions
# This section reshapes test, validation, and training data for predictions, 
# adds coordinates to the datasets, and saves predicted and true values to CSV files 
# within a 'Predictions' subfolder in the 'data' directory.


# Define the directory for Predictions
predictions_dir = os.path.join(path, "Predictions")  # Subfolder for predictions
os.makedirs(predictions_dir, exist_ok=True)  # Create Predictions subfolder if it doesn’t exist

# Reshape temporal data to match model input shape (samples, 1, 1, 240, 1)
X_temp_test = X_test_temp.reshape((X_test_temp.shape[0], 1, 1, X_test_temp.shape[1], X_test_temp.shape[2]))
X_temp_test_2 = X_test_temp_2.reshape((X_test_temp_2.shape[0], 1, 1, X_test_temp_2.shape[1], X_test_temp_2.shape[2]))
X_temp_val = X_val_temp.reshape((X_val_temp.shape[0], 1, 1, X_val_temp.shape[1], X_val_temp.shape[2]))
X_temp_val_2 = X_val_temp_2.reshape((X_val_temp_2.shape[0], 1, 1, X_val_temp_2.shape[1], X_val_temp_2.shape[2]))
X_temp_train = X_train_temp.reshape((X_train_temp.shape[0], 1, 1, X_train_temp.shape[1], X_train_temp.shape[2]))
X_temp_train_2 = X_train_temp_2.reshape((X_train_temp_2.shape[0], 1, 1, X_train_temp_2.shape[1], X_train_temp_2.shape[2]))

# Add coordinates to train, validation, and test sets
# Note: train_xy, val_xy, and test_xy are assumed to come from the data preparation step
y_train_xy = y_train.copy()  # Copy training targets (USGS WSE)
y_train_xy.insert(0, "X", train_xy["X"])  # Insert X coordinates
y_train_xy.insert(1, "Y", train_xy["Y"])  # Insert Y coordinates

y_val_xy = y_val.copy()  # Copy validation targets
y_val_xy.insert(0, "X", val_xy["X"])
y_val_xy.insert(1, "Y", val_xy["Y"])

y_test_xy = y_test.copy()  # Copy test targets
y_test_xy.insert(0, "X", test_xy["X"])
y_test_xy.insert(1, "Y", test_xy["Y"])

# Define file paths within the 'data/Predictions' subfolder for saving predictions and true values
preds_test_csv_path = os.path.join(predictions_dir, "preds_test.csv") # Predicted Test dataset        
y_test_csv_path = os.path.join(predictions_dir, "y_test.csv") # Target Test dataset
preds_val_csv_path = os.path.join(predictions_dir, "preds_val.csv") # Predicted Val dataset
y_val_csv_path = os.path.join(predictions_dir, "y_val.csv") # Target Validation dataset
preds_train_csv_path = os.path.join(predictions_dir, "preds_train.csv") # Predicted Train dataset
y_train_csv_path = os.path.join(predictions_dir, "y_train_xy.csv") # Target Train dataset

# Predict USGS WSE for test set and save results
preds_test = trudn.predict({"temporal": X_temp_test, "udn": X_test_udn, "constants": X_test_const})
np.savetxt(preds_test_csv_path, preds_test, delimiter=',')  # Save predictions as CSV
y_test_xy.to_csv(y_test_csv_path, index=False)              # Save true values with coordinates

# Predict USGS WSE for validation set and save results
preds_val = trudn.predict({"temporal": X_temp_val, "udn": X_val_udn, "constants": X_val_const})
np.savetxt(preds_val_csv_path, preds_val, delimiter=',')
y_val_xy.to_csv(y_val_csv_path, index=False)

# Predict USGS WSE for training set and save results
preds_train = trudn.predict({"temporal": X_temp_train, "udn": X_train_udn, "constants": X_train_const})
np.savetxt(preds_train_csv_path, preds_train, delimiter=',')
y_train_xy.to_csv(y_train_csv_path, index=False)