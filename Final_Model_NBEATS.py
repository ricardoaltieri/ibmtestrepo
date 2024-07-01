# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:55:56 2024

@author: RAltieri
"""

# Load standard Python libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Darts Libraries

from darts import TimeSeries
from darts.utils.callbacks import TFMProgressBar
from darts.models import  NBEATSModel
from darts.metrics import rmsle, mape, smape, mae, mse
from darts.utils.likelihood_models import GaussianLikelihood
#from darts.utils.losses import SmapeLoss
#from darts.dataprocessing.transformers import Scaler

import warnings

warnings.filterwarnings("ignore")

# Load if working locally

import sys
import os
import time

# Change files pat
def change_pythonpath():
    """
    Change the path if the code is running locally.
    """
    if os.path.exists(".local"):
        # Add the local directory to the Python path
        sys.path.insert(0, os.path.abspath(".local"))

# Run the routine to change the path

change_pythonpath()

# Routine to generate plot with train and predictions

def plot_best(train, predictions, low_quantile = 0.05, high_quantile = 0.95, ytitle="Gbps"):
    plt.figure(figsize=(15, 5))
    train.plot()
    predictions.plot(label="forecast", low_quantile=low_quantile, high_quantile=high_quantile)
    plt.ylabel(ytitle) 
    plt.show()

# Routine to generate plot with train, validation and predictions

def plot_best_trial(train, validation, predictions, low_quantile = 0.05, high_quantile = 0.95, ytitle="Gbps"):
    plt.figure(figsize=(15, 5))
    train.plot()
    validation.plot(label="actual")
    predictions.plot(label="forecast", low_quantile=low_quantile, high_quantile=high_quantile)
    plt.ylabel(ytitle) 
    plt.show()


# Routine to print performance metrics including only validation data

def print_performance_metrics(validation, predictions):
    print("RMSLE: {:.4f}".format(rmsle(validation, predictions)))    
    print("MAPE: {:.2f}%".format(mape(validation, predictions)))
    print("sMAPE: {:.2f}%".format(smape(validation, predictions)))
    print("MAE: {:.2f}".format(mae(validation, predictions)))
    print("MSE: {:.2f}".format(mse(validation, predictions)))
    
# Routine to print performance metrics including validation and test data

def print_performance_metrics_test(validation, test, predictions):
    if val_len > 0:
        print("model validation obtains MAPE: {:.2f}%".format(mape(validation, predictions)))
        print("model validation obtains sMAPE: {:.2f}%".format(smape(validation, predictions)))
        print("model validation obtains MAE: {:.2f}".format(mae(validation, predictions)))
        print("model validation obtains RMSLE: {:.4f}".format(rmsle(validation, predictions)))
        print("model validation obtains MSE: {:.2f}".format(mse(validation, predictions)))
        print("On unseen data the model obtains MAPE: {:.2f}%".format(mape(test, predictions)))
        print("On unseen data the model obtains sMAPE: {:.2f}%".format(smape(test, predictions)))
        print("On unseen data the model obtains MAE: {:.2f}".format(mae(test, predictions)))
        print("On unseen data the model obtains RMSLE: {:.4f}".format(rmsle(test, predictions)))
        print("On unseen data the model obtains MSE: {:.2f}".format(mse(test, predictions)))
    else:
        print("On unseen data the model obtains MAPE: {:.2f}%".format(mape(test, predictions)))
        print("On unseen data the model obtains sMAPE: {:.2f}%".format(smape(test, predictions)))
        print("On unseen data the model obtains MAE: {:.2f}".format(mae(test, predictions)))
        print("On unseen data the model obtains RMSLE: {:.4f}".format(rmsle(test, predictions)))
        print("On unseen data the model obtains MSE: {:.2f}".format(mse(test, predictions)))

# Routne to print optimisatoin run details

def opt_details():
    print(f"Validation Length: {val_len}")
    print(f"Number of trials executed: {num_trials}")
    print(f"Elapsed Time for tuning: {elapsed_time_hyperparameter_tune}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

# run torch models on CPU, and disable progress bars for all model stages, except for training.

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

# Read CSV with TotalThrougphputRanGbps monthly data into a PANDAS df    

df = pd.read_csv("C:/Users/ricar/OneDrive/Project/Coding/NetworkDataThroughputPeakMonthly.csv", delimiter=",")
#df = pd.read_csv("C:/Users/raltieri/OneDrive - Three Ireland/Masters/Project/Coding/NetworkDataThroughputPeakMonthly.csv", delimiter=",")

# Convert to TimeSeries

series_TotalThrougphputRanGbps = TimeSeries.from_dataframe(df, time_col="Month", value_cols="TotalThroughputRanGbps")

# Define Split in train/val function to be re-used

def split_train_val(timeseries, validation_length):
    train = timeseries[: -(validation_length)]
    validation = timeseries[-(validation_length) :]
    return train, validation

# Define Split in train/val/test function to be re-used

def split_train_val_test(timeseries, validation_length, test_length):
    train = timeseries[: -(validation_length+test_length)]
    validation = timeseries[-(validation_length+test_length) : -test_length]
    test = timeseries[-test_length:]
    return train, validation, test

# Import Optuna and PyTorch Libraries

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from pytorch_lightning.callbacks import EarlyStopping

# Build and fit a NBEATS model with validation for Hyperparameter Optimisation

def build_fit_nbeats_model(
        fit_series,
        months_in,
        months_out,
        n_epochs,
        generic_architecture,
        batch_size,
        num_stacks,
        num_blocks,
        num_layers,
        layer_widths,
        dropout,
        expansion_coefficient_dim,
        LR,
        callbacks=None
):

    # reproducibility 
    torch.manual_seed(5)

    # some fixed parameters that will be the same for all models

    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    # Training monitoring the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks
    # build the NBEATS model
    
    model = NBEATSModel(
        input_chunk_length=months_in,
        output_chunk_length=months_out,
        batch_size=batch_size,
        n_epochs=n_epochs,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        generic_architecture = generic_architecture,
        dropout=dropout,
        force_reset=True,
        model_name="nbeats_model",
        expansion_coefficient_dim=expansion_coefficient_dim,
        #likelihood=GaussianLikelihood(),
        loss_fn=torch.nn.MSELoss(),
        #add_encoders=encoders,
        save_checkpoints=True,
        optimizer_kwargs={"lr": LR},
        **generate_torch_kwargs(),
        random_state=5,
    )  
    # train the model
    model.fit(
        series=fit_series,
        val_series=val_series,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
    )
    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint("nbeats_model") # Loads the best epoch instead of the last
    
    return model

# Build the same model but Probabilistic

def build_fit_nbeats_prob_model(
        fit_series,
        months_in,
        months_out,
        n_epochs,
        generic_architecture,
        batch_size,
        num_stacks,
        num_blocks,
        num_layers,
        layer_widths,
        dropout,
        expansion_coefficient_dim,
        LR,
        callbacks=None
):

    # reproducibility
    torch.manual_seed(5)

    # some fixed parameters that will be the same for all models

    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    # Training monitoring the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks
    # build the NBEATS model
    
    model = NBEATSModel(
        input_chunk_length=months_in,
        output_chunk_length=months_out,
        batch_size=batch_size,
        n_epochs=n_epochs,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        generic_architecture = generic_architecture,
        dropout=dropout,
        force_reset=True,
        model_name="nbeats_model_prob",
        expansion_coefficient_dim=expansion_coefficient_dim,
        likelihood=GaussianLikelihood(),
        #loss_fn=torch.nn.MSELoss(),
        #add_encoders=encoders,
        save_checkpoints=True,
        optimizer_kwargs={"lr": LR},
        **generate_torch_kwargs(),
        random_state=5,
    )  
    # train the model
    model.fit(
        series=fit_series,
        val_series=val_series,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
    )
    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint("nbeats_model_prob")

    return model

# Buld the same model without using validation during training.

def build_fit_nbeats_model_no_chkpt(
        fit_series,
        months_in,
        months_out,
        n_epochs,
        generic_architecture,
        batch_size,
        num_stacks,
        num_blocks,
        num_layers,
        layer_widths,
        dropout,
        expansion_coefficient_dim,
        LR,
        callbacks=None
):

    # reproducibility 
    torch.manual_seed(5)

    # some fixed parameters that will be the same for all models

    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    # Training monitoring the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks
    # build the NBEATS model
    
    model = NBEATSModel(
        input_chunk_length=months_in,
        output_chunk_length=months_out,
        batch_size=batch_size,
        n_epochs=n_epochs,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        generic_architecture = generic_architecture,
        dropout=dropout,
        force_reset=True,
        model_name="nbeats_model_no_chkpt",
        expansion_coefficient_dim=expansion_coefficient_dim,
        #likelihood=GaussianLikelihood(),
        loss_fn=torch.nn.MSELoss(),
        #add_encoders=encoders,
        save_checkpoints=False,
        optimizer_kwargs={"lr": LR},
        **generate_torch_kwargs(),
        random_state=5,
    )  
    # train the model
    model.fit(
        series=fit_series,
        val_series=val_series,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
    )
    # reload best model over course of training
    #model = NBEATSModel.load_from_checkpoint("nbeats_model_no_chkpt")

    return model

# Build an Objective function for Optuna in order to control the grid search

def objective(trial):
    callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

    # set input_chunk_length, between 5 and 14 days
    months_in = trial.suggest_int("months_in", 1, 12)
   
    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    months_out = trial.suggest_int("months_out", 1, 12)

    # Architecture hyper-params:
    num_stacks = trial.suggest_int("num_stacks", 1, 40)
    num_blocks = trial.suggest_int("num_blocks", 1, 100)
    num_layers = trial.suggest_int("num_layers", 1, 50)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    expansion_coefficient_dim = trial.suggest_int("expansion_coefficient_dim", 3, 5)
    generic_architecture = trial.suggest_categorical("generic_architecture", [True, False])
    
    # Training settings:
    batch_size = trial.suggest_int("batch_size", 32, 512)
    layer_widths = trial.suggest_int("layer_widths", 32, 512)
    n_epochs = trial.suggest_int("n_epochs", 10, 300)
    LR = trial.suggest_float("LR", 0.001, 0.005)
    
    # build and train the NBEATS model with these hyper-parameters:
    model = build_fit_nbeats_model(
        train_series,
        months_in=months_in,
        months_out=months_out,
        n_epochs=n_epochs,
        generic_architecture=generic_architecture,
        batch_size=batch_size,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        dropout=dropout,
        layer_widths=layer_widths,
        expansion_coefficient_dim=expansion_coefficient_dim,
        LR=LR,
        callbacks=callback,
    )

    # Evaluate how good it is on the validation set
    preds = model.predict(series=train_series, n=len(val_series))
    rmsles = rmsle(val_series, preds, n_jobs=-1, verbose=True)

    return rmsles if rmsles != np.nan else float("inf")

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    print(f"Number of trials: {len(study.trials)}")



# Define the serch space

search_space = {"months_in": [11],
                "months_out": [1, 2, 3],
                "n_epochs": [80],
                "num_stacks": [30],
                "num_blocks": [1],
                "num_layers": [4],
                "layer_widths": [256],
                "expansion_coefficient_dim": [5],
                "LR":[1e-3],
                "batch_size": [100],
                'generic_architecture': [True],
                'dropout': [0.0]
                }

# Create a Grid Search Optuna Study

study = optuna.create_study(direction="minimize",sampler=optuna.samplers.GridSampler(search_space))

# Define the validation length
    
val_len = 24 # NBEATS requires a longer validation set of at least input_chunck_length + output_chunk_length

# Split the train and validate sets

train_series, val_series = split_train_val(series_TotalThrougphputRanGbps, val_len)

start_time = time.time() # For computational time calculation

# Run the gridsearch study until timeout reached
study.optimize(objective, timeout=36000, callbacks=[print_callback])

elapsed_time_hyperparameter_tune = time.time() - start_time # End of time taken to run optimisation

# Finally, print the best value and best hyperparameters:
print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

#Retrieve the trials from the study
trials = study.trials

# Create lists to store parameters and metrics:
param_list = []
metric_list = []

# Iterate over the trials and extract parameters and metrics:
for trial in trials:
    params = trial.params  # Dictionary containing the parameters
    metric = trial.value   # Metric value (e.g., accuracy, loss, etc.)
    param_list.append(params)
    metric_list.append(metric)

# Create a Pandas DataFrame:
df_params = pd.DataFrame(param_list)
df_metrics = pd.DataFrame({'metric': metric_list})

# Concatenate parameter and metric DataFrames horizontally
df_study = pd.concat([df_params, df_metrics], axis=1)

# Export Hyperparameters Dataframe to CSV
df_study.to_csv('Trials List NBEATS RMSLE 24M 1.csv', index=False)




search_space = {"months_in": [9, 10, 11, 12],
                "months_out": [1, 2, 3],
                "n_epochs": [80],
                "num_stacks": [30],
                "num_blocks": [1],
                "num_layers": [4],
                "layer_widths": [256],
                "expansion_coefficient_dim": [5, 6],
                "LR":[3e-3, 2e-3],
                "batch_size": [100],
                'generic_architecture': [True],
                'dropout': [0.0]
                }

# Create a Grid Search Optuna Study

study = optuna.create_study(direction="minimize",sampler=optuna.samplers.GridSampler(search_space))

# Define the validation length
 
# Run the gridsearch study until timeout reached
study.optimize(objective, timeout=36000, callbacks=[print_callback])


#Retrieve the trials from the study
trials = study.trials

# Create lists to store parameters and metrics:
param_list = []
metric_list = []

# Iterate over the trials and extract parameters and metrics:
for trial in trials:
    params = trial.params  # Dictionary containing the parameters
    metric = trial.value   # Metric value (e.g., accuracy, loss, etc.)
    param_list.append(params)
    metric_list.append(metric)

# Create a Pandas DataFrame:
df_params = pd.DataFrame(param_list)
df_metrics = pd.DataFrame({'metric': metric_list})

# Concatenate parameter and metric DataFrames horizontally
df_study = pd.concat([df_params, df_metrics], axis=1)

# Export Hyperparameters Dataframe to CSV
df_study.to_csv('Trials List NBEATS RMSLE 24M 2.csv', index=False)





fig = optuna.visualization.plot_contour(study, params=["expansion_coefficient_dim", "LR"]).show(renderer="browser")

# Some Optuna Plots
#fig = optuna.visualization.plot_optimization_history(study).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["months_in", "months_out"]).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["expansion_coefficient_dim", "LR"]).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["LR", "batch_size"]).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["num_stacks", "batch_size"]).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["expansion_coefficient_dim", "LR"]).show(renderer="browser")
fig = optuna.visualization.plot_contour(study, params=["batch_size", "months_out"]).show(renderer="browser")

fig = optuna.visualization.plot_param_importances(study).show(renderer="browser")



""" some of the best parameters for furhter testing
best_params = {'months_in': 9, 'months_out': 5, 'num_stacks': 30, 'num_blocks': 2, 'num_layers': 2, 'expansion_coefficient_dim': 5,
               'generic_architecture': True, 'batch_size': 60, 'layer_widths': 256, 'n_epochs': 100}
best_params = {'months_in': 12, 'months_out': 4, 'num_stacks': 30, 'num_blocks': 2, 'num_layers': 2, 'expansion_coefficient_dim': 5,
               'generic_architecture': True, 'batch_size': 60, 'layer_widths': 256, 'n_epochs': 100}
val_len = 16 # NBEATS requires a longer validation set of at least input_chunck_length + output_chunk_length


******************************* 24 Month
best_params = {'months_in': 12, 'months_out': 2, 'num_stacks': 30, 'num_blocks': 1, 'num_layers': 4, 'dropout': 0.0,
               'expansion_coefficient_dim': 6, 'generic_architecture': True, 'batch_size': 100, 'layer_widths': 256,
               'n_epochs': 80, 'LR': 0.003}


# Split the train and validate sets

train_series, val_series = split_train_val(series_TotalThrougphputRanGbps, val_len)
"""

# Select Best Trial from study
best_trial = study.best_trial
best_params = best_trial.params
# Check the number of trials
num_trials = len(study.trials)


# Test Model with validation data and best prameters
best_params['fit_series'] = train_series
# Fit with the trials best parameters
start_time = time.time() # For computational time calculation
best_model = build_fit_nbeats_model(**best_params)
elapsed_time_training = time.time() - start_time # End of time taken to run optimisation

# Predictions
best_preds = best_model.predict(series=train_series, n=len(val_series), mc_dropout=True)
# Vizually check the model forecast
plot_best_trial(train_series, val_series, best_preds)
# Print the optimisation details and the performance metrics.
print_performance_metrics(val_series, best_preds)
#print optimisatoin details
opt_details()

# Use the best model with the entire time series  to generate 5 year forecast
# Feed the whole series to the built and fit function
best_params['fit_series'] = series_TotalThrougphputRanGbps
#best_params['n_epochs'] = 100
# Fit with the trials best parameters
best_model = build_fit_nbeats_model(**best_params) 
# Predictions for next 5 years
best_preds = best_model.predict(series=series_TotalThrougphputRanGbps, n=12*5, mc_dropout=True)
# Vizually check the model forecast
plot_best(series_TotalThrougphputRanGbps, best_preds)
best_preds.pd_dataframe().to_excel('Forecast_Median_Nbeats_24M_FINAL.xlsx', index=True)

# Back test the model - Cross-Validation of the mo

# Build and fit the model without checkpoint saving in order to run the backtesting with the whole time sereis
#best_params['n_epochs'] = 20+1
best_params['fit_series'] = series_TotalThrougphputRanGbps
best_model_no_chckpt = build_fit_nbeats_model_no_chkpt(**best_params) 

# historical_fcast_nbeats = best_model.historical_forecasts(series, start=0.25, forecast_horizon=6, verbose=True)

historical_fcast_nbeats = best_model_no_chckpt.historical_forecasts(series_TotalThrougphputRanGbps, train_length = None, start=0.3, forecast_horizon=3,
                                                          stride=3, retrain=False, overlap_end=True, last_points_only=True, verbose=True,
                                                          show_warnings=True, predict_likelihood_parameters=False,
                                                          )

series_TotalThrougphputRanGbps.plot(label="data")
historical_fcast_nbeats.plot(label="backtest 3-months ahead forecast")
plt.ylabel("Gbps") 
print("MAPE = {:.2f}%".format(mape(historical_fcast_nbeats, series_TotalThrougphputRanGbps)))
print("RMSLE = {:.4f}".format(rmsle(historical_fcast_nbeats, series_TotalThrougphputRanGbps)))

raw_errors_rmsle = best_model_no_chckpt.backtest(series_TotalThrougphputRanGbps, historical_forecasts=historical_fcast_nbeats,
                                                 train_length = None, start=0.3, forecast_horizon=3, stride=3, retrain=True,
                                                 overlap_end=True, verbose=True, reduction=None, show_warnings=False, metric=rmsle
                                                 )
raw_errors_rmsle = pd.Series(raw_errors_rmsle)
raw_errors_mape = best_model_no_chckpt.backtest(series_TotalThrougphputRanGbps, historical_forecasts=historical_fcast_nbeats,
                                                 train_length = None, start=0.3, forecast_horizon=3, stride=3, retrain=True,
                                                 overlap_end=True, verbose=True, reduction=None, show_warnings=False, metric=mape
                                                 )
raw_errors_mape = pd.Series(raw_errors_mape)

# Calculate the mean of the metrics
rmsle_mean_value = raw_errors_rmsle.mean()
mape_mean_value = raw_errors_mape.mean()

print("Mean rmsle:", rmsle_mean_value)
print("Mean mape:", mape_mean_value)

from darts.utils.statistics import plot_hist

plot_hist(
    raw_errors_rmsle,
    bins=np.arange(0, max(raw_errors_rmsle), 0.01),
    title="Individual backtest RMSLE results histogram",
)

plot_hist(
    raw_errors_mape,
    bins=np.arange(0, max(raw_errors_mape), 0.5),
    title="Individual backtest MAPE results histogram",
)


"""
# Test Model - Probabiilistic

best_prob_model = build_fit_nbeats_prob_model(**best_params)
best_prob_preds = best_prob_model.predict(series=train_series, n=len(val_series)+12, mc_dropout=True, num_samples = 100)

plot_best_trial(train_series, val_series, best_prob_preds)
print_performance_metrics(val_series, best_preds)
"""

# Save the model commands
# Define a file name
nbeats_model_file_path = 'trained_nbeats_model_smape_val15_Final.pth'
# Save the trained model
best_model.save(nbeats_model_file_path)
# Additonally save a dictionary with best model parameters and export to CSV
with open('nbeats_dict_params_1.csv', 'w') as f:
    for key in best_params.keys():
        f.write(f"{key}, {best_params[key]}\n")

"""
# Load the model in case needed
loaded_nbeats_model = NBEATSModel.load(nbeats_model_file_path)
"""


# Apply model to different time series
# Read CSV with 8 different monthly data time series into a PANDAS df    

df = pd.read_csv("C:/Users/raltieri/OneDrive - Three Ireland/Masters/Project/Coding/Dataset2.csv", index_col=0, parse_dates=True, dayfirst=True)

# Convert the single dataframe in a dictionary with the 8 time series

dataframes = {}

for column in df.columns:
        # Create a new DataFrame for each column
        column_df = pd.DataFrame({column: df[column].dropna()}) # Drop non numerical values with dropna() method
        tSeries = TimeSeries.from_dataframe(column_df)          # Convert to a Darts TimeSeries
        dataframes[column] = tSeries
    
# Create a list containing the time sereis
dataframes_list = list(dataframes.values()) # Convert to a list of timeSeries

# Split in train/val/test
val_len = 24 # 12 months

# Split  each sereis in train and validation based on its lenght using 

train_series = [s[: -(val_len)] for s in dataframes_list]
val_series = [s[-(val_len) :] for s in dataframes_list]
figure, ax = plt.subplots(4, 2, figsize=(15, 10), dpi=100)

for i, idx in enumerate(range(8)):
    axis = ax[i // 2, i % 2]
    best_params['fit_series'] = train_series[idx]
    preds = best_model.predict(series=train_series[idx], n=len(val_series[idx]), mc_dropout=True)
    print(df.columns[i])
    print_performance_metrics(val_series[idx], preds)
    train_series[idx].plot(ax=axis)
    val_series[idx].plot(ax=axis, label="actual")
    preds.plot(ax=axis, label="forecast")
    axis.legend(train_series[idx].components)
    axis.set_title("")
    plt.tight_layout()
    
    
    
# Apply pre-trained model to different time series
# Read CSV with 8 different monthly data time series into a PANDAS df    



df = pd.read_csv("C:/Users/raltieri/OneDrive - Three Ireland/Masters/Project/Coding/Dataset2.csv",
                 delimiter=",", index_col=0, parse_dates=True, dayfirst=True)



# totalCorePgwThroughputGbps
series_totalCorePgwThroughputGbps = TimeSeries.from_dataframe(df[["totalCorePgwThroughputGbps"]].dropna(), value_cols="totalCorePgwThroughputGbps")
# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(series_totalCorePgwThroughputGbps, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# FWA_peakTotalGbps

FWA_peakTotalGbps_series = TimeSeries.from_dataframe(df[["MBB+FWA_peakTotalGbps"]].dropna(), value_cols="MBB+FWA_peakTotalGbps")
# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(FWA_peakTotalGbps_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

#Handset_peakTotalGbps
Handset_peakTotalGbps_series = TimeSeries.from_dataframe(df[["Handset_peakTotalGbps"]].dropna(), value_cols="Handset_peakTotalGbps")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(Handset_peakTotalGbps_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# peakDailySubscriberPerVlr

peakDailySubscriberPerVlr_series = TimeSeries.from_dataframe(df[["peakDailySubscriberPerVlr"]].dropna(), value_cols="peakDailySubscriberPerVlr")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(peakDailySubscriberPerVlr_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# totalRanVolumePiB

totalRanVolumePiB_series = TimeSeries.from_dataframe(df[["totalRanVolumePiB"]].dropna(), value_cols="totalRanVolumePiB")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(totalRanVolumePiB_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# OBRPeak Mbps

OBRPeak_series = TimeSeries.from_dataframe(df[["OBRPeak Mbps"]].dropna(), value_cols="OBRPeak Mbps")

# Split in train_series/val_series/test_series

#Remove COVID months
#OBRPeak_series = OBRPeak_series[-27:] 
train_series, val_series = split_train_val(OBRPeak_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# IBRPeak Mbps 
IBRPeak_series = TimeSeries.from_dataframe(df[["IBRPeak Mbps"]].dropna(), value_cols="IBRPeak Mbps")

# Split in train_series/val_series/test_series
#Remove COVID months
#IBRPeak_series = IBRPeak_series[-27:] 

train_series, val_series = split_train_val(IBRPeak_series, val_len)

#best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# Now retrain each series with the choosen hyperparameters
# Re-Define test size (witho no validation)
# totalCorePgwThroughputGbps
# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(series_totalCorePgwThroughputGbps, val_len)

best_model = build_fit_nbeats_model(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# FWA_peakTotalGbps

FWA_peakTotalGbps_series = TimeSeries.from_dataframe(df[["MBB+FWA_peakTotalGbps"]].dropna(), value_cols="MBB+FWA_peakTotalGbps")
# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(FWA_peakTotalGbps_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

#Handset_peakTotalGbps
Handset_peakTotalGbps_series = TimeSeries.from_dataframe(df[["Handset_peakTotalGbps"]].dropna(), value_cols="Handset_peakTotalGbps")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(Handset_peakTotalGbps_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# peakDailySubscriberPerVlr

peakDailySubscriberPerVlr_series = TimeSeries.from_dataframe(df[["peakDailySubscriberPerVlr"]].dropna(), value_cols="peakDailySubscriberPerVlr")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(peakDailySubscriberPerVlr_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# totalRanVolumePiB

totalRanVolumePiB_series = TimeSeries.from_dataframe(df[["totalRanVolumePiB"]].dropna(), value_cols="totalRanVolumePiB")

# Split in train_series/val_series/test_series
train_series, val_series = split_train_val(totalRanVolumePiB_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# OBRPeak Mbps

OBRPeak_series = TimeSeries.from_dataframe(df[["OBRPeak Mbps"]].dropna(), value_cols="OBRPeak Mbps")

# Split in train_series/val_series/test_series

#Remove COVID months
#OBRPeak_series = OBRPeak_series[-27:] 
train_series, val_series = split_train_val(OBRPeak_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)

# IBRPeak Mbps 
IBRPeak_series = TimeSeries.from_dataframe(df[["IBRPeak Mbps"]].dropna(), value_cols="IBRPeak Mbps")

# Split in train_series/val_series/test_series
#Remove COVID months
#IBRPeak_series = IBRPeak_series[-27:] 

train_series, val_series = split_train_val(IBRPeak_series, val_len)

best_model = build_fit_nbeats_model_no_chkpt(**best_params)
best_preds = best_model.predict(series=train_series, n=len(val_series)+12*3, mc_dropout=True)
plot_best_trial(train_series, val_series, best_preds)
print_performance_metrics(val_series, best_preds)


