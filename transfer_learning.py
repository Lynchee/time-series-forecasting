import ast
import json
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
from sklearn.utils import shuffle
import os
import time
# import psutil
np.random.seed(1992)
tf.keras.utils.set_random_seed(1992)


class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, startEpoch, displyID):
        self.total_epochs = total_epochs
        self.displyID = displyID
        self.epoch_count = startEpoch

    def on_epoch_end(self, epoch, logs=None):

        self.epoch_count += 1
        logs = logs or {}
        progress = {
            "epoch": self.epoch_count,
            "total_epochs": self.total_epochs,
            "loss": round(logs.get('loss'), 7),
            "val_loss": round(logs.get('val_loss'), 7),
            "lr": round(float(tf.keras.backend.get_value(self.model.optimizer.lr)), 7)
        }

        # Read existing data
        if os.path.exists(self.displyID):
            with open(self.displyID, 'r') as f:
                status_data = json.load(f)
        else:
            status_data = []

        # Append new progress
        status_data.append(progress)
        # Write updated data back to file
        with open(self.displyID, 'w') as f:
            json.dump(status_data, f)


def trainModel(modelPath, baseModel, DataX_train, y_train, displayID, firstEpochs, secondEpochs):

    BatchSize = 128
    totalEpochs = firstEpochs + secondEpochs

    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-15,
        verbose=0
    )

    def lr_schedule(epoch, initial_lr=0.01, warmup_epochs=30, warmup_factor=0.5):
        if epoch < warmup_epochs:
            return initial_lr * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
        return initial_lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr_schedule(epoch)
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=modelPath,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_freq="epoch"
    )

    baseModel.compile(loss='mean_squared_error', optimizer=opt)

    # Custom callback for Streamlit
    streamlit_callback = StreamlitCallback(totalEpochs, 0, displayID)

    hist = baseModel.fit(
        DataX_train,
        y_train,
        validation_split=0.2,
        batch_size=BatchSize,
        epochs=firstEpochs,
        verbose=0,
        callbacks=[checkpoint, lr_scheduler, streamlit_callback],
        shuffle=True
    )

    # Additional garbage collection to ensure memory is freed
    gc.collect()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    baseModel.compile(loss='mean_squared_error',
                      optimizer=opt)

    # Custom callback for Streamlit
    streamlit_callback = StreamlitCallback(
        totalEpochs, firstEpochs, displayID)

    hist = baseModel.fit(DataX_train,
                         y_train,
                         validation_split=0.2,
                         batch_size=BatchSize,
                         epochs=secondEpochs,
                         verbose=0,
                         callbacks=[checkpoint, reduce_lr,
                                    streamlit_callback],
                         shuffle=True)

    # Clear the session to free up memory
    tf.keras.backend.clear_session()

    # Delete objects
    del baseModel, opt, streamlit_callback, hist, reduce_lr, lr_scheduler, checkpoint
    baseModel = None
    opt = None
    streamlit_callback = None
    hist = None
    reduce_lr = None
    lr_scheduler = None
    checkpoint = None
    gc.collect()


def MinMaxScale(df):
    df_norm = pd.DataFrame([])
    min = []
    max = []
    MinMaxScaleColumnName = []
    for c in df.columns:
        new_df = (df[[c]]-df[[c]].min())/(df[[c]].max()-df[[c]].min())
        min.append(df[[c]].min())
        max.append(df[[c]].max())
        MinMaxScaleColumnName.append(c)

        df_norm = pd.concat([df_norm, new_df], axis=1)

    return df_norm, min, max


def MinMaxScaleForTest(df, minScale, maxScale):
    df_norm = pd.DataFrame([])
    for i, c in enumerate(df.columns):
        new_df = (df[[c]]-minScale[i][c])/(maxScale[i][c]-minScale[i][c])
        df_norm = pd.concat([df_norm, new_df], axis=1)

    return df_norm


def generate_data(X, sequence_length=14, step=1):
    X_local = []
    y_local = []
    for start in range(0, len(X) - sequence_length-1, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(X[end][0])
    return np.array(X_local), np.array(y_local)


def MinMaxScaleForUserInput(df, minScale, maxScale, status):
    if status == 'Confirmed':
        i = 0
    elif status == 'Deaths':
        i = 1
    df_norm = pd.DataFrame([])
    for c in df.columns:
        new_df = (df[[c]]-minScale[i][c]) / \
            (maxScale[i][c]-minScale[i][c])
        df_norm = pd.concat([df_norm, new_df], axis=1)

    return df_norm


def preprocessData(country, status):
    # ---------------------------- Preproces data --------------------------------
    df_Confirmed = pd.read_csv(
        'UCovid-19PMFit/data/Preparing_time_series_covid19_confirmed_global.csv')
    df_Deaths = pd.read_csv(
        'UCovid-19PMFit/data/Preparing_time_series_covid19_deaths_global.csv')

    _df_Confirmed = df_Confirmed[df_Confirmed['Country/Region']
                                 == country].drop(columns="Country/Region")
    _df_Deaths = df_Deaths[df_Deaths['Country/Region']
                           == country].drop(columns="Country/Region")

    _df_Confirmed = _df_Confirmed.T
    _df_Confirmed.columns = ["Confirmed"]
    df_Confirmed_train = _df_Confirmed.iloc[:-30]
    df_Confirmed_test = _df_Confirmed.iloc[-45:]

    _df_Deaths = _df_Deaths.T
    _df_Deaths.columns = ["Deaths"]
    df_Deaths_train = _df_Deaths.iloc[:-30]
    df_Deaths_test = _df_Deaths.iloc[-45:]

    data = pd.concat([df_Confirmed_train, df_Deaths_train], axis=1)
    data, minScale, maxScale = MinMaxScale(data)

    data_test = pd.concat([df_Confirmed_test, df_Deaths_test], axis=1)
    data_test = MinMaxScaleForTest(data_test, minScale, maxScale)

    X_sequence, y = generate_data(data.loc[:, [status]].values)
    X_sequence_test, y_test = generate_data(data_test.loc[:, [status]].values)

    X_sequence, y = shuffle(X_sequence, y, random_state=0)
    x_train, y_train = X_sequence, y

    return x_train, y_train, minScale, maxScale


def MinMaxScaleInverse(y, minScale, maxScale, col):
    if col == 'Confirmed':
        c = 0
    elif col == 'Deaths':
        c = 1

    y = y*(maxScale[c][col]-minScale[c][col])+minScale[c][col]
    return y


def getPredictionResults(baseModelName, numberInputList, nextSteps, minScale, maxScale, status, modelPath, predID):

    # Load the best weight
    best_model = tf.keras.models.load_model(modelPath)

    # Preprocessin user's input
    _data_test = pd.DataFrame({status: numberInputList})

    data_test = MinMaxScaleForUserInput(_data_test, minScale, maxScale, status)

    # Predict a result
    # Read existing data
    if os.path.exists(predID):
        with open(predID, 'r') as f:
            predDict = json.load(f)

    else:
        predDict = {}

    _inputArray = np.array([data_test])
    for dayth in range(nextSteps):
        _DataX_valid = _inputArray[:, -14:, :]

        y_pred = best_model.predict(
            _DataX_valid, verbose=0)

        # Concatenate the arrays along the second axis (axis=1)
        _inputArray = np.concatenate(
            (_inputArray, y_pred.reshape(1, 1, 1)), axis=1)

    # Get normal scale back
    _inputArray = MinMaxScaleInverse(
        _inputArray.flatten(), minScale, maxScale, status)

    predDict[baseModelName] = _inputArray.tolist()

    # Write updated data back to file
    with open(predID, 'w') as f:
        json.dump(predDict, f)

    # Remove model
    os.remove(modelPath)


if __name__ == "__main__":

    # for proc in psutil.process_iter(['pid', 'name']):
    #     if 'python' in proc.info['name']:
    #         print(f"PID: {proc.info['pid']}, Name: {proc.info['name']}")

    # time.sleep(20)
    modelPath = sys.argv[1]
    displayID = sys.argv[2]
    country = sys.argv[3]
    baseModelName = sys.argv[4]
    status = sys.argv[5]
    numberInputListStr = sys.argv[6]
    nextStepsStr = sys.argv[7]
    predID = sys.argv[8]
    firstEpochsStr = sys.argv[9]
    secondEpochsStr = sys.argv[10]

    numberInputList = ast.literal_eval(numberInputListStr)
    nextSteps = int(nextStepsStr)
    firstEpochs = int(firstEpochsStr)
    secondEpochs = int(secondEpochsStr)

    # Preprocessing data
    x_train, y_train, minScale, maxScale = preprocessData(country, status)

    # Load a pretrain model
    baseModel = tf.keras.models.load_model(
        f'UCovid-19PMFit/weights/{status}/ModelWeight_{baseModelName}_PretrainedModel({status}).h5')

    # Train the pretrain model
    trainModel(modelPath, baseModel, x_train, y_train,
               displayID, firstEpochs, secondEpochs)

    # Prdiction
    getPredictionResults(baseModelName, numberInputList,
                         nextSteps, minScale, maxScale, status, modelPath, predID)
    print('End')
