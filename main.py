import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import backend as K
import math
import os
import pickle
import plotly.figure_factory as ff
import plotly.graph_objects as go

initInputDict = {'D1': 4479866.23623163,
                 'D2': 4483912.32937843,
                 'D3': 4487818.9020719,
                 'D4': 4491864.9952187,
                 'D5': 4495632.04745883,
                 'D6': 4499259.57924562,
                 'D7': 4503026.63148574,
                 'D8': 4506933.20417921,
                 'D9': 4510421.21551266,
                 'D10': 4513769.70639278,
                 'D11': 4516978.67681955,
                 'D12': 4520466.688153,
                 'D13': 4523536.13812644,
                 'D14': 4526605.58809988,
                 }


@st.cache_resource
def getDataDict():
    with open('dataDict.pickle', 'rb') as handle:
        data = pickle.load(handle)
        # Get country list
        countryList = data.keys()
        return data, countryList


dataDict, countryList = getDataDict()

st.title("UCovid-19PMFit")
st.write("  The researcher is interested in applying the concepts of NLP and transfer learning to improve the ability to forecast the time series of the COVID-19 pandemic spread. The proposed approach involves designing a concept of learning through a General domain of time series patterns from multiple countries, totaling over 200 countries. This concept is referred to as the Universal Covid-19 Time Series Pattern Model Fine-tuning for Covid-19 Time Series Forecasting (UCovid-19PMFit). Subsequently, this model is fine-tuned to learn and forecast the pandemic spread in countries of interest. The 15 countries for which the model is fine-tuned to forecast the pandemic spread are Thailand, Malaysia, Japan, India, Vietnam, Norway, United Kingdom, Italy, Spain, France, Canada, Mexico, Cuba, Brazil, and Argentina.")
st.write('-----------')
# countryList = ['Thailand', 'UK']
# print([ind for ind, _country in enumerate(
#     countryList) if _country == 'Thailand'][0])
st.subheader("Input data", divider='rainbow')
headCol1, headCol2, _, _ = st.columns(4)
with headCol1:
    country = st.selectbox("Country:", countryList, index=177)

with headCol2:
    status = st.selectbox("Status", ["Confirmed", "Deaths"])

nextSteps = st.slider(
    label="Next steps (days)", min_value=1, max_value=30, step=1)


st.write(f'{status} over 14 days')


colD1, colD2, colD3, colD4, colD5, colD6 = st.columns(6)
with colD1:
    numD1 = st.number_input("D1", value=initInputDict['D1'],  key='D1')

with colD2:
    numD2 = st.number_input("D2", value=initInputDict['D2'],  key='D2')

with colD3:
    numD3 = st.number_input("D3", value=initInputDict['D3'], key='D3')

with colD4:
    numD4 = st.number_input("D4", value=initInputDict['D4'],  key='D4')

with colD5:
    numD5 = st.number_input("D5", value=initInputDict['D5'],
                            placeholder="", label_visibility="visible", key='D5')

with colD6:
    numD6 = st.number_input("D6", value=initInputDict['D6'],
                            placeholder="", label_visibility="visible", key='D6')


colD7, colD8, colD9, colD10, colD11, colD12 = st.columns(6)
with colD7:
    numD7 = st.number_input("D7", value=initInputDict['D7'],
                            placeholder="", label_visibility="visible", key='D7')

with colD8:
    numD8 = st.number_input("D8", value=initInputDict['D8'],
                            placeholder="", label_visibility="visible", key='D8')

with colD9:
    numD9 = st.number_input("D9", value=initInputDict['D9'],
                            placeholder="", label_visibility="visible", key='D9')

with colD10:
    numD10 = st.number_input("D10", value=initInputDict['D10'],
                             placeholder="", label_visibility="visible", key='D10')

with colD11:
    numD11 = st.number_input("D11", value=initInputDict['D11'],
                             placeholder="", label_visibility="visible", key='D11')

with colD12:
    numD12 = st.number_input("D12", value=initInputDict['D12'],
                             placeholder="", label_visibility="visible", key='D12')

colD13, colD14, _, _, _, _ = st.columns(6)
with colD13:
    numD13 = st.number_input("D13", value=initInputDict['D13'],
                             placeholder="", label_visibility="visible", key='D13')
with colD14:
    numD13 = st.number_input("D14", value=initInputDict['D14'],
                             placeholder="", label_visibility="visible", key='D14')

st.write('-----------')
# ---------------------------- Preproces data --------------------------------
# df_Confirmed = pd.read_csv(
#     'UCovid-19PMFit/data/Preparing_time_series_covid19_confirmed_global.csv')
# df_Deaths = pd.read_csv(
#     'UCovid-19PMFit/data/Preparing_time_series_covid19_deaths_global.csv')

# Country = country
# df_Confirmed = df_Confirmed[df_Confirmed['Country/Region']
#                             == Country].drop(columns="Country/Region")
# df_Deaths = df_Deaths[df_Deaths['Country/Region']
#                       == Country].drop(columns="Country/Region")

# df_Confirmed = df_Confirmed.T
# df_Confirmed.columns = ["Confirmed"]
# df_Confirmed_train = df_Confirmed.iloc[:-30]
# df_Confirmed_test = df_Confirmed.iloc[-45:]

# df_Deaths = df_Deaths.T
# df_Deaths.columns = ["Deaths"]
# df_Deaths_train = df_Deaths.iloc[:-30]
# df_Deaths_test = df_Deaths.iloc[-45:]

# data = pd.concat([df_Confirmed_train, df_Deaths_train], axis=1)


# def MinMaxScale(df):
#     df_norm = pd.DataFrame([])
#     min = []
#     max = []
#     MinMaxScaleColumnName = []
#     for c in df.columns:
#         new_df = (df[[c]]-df[[c]].min())/(df[[c]].max()-df[[c]].min())
#         min.append(df[[c]].min())
#         max.append(df[[c]].max())
#         MinMaxScaleColumnName.append(c)
#         df_norm = pd.concat([df_norm, new_df], axis=1)
#     return df_norm, min, max


# data, minScale, maxScale = MinMaxScale(data)
# data_test = pd.concat([df_Confirmed_test, df_Deaths_test], axis=1)


# def MinMaxScaleForTest(df, minScale, maxScale):
#     df_norm = pd.DataFrame([])
#     for i, c in enumerate(df.columns):
#         new_df = (df[[c]]-minScale[i][c])/(maxScale[i][c]-minScale[i][c])
#         df_norm = pd.concat([df_norm, new_df], axis=1)

#     return df_norm


# data_test = MinMaxScaleForTest(data_test, minScale, maxScale)

# sequence_length = 14


# def generate_data(X, sequence_length=14, step=1):
#     X_local = []
#     y_local = []
#     for start in range(0, len(X) - sequence_length-1, step):
#         end = start + sequence_length
#         X_local.append(X[start:end])
#         y_local.append(X[end][0])
#     return np.array(X_local), np.array(y_local)


# X_sequence, y = generate_data(data.loc[:, ["Confirmed"]].values)
# X_sequence_test, y_test = generate_data(data_test.loc[:, ["Deaths"]].values)


# X_sequence, y = shuffle(X_sequence, y, random_state=0)
# X_sequence.shape, y.shape

# DataX_train, y_train = X_sequence, y
# DataX_valid, y_valid = X_sequence_test, y_test


# print(DataX_train.shape, DataX_valid.shape)
# print(y_train.shape, y_test.shape)


# Load data
DataX_train = dataDict[country]['xTrain']
y_train = dataDict[country]['yTrain']
X_sequence_test = dataDict[country]['xTest']
y_test = dataDict[country]['yTest']
minScale = dataDict[country]['minScale']
maxScale = dataDict[country]['maxScale']


numberInputList = []
for i in range(14):
    numberInputList.append(st.session_state[f'D{i+1}'])


_data_test = pd.DataFrame({status: numberInputList})


def MinMaxScaleForTest(df, minScale, maxScale, status):
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


data_test = MinMaxScaleForTest(_data_test, minScale, maxScale, status)

DataX_valid = np.array([data_test])


# ----------------------------------- Load pretrain model -----------------------------
@st.cache_resource
def getPretrainModel():
    # Load the trained model
    # Fetch data from URL here, and then clean it up.
    pretrainModelDict = dict()
    pretrainModelDict["Deaths"] = [('LSTM', tf.keras.models.load_model(
        'UCovid-19PMFit/weights/Deaths/ModelWeight_LSTM_PretrainedModel(Deaths).h5')),
        ('GRU', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Deaths/ModelWeight_GRU_PretrainedModel(Deaths).h5')),
        ('BiLSTM', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Deaths/ModelWeight_BiLSTM_PretrainedModel(Deaths).h5')),
        ('BiGRU', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Deaths/ModelWeight_BiGRU_PretrainedModel(Deaths).h5'))]

    pretrainModelDict['Confirmed'] = [('LSTM', tf.keras.models.load_model(
        'UCovid-19PMFit/weights/Confirmed/ModelWeight_LSTM_PretrainedModel(Confirmed).h5')),
        ('GRU', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Confirmed/ModelWeight_GRU_PretrainedModel(Confirmed).h5')),
        ('BiLSTM', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Confirmed/ModelWeight_BiLSTM_PretrainedModel(Confirmed).h5')),
        ('BiGRU', tf.keras.models.load_model(
            'UCovid-19PMFit/weights/Confirmed/ModelWeight_BiGRU_PretrainedModel(Confirmed).h5'))]
    return pretrainModelDict


pretrainModelDict = getPretrainModel()


# ------------------------------------- Train model -----------------------------------
st.subheader("Fine-Tune Pretrained Models", divider='rainbow')


def MinMaxScaleInverse(y, minScale, maxScale, col, c):
    y = y*(maxScale[c][col]-minScale[c][col])+minScale[c][col]
    return y


def S_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / ((np.abs(y_true)+np.abs(y_pred))/2))) * 100


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) /
                 K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)


def root_mean_squared_log_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.mean((np.log(y_true)-np.log(y_pred))**2))**(1/2)


def explained_variance(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 1-(np.var(y_pred-y_true)/np.var(y_true))


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate  # Get the current learning rate
        self.learning_rates.append(lr)
        # print(f'Epoch {epoch + 1}: Learning Rate = {lr.numpy()}')


def lr_schedule(epoch, initial_lr=0.01, warmup_epochs=30, warmup_factor=0.5):
    if epoch < warmup_epochs:
        return initial_lr * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
    return initial_lr


class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, totalEpochs, display_id):
        super().__init__()
        self.totalEpochs = totalEpochs
        self.display_id = display_id
        self.progress_bar = st.progress(0)
        self.loss_text = st.empty()
        # self.val_loss_text = st.empty()
        # self.lr_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.totalEpochs
        self.progress_bar.progress(progress)
        self.loss_text.text(
            f"Epoch {epoch+1}/{self.totalEpochs}: loss: {logs['loss']:.6f}, val Loss: {logs['val_loss']:.6f} lr: {tf.keras.backend.get_value(self.model.optimizer.lr):.7f}")
        # self.val_loss_text.text(f"")
        # self.lr_text.text(
        #     f"Learning Rate: {tf.keras.backend.get_value(self.model.optimizer.lr):.6f}")


trainBtn = st.button('Submit')
if trainBtn:
    baseModelDict = dict()
    for baseModelName, Final_model in pretrainModelDict[status]:

        # list_loss_train = []
        # list_loss_val = []
        # list_lr = []
        st.write(baseModelName)
        BatchSize = 128
        StartEpochs = 1
        Epochs = 1

        model_path = f"Models/{country}_{baseModelName}_{status}.h5"
        opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Metric to monitor
            factor=0.5,          # Factor by which the learning rate will be reduced
            # Number of epochs with no improvement after which learning rate will be reduced
            patience=15,
            min_lr=1e-15,         # Lower bound on the learning rate
            verbose=1            # Verbosity mode
        )

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_schedule(epoch))

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss', mode='min',
            save_freq="epoch")

        lr_logger = LearningRateLogger()

        Final_model.compile(loss='mean_squared_error',
                            optimizer=opt)

        # Custom callback for Streamlit
        streamlit_callback = StreamlitCallback(StartEpochs, 'training_display')

        hist = Final_model.fit(DataX_train,
                               y_train,
                               validation_split=0.2,
                               batch_size=BatchSize,
                               epochs=StartEpochs,
                               verbose=2,
                               callbacks=[checkpoint,
                                          lr_scheduler,
                                          lr_logger,
                                          streamlit_callback
                                          ],
                               shuffle=True)

        # list_loss_train = list_loss_train + hist.history['loss']
        # list_loss_val = list_loss_val + hist.history['val_loss']
        # list_lr = list_lr + hist.history['lr']

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        Final_model.compile(loss='mean_squared_error',
                            optimizer=opt)

        # Custom callback for Streamlit
        streamlit_callback = StreamlitCallback(Epochs, 'training_display')

        hist = Final_model.fit(DataX_train,
                               y_train,
                               validation_split=0.2,
                               batch_size=BatchSize,
                               epochs=Epochs,
                               verbose=2,
                               callbacks=[checkpoint,
                                          reduce_lr,
                                          lr_logger,
                                          streamlit_callback
                                          ],
                               shuffle=True)

        # list_loss_train = list_loss_train + hist.history['loss']
        # list_loss_val = list_loss_val + hist.history['val_loss']
        # list_lr = list_lr + hist.history['lr']

        # history = {"loss": list_loss_train,
        #            'val_loss': list_loss_val,
        #            'lr': list_lr}

        best_model = tf.keras.models.load_model(model_path)

        baseModelDict[baseModelName] = best_model
        # y_pred = best_model.predict(X_sequence_test)
        # y_pred = y_pred.flatten()

        # y_pred = MinMaxScaleInverse(y_pred, minScale, maxScale, "Deaths", 1)
        # y_test = MinMaxScaleInverse(y_test, minScale, maxScale, "Deaths", 1)

        # st.write(y_pred)
        # st.write(y_test)
        # st.title("Results")

        # --------------------------------------- Results -------------------------------------
        # testScore_RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
        # st.write('Test Score: %.5f RMSE' % (testScore_RMSE))

        # testScore_MAE = mean_absolute_error(y_test, y_pred)
        # st.write('Test Score: %.5f MAE' % (testScore_MAE))

        # testScore_MAPE = mean_absolute_percentage_error(y_test, y_pred)
        # st.write('Test Score: %.5f MAPE' % (testScore_MAPE))

        # testScore_SMAPE = S_mean_absolute_percentage_error(y_test, y_pred)
        # st.write('Test Score: %.5f SMAPE' % (testScore_SMAPE))

        # testScore_RMSLE = root_mean_squared_log_error(y_test, y_pred)
        # st.write('Test Score: %.5f RMSLE' % (testScore_RMSLE))

        # testScore_EV = explained_variance(y_test, y_pred)
        # st.write('Test Score: %.5f EV' % (testScore_EV))

    def MinMaxScaleInverse(y, minScale, maxScale, col):
        if col == 'Confirmed':
            c = 0
        elif col == 'Deaths':
            c = 1

        y = y*(maxScale[c][col]-minScale[c][col])+minScale[c][col]
        return y

    # print('x', DataX_train[:1, :, :])
    # print('rawx', MinMaxScaleInverse(
    #     DataX_train[:1, :, :].flatten(), minScale, maxScale, status))

    # x = np.concatenate((DataX_train[:1, :, :],
    #                    y_train[:1].reshape(1, 1, 1)), axis=1)
    # print('allx', x)
    # print('rawallx', MinMaxScaleInverse(
    #     x.flatten(), minScale, maxScale, status))

    # for modelName in baseModelDict.keys():

    #     pred = baseModelDict[modelName].predict(
    #         DataX_train[:1, :, :], verbose=0)
    #     pred = np.concatenate(
    #         (DataX_train[:1, :, :], pred.reshape(1, 1, 1)), axis=1)

    #     print(modelName)
    #     print('pred', pred)
    #     print('rawPred', MinMaxScaleInverse(
    #         pred.flatten(), minScale, maxScale, status)
        # )
    predDict = {modelName: DataX_valid for modelName in baseModelDict.keys()}
    for modelName in baseModelDict.keys():

        for day in range(nextSteps):
            _DataX_valid = predDict[modelName][:, -14:, :]
            # print(_DataX_valid.shape)
            y_pred = baseModelDict[modelName].predict(_DataX_valid, verbose=0)

            # Concatenate the arrays along the second axis (axis=1)
            predDict[modelName] = np.concatenate(
                (predDict[modelName], y_pred.reshape(1, 1, 1)), axis=1)

        # Get normal scale back
        predDict[modelName] = MinMaxScaleInverse(
            predDict[modelName].flatten(), minScale, maxScale, status)

    # ----------------------------- Results -----------------------------------
    st.write('-----------')

    st.subheader("Model Results", divider='rainbow')

    # Lin chart 1 : predicttion
    days = list(range(1, len(predDict[modelName]) + 1))

    # Create the figure
    fig = go.Figure()

    initX = days[:-nextSteps]
    predX = days[-nextSteps:]

    # Add the first trace
    fig.add_trace(go.Scatter(
        x=initX, y=predDict['LSTM'][:-nextSteps], mode='lines+markers', name='Inputs'))

    for modelName in baseModelDict.keys():
        fig.add_trace(go.Scatter(
            x=predX, y=predDict[modelName][-nextSteps:], mode='lines+markers', name=modelName))

    # Update layout
    fig.update_layout(
        title=f'All models',
        xaxis_title='Days',
        yaxis_title=f'Number of {status} Over Days',
        template='plotly_dark',
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        )
    )

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

    for modelName in baseModelDict.keys():

        data = predDict[modelName]

        # Calculate mean
        mean = sum(data) / len(data)

        # Calculate variance
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)

        # Calculate standard deviation
        std_dev = math.sqrt(variance)

        # Calculate upper and lower bounds for the confidence interval
        upper_bound = [i + 1.96 * std_dev for i in data]
        lower_bound = [i - 1.96 * std_dev for i in data]

        # Create an x-axis representing the index of the data points
        x = list(range(1, len(data) + 1))

        fig = go.Figure()

        # Add shaded area between the bounds
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.05)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False,
            name='Confidence Interval'
        ))

        # Add trace for the data points
        fig.add_trace(go.Scatter(
            x=x[:-nextSteps], y=data[:-nextSteps], mode='lines+markers', name='Observed'))

        fig.add_trace(go.Scatter(
            x=x[-nextSteps:], y=data[-nextSteps:], mode='lines+markers', name=modelName))

        # Add trace for the mean
        fig.add_trace(go.Scatter(
            x=x, y=[mean]*len(data), mode='lines', name='Mean', line=dict(color='green', dash='dash')))

        # Update layout
        fig.update_layout(
            title=modelName,
            xaxis_title='Day',
            yaxis_title=f'Number of {status} Over Days',
            template='plotly_dark',
            xaxis=dict(
                showgrid=True,
                gridcolor='gray',
                gridwidth=0.5
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='gray',
                gridwidth=0.5
            )
        )

        # Show the figure
        st.plotly_chart(fig, use_container_width=True)
