import json
import subprocess
import streamlit as st
import pandas as pd
import math
import os
import plotly.graph_objects as go
import gc
import time
import datetime
import locale

locale.setlocale(locale.LC_ALL, '')

initInputDict = {'D1': 4479866,
                 'D2': 4483912,
                 'D3': 4487818,
                 'D4': 4491864,
                 'D5': 4495632,
                 'D6': 4499259,
                 'D7': 4503026,
                 'D8': 4506933,
                 'D9': 4510421,
                 'D10': 4513769,
                 'D11': 4516978,
                 'D12': 4520466,
                 'D13': 4523536,
                 'D14': 4526605,
                 }


@st.cache_resource
def getPretrainData():
    df_Confirmed = pd.read_csv(
        'UCovid-19PMFit/data/Preparing_time_series_covid19_confirmed_global.csv')
    df_Deaths = pd.read_csv(
        'UCovid-19PMFit/data/Preparing_time_series_covid19_deaths_global.csv')

    countryList = df_Confirmed['Country/Region'].values
    return df_Confirmed, df_Deaths, countryList


df_Confirmed, df_Deaths, countryList = getPretrainData()

# Function to display centered title using CSS


def centered_title(title):
    st.markdown(
        f'<h2 style="text-align: center;">{title}</h2>', unsafe_allow_html=True)


def section_title(text, h='h4'):
    st.markdown(
        f'<{h} style="text-align: left;">{text}</{h}>', unsafe_allow_html=True)

# Function to add an indent to text using Markdown


def indented_text(text, indent_level=1):
    # Four spaces per level of indentation
    indent_spaces = '&nbsp;' * 4 * indent_level
    indented_text = f"{indent_spaces}{text}"
    st.markdown(indented_text, unsafe_allow_html=True)


centered_title(
    "Universal Covid-19 Time Series Pattern Model Fine-tuning for Covid-19 Time Series Forecasting (UCovid-19PMFit)")

indented_text("The researcher is interested in applying the concepts of Natural language processing (NLP) and transfer learning to improve the ability to forecast the time series of the COVID-19 pandemic spread. The proposed approach involves designing a concept of learning through a general domain of time series patterns from multiple countries, totaling over 200 countries. This concept is referred to as the Universal Covid-19 Time Series Pattern Model Fine-tuning for Covid-19 Time Series Forecasting (UCovid-19PMFit). Subsequently, this model is fine-tuned to learn and forecast the pandemic spread in countries of interest.", indent_level=4)

st.image('cover.jpg')

indented_text("From the figure above, the methodology involves collecting COVID-19 confirmed/death case data from several countries and randomly selecting time series patterns from this data. These patterns are labeled as either 'Next Pattern' (1) or 'Not Next Pattern' (0) based on their sequence continuity. A universal pattern prediction model is then pre-trained using LSTM/GRU/BiLSTM/BiGRU networks for binary classification. This universal model is then fine-tuned for a specific target country through transfer learning, allowing the model to leverage the learned features and improve prediction accuracy for the target country's COVID-19 case trends.", indent_level=4)
indented_text("This web application applied this concept to forecast the target country showing the graphs of validation loss and predicted graph of confirmed/death cases in the next steps.", indent_level=4)

st.write('-----------')

# Get today's date
today = datetime.date.today()

st.subheader("Input data", divider='rainbow')
headCol1, headCol2, headCol3, headCol4 = st.columns(4)
with headCol1:
    country = st.selectbox("Country:", countryList, index=177)

with headCol2:
    status = st.selectbox("Status", ["Confirmed", "Deaths"])

with headCol3:

    startDate = st.date_input("Start date", today)

with headCol4:
    # Calculate the date 14 days from today
    endDate = st.date_input("End date", today + datetime.timedelta(days=13))

nextSteps = st.slider(
    label="Next steps (days)", min_value=1, max_value=90, step=1)


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


# ------------------------------------- Train model -----------------------------------
st.subheader("Fine-Tune Pretrained Models", divider='rainbow')


numberInputList = []
for i in range(14):
    numberInputList.append(st.session_state[f'D{i+1}'])

numberInputListStr = str(numberInputList)
nextStepsStr = str(nextSteps)

submitBtn = st.button('Submit')

excID = time.time()
predID = f"{excID}_predictions.json"
baseModelNames = ['LSTM', 'GRU', 'BiLSTM', 'BiGRU']
firstEpochs = 30
secondEpochs = 150
totalEpochs = firstEpochs + secondEpochs

firstEpochsStr = str(firstEpochs)
secondEpochsStr = str(secondEpochs)
if submitBtn:

    baseModelDict = dict()
    baseHistoryDict = dict()
    for baseModelName in baseModelNames:

        st.write(baseModelName)

        modelPath = f"Models/{excID}_{country}_{baseModelName}_{status}.h5"
        displayID = f"{excID}_{country}_{baseModelName}_{status}.json"

        # Run the training script
        process = subprocess.Popen(
            ["python", "transfer_learning.py", modelPath,
             displayID, country, baseModelName, status, numberInputListStr,
             nextStepsStr, predID, firstEpochsStr, secondEpochsStr])

        # Display progress
        progress_bar = st.progress(0)
        epochth = 1
        resultDisplay = st.empty()

        for i in range(600):
            if os.path.exists(displayID):
                try:
                    with open(displayID, 'r') as f:
                        if os.path.getsize(displayID) > 0:
                            progress = json.load(f)
                        else:
                            continue
                except json.JSONDecodeError:
                    # Wait a bit and try again if there's a JSON error
                    time.sleep(0.1)
                    continue

                if process:
                    newProgress = None
                    for p in progress:
                        if p['epoch'] == epochth:
                            newProgress = p
                            break
                    if newProgress:

                        progress_bar.progress(
                            int(newProgress['epoch']) / int(newProgress['total_epochs']))
                        resultDisplay.text(
                            f"Epoch {newProgress['epoch']}/{newProgress['total_epochs']}, Loss: {newProgress['loss']}, Val Loss: {newProgress['val_loss']}, lr: {newProgress['lr']}")

                        epochth += 1
                        if epochth > totalEpochs:
                            loss = []
                            val_loss = []
                            for obj in progress:
                                loss.append(obj['loss'])
                                val_loss.append(obj['val_loss'])
                            baseHistoryDict[baseModelName] = {
                                'loss': loss, 'val_loss': val_loss}
                            os.remove(displayID)
                            break
            time.sleep(0.1)

    # ----------------------------- Results -----------------------------------
     # Create the figure
    fig = go.Figure()
    x = list(range(1, len(baseHistoryDict["LSTM"]['val_loss'])+1))
    for modelName in baseModelNames:
        fig.add_trace(go.Scatter(
            x=x[50:], y=baseHistoryDict[modelName]['val_loss'][50:], mode='lines+markers', name=modelName))

    # Update layout
    section_title("Validation loss")
    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title=f'MSE',
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

    # Best validation loss
    section_title("Minimum validation loss")
    for modelName in baseModelNames:
        st.write(
            f"**{modelName}:** {round(min(baseHistoryDict[modelName]['val_loss']),7)}")

    st.write('-----------')
    if os.path.exists(predID):
        for _ in range(10):
            # Read from the JSON file
            with open(predID, 'r') as json_file:
                predDict = json.load(json_file)

            if len(predDict.keys()) == 4:
                os.remove(predID)
                break
            time.sleep(1)

    st.subheader("Model Results", divider='rainbow')

    # Create a list of dates from startDate to startDate
    x_dates = [startDate + datetime.timedelta(days=i)
               for i in range(len(predDict['LSTM']))]

    # Create the figure
    fig = go.Figure()

    # Add the first trace
    fig.add_trace(go.Scatter(
        x=x_dates[:14], y=predDict['LSTM'][:14], mode='lines+markers', name='Inputs'))

    for modelName in baseModelNames:
        fig.add_trace(go.Scatter(
            x=x_dates[14:], y=predDict[modelName][14:], mode='lines+markers', name=modelName))

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

    for modelName in baseModelNames:

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

        # Create a list of dates from startDate to startDate + 13 days
        x_dates = [startDate +
                   datetime.timedelta(days=i) for i in range(len(data))]

        fig = go.Figure()

        # Add shaded area between the bounds
        fig.add_trace(go.Scatter(
            x=x_dates + x_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.05)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False,
            name='Confidence Interval'
        ))

        fig.add_trace(go.Scatter(x=x_dates, y=upper_bound,
                                 mode='lines', showlegend=False, name="Upper bound", line=dict(color='red', width=0.5)))

        fig.add_trace(go.Scatter(x=x_dates, y=lower_bound,
                                 mode='lines', showlegend=False, name="Lower bound", line=dict(color='red', width=0.5)))

        # Add trace for the data points
        fig.add_trace(go.Scatter(
            x=x_dates[:-nextSteps], y=data[:-nextSteps], mode='lines+markers', name='Observed'))

        fig.add_trace(go.Scatter(
            x=x_dates[-nextSteps:], y=data[-nextSteps:], mode='lines+markers', name=modelName))

        # Add trace for the mean
        fig.add_trace(go.Scatter(
            x=x_dates, y=[mean]*len(data), mode='lines', name='Mean', line=dict(color='green', dash='dash')))

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

        def readableFormatNumber(number):
            if math.isnan(number):
                return "NaN"
            else:
                return locale.format_string("%d", round(number, 1), grouping=True)

        st.write(f"**Mean:** {readableFormatNumber(mean)}")
        st.write(f"**variance:** {readableFormatNumber(variance)}")
        st.write(f"**Standard deviation:** {readableFormatNumber(std_dev)}")
