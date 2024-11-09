import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier

import prometheus_client as prom

# FastAPI object
app = FastAPI()

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df

def load_dataset():
    file_name = 'heart_failure_clinical_records_dataset.csv'    
    df = pd.read_csv(Path(f"{parent}/{file_name}"))
    outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    df1 = df.copy()
    for colm in outlier_colms:
        df1 = handle_outliers(df1, colm)
    
    return df1

data = load_dataset()    # read complete data

acc_metric = prom.Gauge('patient_survival_prediction', 'Accuracy score')
f1_metric =  prom.Gauge('patient_survival_f1score', 'F1 score')

loaded_model = joblib.load(f"{parent}/xgboost-model.pkl")


#X = data.iloc[:, :-1].values
#y = data['DEATH_EVENT'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 123)

#test_data = X_test.copy()
#test_data['target'] = y_test.copy()



# Function for updating metrics
def update_metrics():
    test_data = data.sample(100)
    test_features = test_data.iloc[:, :-1].values
    test_target = test_data['DEATH_EVENT'].values
    # Performance on test set
    y_pred = loaded_model.predict(test_features)             # prediction
    acc = accuracy_score(test_target, y_pred).round(3)                    # accuracy score
    f1 = f1_score(test_target, y_pred).round(3)                           # F1 score
    
    acc_metric.set(acc)
    f1_metric.set(f1)
  

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())


################################# Prometheus related code END ######################################################


# UI - Input components
in_Age = gradio.Slider(20, 90, value="79", label="Age", info="Choose between 20 and 90")
#gradio.Textbox(lines=1, placeholder=None, value="79", label='Age')
in_Anaemia = gradio.Radio(['0', '1'], type="value", label='Anaemia')
in_Cret = gradio.Slider(200,700, value="589", label='CReatinine phosphokinase')
in_Diabetes = gradio.Radio(['0', '1'], type="value", label='Diabetes')
#in_Ejection = gradio.Textbox(lines=1, placeholder=None, value="38", label='Ejection fraction')
in_Ejection = gradio.Slider(20,60, value="38", label='Ejection fraction')
in_High_BP = gradio.Radio(['0', '1'], type="value", label='High blood pressure')
#in_Platelets = gradio.Textbox(lines=1, placeholder=None, value="265000", label='Platelets')
in_Platelets = gradio.Slider(20000,80000, value="265000", label='Platelets')
in_Sex = gradio.Radio(['0', '1'], type="value", label='Sex')
#in_Serum_Creat = gradio.Textbox(lines=1, placeholder=None, value="1.1", label='Serum creatinine')
in_Serum_Creat = gradio.Slider(0.5,2.0, value="1.1", label='Serum creatinine')
#in_Serum_Sodium = gradio.Textbox(lines=1, placeholder=None, value="135", label='Serum sodium')
in_Serum_Sodium = gradio.Slider(100,250,value="135", label='Serum sodium')
in_Smoking = gradio.Radio(['0', '1'], type="value", label='Smoking')
#in_Time = gradio.Textbox(lines=1, placeholder=None, value="4", label='Follow-up period')
in_Time = gradio.Slider(2,6, value="4", label='Follow-up period')


# Output response
# YOUR CODE HERE
out_Death_Event = gradio.Textbox(lines=1, placeholder=None, label='Death event')

# Label prediction function
def predict_death_event(age,anaemia, cret, diabetes, ejection, bp, platelets, sex, serum_creat, serum_sodium, smoking, time):
    '''Function to predict survival of patients with heart failure'''
    input_data = np.array([[age, anaemia, bp, cret, diabetes, ejection, platelets, sex, serum_creat, serum_sodium, smoking, time]],dtype=object)
    prediction = loaded_model.predict(input_data)[0]
    #print(result)
    label = "Deceased" if prediction == 1 else "Survived"
    #print(label)
    return label

# Create Gradio interface object
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [in_Age,in_Anaemia, in_Cret, in_Diabetes, in_Ejection, in_High_BP, in_Platelets, in_Sex, in_Serum_Creat, in_Serum_Sodium, in_Smoking, in_Time],
                         outputs = [out_Death_Event],
                         title = title,
                         description = description,
                         allow_flagging='never')

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
