#Importing libraries
from flask import Flask, render_template, request,session,flash,redirect, url_for
from werkzeug.utils import secure_filename

import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from xgboost import XGBRegressor
import pickle
import joblib
from keras.models import load_model

import pydicom as dicom
from PIL import Image




app = Flask(__name__)
app.secret_key='data'

#Set location for uploading images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Defining custom loss functions
C70, C10 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def LaplaceLogLikelihood(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    
    sigma_clip = tf.maximum(y_pred[:, 1], C70)
    
    delta = tf.minimum(tf.abs(y_true[:, 0] - y_pred[:, 0]), C10)
  
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
#=============================#
def regressionloss (y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / y_true[:, 0] )
    #spred = tf.square(y_true, y_pred[:, 0])
    return K.mean(spread)
#=============================#

def OSICloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * LaplaceLogLikelihood(y_true, y_pred) + (1 - _lambda)*regressionloss(y_true, y_pred)
    return loss

#Chart functions
fvc_range=pd.read_csv('fvc_ranges.txt')
def fetch_risk_type(age,sex,fvc):
  fvc_sub=fvc_range[(fvc_range["Gender"] == sex)& 
                    (fvc_range["Age_min"]<=age) & 
                    (fvc_range["Age_max"]>=age) & 
                    (fvc_range["FVC_min"]<=fvc) &
                    (fvc_range["FVC_max"]>=fvc) ]
  risk=fvc_sub["Risk_type"].iloc[0]
  return risk

def donut_chart(age,sex,fvc_all):
  nc=0
  mc=0
  sc=0
  for i in range(len(fvc_all)):
    risk_type=fetch_risk_type(age,sex,fvc_all[i])
    if risk_type=='Normal':
      nc+=1
    elif risk_type=='Mild Risk':
      mc+=1
    elif risk_type=='Severe Risk':
      sc+=1
  
  a=((nc/52)*100)
  b=((mc/52)*100)
  c=((sc/52)*100)
  donut_new=[a,b,c]
  if max(donut_new)==c:
      risk='High Risk'
  elif max(donut_new)==b:
      risk='Mild Risk'
  else:
      risk='No Risk'
      
  return donut_new,risk

def line_chart(fvc_all):
  Month=[4,8,12,16,20,24,28,32,36,40,44,52]
  line=[]
  for i in Month:
    line.append(int(fvc_all[i-1]))
  return line

#Load models
cnn_model=load_model('cnn_new.h5',custom_objects={'loss':OSICloss(0.5),'LaplaceLogLikelihood':LaplaceLogLikelihood})
reg_model=pickle.load(open('reg_model.pkl', 'rb'))

 
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict",methods=["POST"])
def predict():
    error_msg1=''
    error_msg2=''
    
    #Fetch values from the form
    PatientID = request.form.get("patient")
    BaselineWeek = request.form.get("fweeks")
    BaselineFVC = request.form.get("fvc")
    Percent = request.form.get("percent")
    Age = request.form.get("age")
    Sex = request.form.get("sex")
    SmokingStatus = request.form.get("smokingstatus")
    TargetWeek = request.form.get("pweeks")
    
    #File Validation
    file = request.files['file']
    if file.filename == '':
        error_msg1='No image selected for uploading'
        return render_template('index.html',error_msg1=error_msg1)
    filename = secure_filename(file.filename)
    if filename[-3:]!='dcm':
        error_msg2='Incorrect file type,Expected .dcm file'
        return render_template('index.html',error_msg2=error_msg2)
    
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    #Change String tyoe to int
    BaselineWeek=abs(int(BaselineWeek))
    BaselineFVC=int(BaselineFVC)
    Age=int(Age)
    TargetWeek=int(TargetWeek)
    session['gender'] = Sex
    session['age']=Age
    #Create test data
    test_data=pd.DataFrame([{'PatientID':PatientID,'BaselineWeek':BaselineWeek,'BaselineFVC':BaselineFVC,'Percent':Percent,'Age':Age,
                           'Sex':Sex,'SmokingStatus':SmokingStatus,'TargetWeek':TargetWeek}])
    
    #Preprocess test data
    test_data.drop(columns = ['Percent'], inplace = True)
    test_data["Sex"]=test_data["Sex"].astype("category").cat.codes
    test_data["SmokingStatus"]=test_data["SmokingStatus"].astype("category").cat.codes
    dcms=[]
    ds = dicom.dcmread(file_path)
    dcms.append(ds.pixel_array)
    im = Image.fromarray(dcms[0])
    im = im.resize((128,128),resample=Image.NEAREST) 
    dcms[0] = np.array(im).reshape((128,128,1))
    dcms=np.array(dcms)
    res=dcms
    
    #Send filename to progression route
    session['my_var'] = filename
    session['pat_id']= PatientID
    #Prediction 
    features_test=test_data[[ 'BaselineWeek', 'BaselineFVC', 'Age', 'Sex','SmokingStatus','TargetWeek']]   
    features_test.to_csv('temp.csv',index=False)
    fvc_reg=reg_model.predict(features_test)
    fvc_cnn=cnn_model.predict([dcms, features_test], batch_size=100, verbose=1)
    fvc=0.35*fvc_reg[0]+0.65*fvc_cnn[0][0]
    fvc=np.round(fvc,2)
    
    return render_template('index.html',res = fvc)

@app.route("/progression")
def progression():
    #Fetch session variables from previous route
    filename = session.get('my_var', None)
    pat_id = session.get('pat_id', None)
    age_var = session.get('age', None)
    sex_var = session.get('gender',None)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    #Preprocess test data
    dcms=[]
    ds = dicom.dcmread(file_path)
    dcms.append(ds.pixel_array)
    im = Image.fromarray(dcms[0])
    im = im.resize((128,128),resample=Image.NEAREST) 
    dcms[0] = np.array(im).reshape((128,128,1))
    dcms=np.array(dcms)
    res=dcms
    while res.shape[0]!=52:
        res=np.append(res, np.array(dcms), axis=0)
    print(res.shape)
    test_new=pd.read_csv('temp.csv')
    age=test_new["Age"].values
    df=test_new
    df["TargetWeek"]=1
    df= df.append([df]*51,ignore_index=True)
    for i in range(1,52):
        df['TargetWeek'].iloc[i]=i+1
        
    #Predict for 52 weeks
    fvc_reg=reg_model.predict(df)
    fvc_cnn=cnn_model.predict([res, df], batch_size=100, verbose=1)
    new_out=np.zeros(52)
    
    #Ensemble
    for i in range(52):
        new_out[i]=0.35*fvc_reg[i]+0.65*fvc_cnn[i][0]
    min_fvc=int(min(new_out))
    avg_fvc=int(np.mean(new_out))
    new_out=np.round(new_out,2)
    new_res=list(new_out)

    df["PredictedFVC"]=list(new_out)
    df.drop(columns = ['Sex','SmokingStatus'], inplace = True)
    df.to_csv('final_result.csv',index=False)
    area=line_chart(new_res)
    donut,risk_type=donut_chart(age_var,sex_var,new_res)

    #os.remove('temp.csv')
    #os.remove(file_path)
    return render_template('index2.html',output=new_out,pat_id=pat_id,age=age[0],min_fvc=min_fvc,avg_fvc=avg_fvc,donut=donut,area=area,risk_type=risk_type)

@app.route("/table")
def table():

    df=pd.read_csv('final_result.csv')
    PatientID=session.get('pat_id', None)
    #os.remove('final_result.csv')
    return render_template('tables.html',column_names=df.columns.values, row_data=list(df.values.tolist()),
                           zip=zip,PatientID=PatientID)
    
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)  