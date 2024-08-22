from keras.models import Model
from keras import *
import keras
from keras.layers import *
# from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
# import torch
from pyts.datasets import load_basic_motions
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize,MinMaxScaler,Binarizer
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, Conv2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import *

def evaluate_model(model, X, y):
    pred_prob = model.predict(X)[:,1] #predicted classes
    fpr, tpr, _ = roc_curve(y, pred_prob) # roc_curve
    auc_value = auc(fpr,tpr) # auc_value
    # precision_rf, recall_rf=precision_recall_curve(y, pred)
    auc_rf =average_precision_score(y, pred_prob)
    #rounding the values
    y_pred=np.argmax(model.predict(X), axis=1)
    MCC=matthews_corrcoef(y, y_pred)
    accuracy = accuracy_score(y,y_pred) # calculate accuracy
    report = classification_report(y, y_pred, labels=[0,1], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.reset_index()
    model_eval  = report_df[report_df['index'].str.contains('1')][['precision','recall','f1-score']]
    model_eval['accuracy']  = accuracy
    # list(report_df[report_df['index'].str.contains('accuracy')]['support'])
    model_eval['ROC']  = auc_value
    model_eval['PR']  = auc_rf
    model_eval['MCC']  = MCC
    # model_eval['time'] = time
    cf_matrix = confusion_matrix(y, y_pred)
    
    return model_eval, cf_matrix


def model_eval_data(model, X_train, y_train, 
                         X_test, y_test, 
                         model_eval_train, 
                         model_eval_test,
                         Name=None):
    # start = datetime.datetime.now()
    temp_eval_train, cf_matrix_train = evaluate_model(model, X_train, y_train)
    temp_eval_test, cf_matrix_test = evaluate_model(model, X_test, y_test)
    temp_eval_train.index = [Name]
    temp_eval_test.index = [Name]
    # end = datetime.datetime.now()
    # time=(end -start)*1000
    try:
        model_eval_train=pd.concat([model_eval_train,temp_eval_train],axis=0)
        model_eval_test=pd.concat([model_eval_test,temp_eval_test],axis=0)
    except:
        model_eval_train = temp_eval_train
        model_eval_test = temp_eval_test
    return model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test




