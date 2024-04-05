import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.typeDict = dict()
from tensorflow.keras.utils import to_categorical
from scipy import fftpack
from scipy.signal import butter,filtfilt


def folder_files(path: string) -> list:
    """
    Returns a list with the names of all of the files in a folder: 
    
    Args: 
        path (string): Path to the folder containing the files.   
    
    Returns:
        filenames (list): List of the names of each file present in the folder.
    """ 
    # Load the data:
    
    filenames = [ path+'/'+name for name in os.listdir(path)]
    
    return filenames


def train_test_split(filenames):
    """ Randomly csv files in folder in training and test: """
    n_of_patients = len(filenames)
    TRAIN_TEST_SPLIT = 0.8
    patients_training = int(n_of_patients*TRAIN_TEST_SPLIT)
    filenames_train = filenames[:patients_training]
    filenames_test = filenames[patients_training:]
    
    return filenames_train, filenames_test


def load_data(filenames):
    """ """
    df = pd.concat( pd.read_csv(file) for file in filenames )
    
    return df
    

def get_labels(df, label_name):
    """ Return the dataframe without the column mentioned and the column mentioned as 
        numpy array, i.e. the labels """
    labels = df[label_name].to_numpy()
    df.drop(label_name, axis=1, inplace=True)
    
    return labels, df 


def one_hot_labels(labels):
    """ Labels one-hot encoding: """
    labels_encoded = to_categorical(labels)
    n_classes = labels_encoded.shape[1]

    return labels_encoded, n_classes

def drop_first_col(df, col_name):
    """ Drop first column: """
    df.drop(col_name, axis=1, inplace=True)
    
    return 

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y

def frequency_filtering(df, fs, cutoff, order):
    """ Takes a dataframe in input and outputs the dataframe with each column filtered with a low 
        pass filter """
    sec = int( df.shape[0] / fs )
    time = np.linspace(0, sec, df.shape[0], endpoint=True)
    for feat in df.columns:
        signal = df[feat].to_numpy()
        sig_noise_fft = fftpack.fft(signal)
        sig_noise_amp = 2 / time.size * np.abs(sig_noise_fft)
        sig_noise_freq = np.abs(fftpack.fftfreq(time.size, 1/fs))
        # Filter the data:
        y = butter_lowpass_filter(signal, cutoff, fs, order)
        df[feat] = y

    return df 


def split_and_reshape(m, x, overlap, labels_encoded):
    """ Split the data in windows and reshape the 2D numpy array (x) in a 3D array, with 
        [windows,samples_per_windows,features] as dimensions:
    """

    n_rows = x.shape[0]
    total_windows = int( n_rows / (m - overlap) ) - 1
    
    lx = []
    ly = []
    for w in range(total_windows):
        if w == 0:
            lx.append(x[w*m:w*m+m,:])
            ly.append(labels_encoded[w*m:w*m+m,:])
        else:
            lx.append(x[w*m-w*overlap:w*m+m-w*overlap,:])
            ly.append(labels_encoded[w*m-w*overlap:w*m+m-w*overlap,:])
            
    X = np.array(lx)
    Y = np.array(ly)
    
    return X, Y


def create_window_labels(labels_encoded):
    # We need one label per window:
    y = np.zeros((labels_encoded.shape[0],labels_encoded.shape[2]))
    for i in range(labels_encoded.shape[0]):
        decoded_labels = np.argmax(labels_encoded[i,:,:], axis=1)
        most_common_label = np.argmax(np.bincount(decoded_labels)) # We use the most common label to label the window.
        mcl_encoded = to_categorical(most_common_label,num_classes=12)
        y[i,:] = mcl_encoded 

    return y


def make_dataset(filenames: list):
    """
    Returns the timeseries and their labels organized in train and test.

    Args:
        filenames (list): List of the names of each file present in the folder we are working with.
        
    Returns:
        X_train : _description_
    """
    
    
    
    
    
    return X_train, y_train, X_test, y_test