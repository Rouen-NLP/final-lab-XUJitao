import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import itertools
from time import time
from sklearn import preprocessing
from keras.utils import np_utils
from keras.layers import Dense, Embedding, Input, Reshape
from keras.layers import Dropout, MaxPooling1D, Conv1D, Conv2D, Flatten, LSTM, Softmax
from keras.models import Model
from keras.preprocessing import text, sequence
from nn_utils import TrainingHistory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import tensorflow as tf
from datetime import timedelta
from tensorflow.python.layers import base
import tensorflow.contrib.slim as slim
import keras.backend as K
import keras

base_data_dir = "data/Tobacco3482-OCR/"
list_dir = os.listdir(base_data_dir)
print(str(list_dir))
nbs = []
x = []
y = []
for repo in list_dir:
    prefix = base_data_dir + repo + '/'
    files = os.listdir(prefix)
    for file in files:
        with open(prefix + file, 'r') as f:
            txt = f.read()
        txt.replace(',', "").replace('.', "").replace('"', "")
        txt.replace("'", "").replace('?', '').replace('!', "")
        txt.replace(':', '').replace(';', '')
        x.append(txt.replace('\n', ' '))
        y.append(repo)
    nbs.append(len(files))
print(str(nbs))
x = np.array(x)
y = np.array(y)

x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(x, y, test_size=0.2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_fit_predict(model, x_train, x_test, y_train, history):
    
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1,
              validation_split=VALIDATION_SPLIT)

    return model.predict(x_test)


def plot_conf_mat(y_test, y_predicted):
    conf_mat = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plot_confusion_matrix(conf_mat, CLASSES_LIST, title='Confusion matrix, without normalization')
    plt.subplot(122)
    plot_confusion_matrix(conf_mat, CLASSES_LIST, normalize=True, title='Normalized confusion matrix')
    plt.show()

    
TF_IDF_FEATURES = 2000
# Create document vectors
vectorizer = CountVectorizer(max_features=TF_IDF_FEATURES)
vectorizer.fit(x_train_orig)
x_train_counts = vectorizer.transform(x_train_orig)
x_test_counts = vectorizer.transform(x_test_orig)

# With TF-IDF representation
tf_transformer = TfidfTransformer()
tfidf = tf_transformer.fit(x_train_counts)
x_train_tf = tfidf.transform(x_train_counts)
x_test_tf = tfidf.transform(x_test_counts)

def get_model():

    inp = Input(shape=(TF_IDF_FEATURES,))
    model = Dense(1024, activation='relu')(inp)
    model = Dropout(0.8)(model)
    model = Dense(NUM_CLASS, activation="softmax")(model)
    model = Model(inputs=inp, outputs=model)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def class_str_2_ind_label(y_train, y_test):
    le = preprocessing.LabelEncoder()
    CLASSES_LIST = np.unique(y_train)
    n_out = len(CLASSES_LIST)
    le.fit(CLASSES_LIST)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    train_y_cat = np_utils.to_categorical(y_train, n_out)
    test_y_cat = np_utils.to_categorical(y_test, n_out)
    
    return y_train, y_test, train_y_cat, test_y_cat

BATCH_SIZE = 128
EPOCHS = 30
VALIDATION_SPLIT = 0.2
CLASSES_LIST = np.unique(y_train_orig)
NUM_CLASS = len(CLASSES_LIST)

y_train, y_test, train_y_cat, test_y_cat = class_str_2_ind_label(y_train_orig, y_test_orig)
model = get_model()
history = TrainingHistory(x_test_tf, y_test, CLASSES_LIST)
y_predicted = train_fit_predict(model, x_train_tf, x_test_tf, train_y_cat, history).argmax(1)
plot_conf_mat(y_test, y_predicted)

print("\n\n")
print("Test F1-score:", f1_score(y_test, y_predicted, average="micro"))
print("\n\n")
