import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import random
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.metrics import categorical_crossentropy



train_label = []
train_sample = []

#we have to create data about a drug which divides the order and young group of people into
# two part. 95% older group who have side effects and 5% which does not have side effects
# 95% of younger people which dont have side effects and 5% who have side effects

for i in range(1000):
    #95% older people who have side effects
    random_older = random.randint(65,100)
    train_sample.append(random_older)
    train_label.append(1)  #people with side effects are shown by 1

    #95% younger people who didnt have side effects
    random_younder = random.randint(13,64)
    train_sample.append(random_younder)
    train_label.append(0)
for i in range(50):
    # 5% older people who didnt had side effects
    
    random_older = random.randint(65,100)
    train_sample.append(random_older)
    train_label.append(0)

    # 5% younger people who had side effects
    random_younder = random.randint(13,64)
    train_sample.append(random_younder)
    train_label.append(1)


train_label = np.array(train_label)
train_sample = np.array(train_sample)
train_sample, train_label = shuffle(train_sample , train_label)

# convert the samples array between 0 and 1

scalar = MinMaxScaler(feature_range=(0,1))
scalar_train_sample = scalar.fit_transform(train_sample.reshape(-1,1))

train_sample, test_sample, train_label, test_label = train_test_split(scalar_train_sample, train_label, test_size=0.33)


# create the model

model = Sequential([
 Dense(units=16, input_shape=(1,), activation="relu"),
 Dense(units=32, activation="relu"),
 Dense(units=2, activation="softmax")
]) 

#compile the model

model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#fit the model
model.fit(train_sample, train_label, batch_size=8, epochs=30, shuffle=True, verbose=2)


prediction = model.predict(test_sample)

rounded_prediction = np.argmax(prediction, axis = -1)
print(rounded_prediction)
# CONFUSION MATRIX
cm = confusion_matrix(y_true=test_label, y_pred=rounded_prediction)


# this confusion matrix function is copied from sklearn website to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



cm_plot_label = ["no side effects", "had side effects"]
plot_confusion_matrix(cm=cm, classes= cm_plot_label, title="confusion matrix")


plt.show()