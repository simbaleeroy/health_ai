import numpy as np
from random import randint
from sklearn.utils import shuffle 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow import keras
import keras as kr
from keras.models import Sequential 
from keras.optimizers import Adam 
from keras.layers import Activation, Dense
from keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import os.path
from keras.models import load_model


# An experimental drug was tested on individuals who range from 13 to 100 in a clinical trial
#The trial had 2100 applicants. half were under 65 years and half were 65 years and older
# 95% of patients under 65 years experienced no side effects
# 95% of patients 65 or older experienced side effects


def main():
    train_labels = [] 
    train_samples = [] 

    for i in range(50):
        # 5% of younger individuals who did experience side effects
        random_younger = randint(13,64)
        train_samples.append(random_younger)
        train_labels.append(1)
        # 5% of older individuals who did not experiance side effects
        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(0)

    for i in range(1000):
        # 95% of younger individuals who did not experience side effects
        random_younger = randint(13,64)
        train_samples.append(random_younger)
        train_labels.append(0)
        # 95% of older individuals who experienced side effects
        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(1)

    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples) 
    train_labels, train_samples = shuffle(train_labels, train_samples) 

    scaler = MinMaxScaler(feature_range=(0,1)) 
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #print(len(physical_devices))
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = Sequential([Dense(units=16, input_shape=(1,), activation='relu'), Dense(units=32, activation = 'relu'), Dense(units=2, activation = 'softmax')])
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=scaled_train_samples, y=train_labels, validation_split = 0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

    test_labels = [] 
    test_samples = [] 

    for i in range(10):
        # 5% of younger individuals who did experience side effects
        random_younger = randint(13,64)
        test_samples.append(random_younger)
        test_labels.append(1)
        # 5% of older individuals who did not experiance side effects
        random_older = randint(65,100)
        test_samples.append(random_older)
        test_labels.append(0)

    for i in range(200):
         # 95% of younger individuals who did not experience side effects
        random_younger = randint(13,64)
        test_samples.append(random_younger)
        test_labels.append(0)
        # 95% of older individuals who experienced side effects
        random_older = randint(65,100)
        test_samples.append(random_older) 
        test_labels.append(1)  

    test_labels = np.array(test_labels) 
    test_samples = np.array(test_samples) 
    test_samples, test_labels = shuffle(test_samples, test_labels) 
    scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1)) 

    predictions = model.predict(x = scaled_test_samples, batch_size= 10, verbose = 0)
    rounded_predictions = np.argmax(predictions, axis = -1)

    cm = confusion_matrix(y_true = test_labels, y_pred = rounded_predictions) 
    def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap = cmap)
        plt.title(title) 
        plt.colorbar() 
        tick_marks = np.arange(len(classes)) 
        plt.xticks(tick_marks, classes, rotation = 45) 
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float')/ cm.sum(axis = 1)[:, np.newaxis] 
            print("Normalized confusion matrix") 
        else:
            print('Confusion matrix without normalization')
        #plt.show()
        print(cm)
        #plt.show() 

        thresh = cm.max()/2
        for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i,cm[i,j], horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm_plot_labels = ['no_side_effects', 'had_side_effects']
    plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')
    plt.show()

    if os.path.isfile('models/medical_trial_model.h5') is False:
        model.save('models/medical_trial_model.h5')

    new_model = load_model('models/medical_trial_model.h5')
    new_model.get_weights()
    new_model.optimizer
     

if __name__ == '__main__':
    main()
