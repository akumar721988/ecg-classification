import pandas as pd
import numpy as np
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import os

current_dir = os.getcwd()

#h5py=2.10.0
df_train = pd.read_csv(current_dir+"/data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)

df_test = pd.read_csv(current_dir+"/data/mitbih_test.csv", header=None)

##### distribution of labels ##########
print(df_train[187].value_counts())
#0.0    72471
#4.0     6431
#2.0     5788
#1.0     2223
#3.0      641

#Label details
#0	Normal, Left/Right bundle branch block, Atrial escape, Nodal escape
#1	Atrial premature, Aberrant atrial premature, Nodal premature, Supra-ventricular premature
#2	Premature ventricular contraction, Ventricular escape
#3	Fusion of ventricular and norma
#4	Paced, Fusion of paced and normal, Unclassifiable

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

####### class-weight balanced #########
class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)

print(class_weights)

def get_model():

    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    #model.summary()
    return model

model = get_model()
file_path = current_dir+"/baseline_cnn_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

def train_model():

    model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1,class_weight=class_weights)
    
    model.load_weights(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    print("Test f1 score : %s "% f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s "% acc)

def test_model():

    model.load_weights(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")

    print("Test f1 score : %s "% f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s "% acc)

    print(classification_report(Y_test,pred_test,digits=4))

if __name__=="__main__":
    #train_model()
    test_model()