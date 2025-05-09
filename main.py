from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout,Lambda
from utility import *
import numpy as np
from biometric.multimodel import authenticate

face_data = np.load("face_features.npz")
face_features_raw = face_data['features']
face_labels_raw = face_data['labels']

iris_data = np.load("iris_features.npz")
iris_fatures_raw = iris_data['features']
iris_labels_raw = iris_data['labels']

sig_data = np.load("sig_features.npz")
sig_fatures_raw = sig_data['features']
sig_labels_raw = sig_data['labels']



face_features, face_labels = group_features_by_user(face_features_raw, face_labels_raw)
iris_features, iris_labels = group_features_by_user(iris_fatures_raw, iris_labels_raw)
sig_features, sig_labels = group_features_by_user(sig_fatures_raw, sig_labels_raw)

def fuse_features(iris_features, sig_features, face_features):
    try:
        #iris_features = normalize(iris_features, axis=1)
        #fingerprint_features = normalize(fingerprint_features, axis=1)
        #face_features = normalize(face_features, axis=1)
        fused_features = np.concatenate([face_features,iris_features, sig_features], axis=1)
        return fused_features
    except Exception as e:
        print(e)
        return []

def create_cnn_model(input_shape, num_users, template_size):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),


        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),

        Flatten(),
        Dense(8192, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(template_size, activation='sigmoid'),
        Dense(num_users, activation='softmax')

    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def enroll():
    reshaped_features =[]
    # face_featur =face_features.reshape(100, 512)
    # iris_featur = iris_features.reshape(100, 512)
    # sig_featur = sig_features.reshape(100, 512)

    fused_features = fuse_features(iris_features, sig_features, face_features)

    for feature in fused_features:
        total_elements = feature.size
        padding = target_size - total_elements
        features_padded = np.pad(feature, (0, padding), mode='constant', constant_values=0)
        reshaped_features.append(features_padded.reshape(tz, tz, 1))

    #reshaped_features=face_featur.reshape(tz, tz, 1)
    reshaped_features = np.array(reshaped_features)
    #labels=face_labels
    labels = np.array([np.eye(number_of_user)[i] for i in range(len(reshaped_features))])
    cnn_model = create_cnn_model((tz, tz, 1), num_users=number_of_user, template_size=template_size)
    #lbl=getlevals(labels)
    cnn_model.fit(reshaped_features, labels, epochs=100, batch_size=64, verbose=1)
    cnn_model.save("cnn_model.h5")
    enrolled_templates=[]
    enrolled_labels=[]
    for i, user_label in enumerate(labels):
        cancelable_template = cnn_model.predict(reshaped_features[i])
        #user_number=int(user_label.split('_')[1])

        _temp = cancelable_template[0]
        #enrolled_templates.append(userKey.tolist()[user_number])x`
        enrolled_templates.append(_temp)
        enrolled_labels.append(user_label)

    np.savez("enrolled_templates.npz", templates=np.array(enrolled_templates), labels=np.array(enrolled_labels))
    print("Enrollment complete.")

enroll()
authenticate()