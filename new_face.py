import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

#image_path=r'D:\Amber\Multimodel Root\CancelableTemplate_girls\output_dataset\face\user_1\4_MariaCallas_41_f.jpg'
root=r'D:\Amber\Multimodel Root\CancelableTemplate_girls\output_dataset\face'
# Step 1: Feature Extraction
base_model = tf.keras.applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(512, activation="relu")  # Reduce to 512D
])

def preprocess_image(image_path):
    #img = load_img(image_path, target_size=(224, 224))
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array


def get_face_feature_Old1(face_path):
    img_tensor = preprocess_image(face_path)
    features = feature_extractor.predict(img_tensor)
    features = tf.nn.l2_normalize(features, axis=1)
    f=new_logic(features,8)
    # Step 2: Binarize to Bit String
    thresh = np.mean(f)
    print('threshold :',thresh)
    #numpy.where(mlphash > thresh, 1, 0)
    bit_string = np.where(f > thresh, 1,0)
    #bit_string = tf.cast(features >thresh, tf.int32)
    #bit_string = tf.squeeze(bit_string)
    #print(bit_string)
    return bit_string

def get_face_feature(face_path):
    img_tensor = preprocess_image(face_path)
    features = feature_extractor.predict(img_tensor)
    features = tf.nn.l2_normalize(features, axis=1)

    # Step 2: Binarize to Bit String
    #thresh = np.mean(features)
    #print('threshold :',thresh)
    #numpy.where(mlphash > thresh, 1, 0)
    #bit_string = np.where(features > thresh, 1,0)
    #bit_string = tf.cast(features >thresh, tf.int32)
    #bit_string = tf.squeeze(bit_string)
    #print(bit_string)
    return features

def nonlinear_funcftion(in_vector):
    return np.where(in_vector<0,0,in_vector)

def new_logic(features,_len):
    rand_mat = np.random.rand(len(features), _len)
    orth_mat, _ = np.linalg.qr(rand_mat, mode='reduced')
    mlphash=features*orth_mat
    mlphash = nonlinear_funcftion(mlphash)
    return mlphash



def prepare_face_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    label_dict = {}
    label = 0
    for dir_name in dirs:
        if not dir_name.startswith("."):
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            subject_images_names = [f for f in os.listdir(subject_dir_path) if f.endswith('.jpg')]
            for image_name in subject_images_names:
                if not image_name.startswith("."):
                    image_path = os.path.join(subject_dir_path, image_name)
                    face=get_face_feature(image_path)
                    faces.append(face)
                    labels.append(label)
            label_dict[label] = dir_name
            label += 1

    return np.array(faces), np.array(labels), label_dict


def process_face_dataset(data_folder_path,output_file="face_features.npz"):
    faces, labels, label_dict = prepare_face_data(data_folder_path)
    print(f"Prepared {len(faces)} face images for feature extraction.")
    user_labels = [f"user_{label+1}" for label in labels]
    np.savez(output_file, features=faces, labels=user_labels)
    print(f"Saved face features and labels to {output_file}")
    return faces, user_labels


process_face_dataset(root)