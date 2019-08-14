from keras.callbacks import History

import json
from tensorflow.python.framework.ops import Tensor
from keras.utils import to_categorical

from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, Layer, Lambda

import warnings

from typing import List, Dict, Any
import os
import random
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input


import numpy as np
import logging
from typing import Tuple
from keras.preprocessing.image import load_img
import boto3


def get_mappings(bucket_name: str, mapping_path: str) -> Dict[str, str]:
    s3 = boto3.resource('s3')
    faces_bucket = s3.Bucket(bucket_name)  # instantiate the bucket object

    obj = s3.Object(bucket_name, mapping_path)  # fetch the mapping dictionary

    json_string: str = obj.get()['Body'].read().decode('utf-8')
    mappings_dict: Dict[str, str] = json.loads(
        json_string)  # this mappings_dict contains filename -> gender class mapping
    print(list(mappings_dict.items())[:3])  # print the first three entries of the mappings dictionary
    return mappings_dict


def expand_tensor_shape(X_train: np.ndarray) -> np.ndarray:
    new_shape: Tuple = X_train.shape + (1,)
    new_tensor = X_train.reshape(new_shape)
    print(f"Expanding shape from {X_train.shape} to {new_tensor.shape}")
    return new_tensor


def build_vanilla_cnn(filters_layer1: int, filters_layer2: int, kernel_size: int,
                      input_dims: Tuple[int, int, int]) -> Model:
    inputs: Tensor = Input(shape=input_dims)
    x: Tensor = Conv2D(filters=filters_layer1, kernel_size=32, activation='relu')(inputs)
    x: Tensor = MaxPooling2D(pool_size=(4, 4))(x)
    x: Tensor = Conv2D(filters=filters_layer2, kernel_size=kernel_size, activation='relu')(x)
    x: Tensor = MaxPooling2D(pool_size=(16, 16))(x)
    x: Tensor = Flatten()(x)
    x: Tensor = Dense(32, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)

    # compile model using accuracy to measure model performance
    model: Model = Model(inputs=inputs, outputs=predictions)
    print(model.summary())
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    return model


if __name__ == "__main__":
    cwd = os.getcwd()
    warnings.filterwarnings('ignore')
    S3_BUCKET_NAME = "fei-faces-sao-paulo"
    mapping = 'classification/gender.json'

    print(f"Current working directory is {cwd}")
    s3 = boto3.client('s3')
    warnings.filterwarnings('ignore')
    IMAGE_LIMIT = 3000
    LOCAL_IMAGES_FOLDER = "faces"

    mappings_dict: Dict[str, str] = get_mappings(S3_BUCKET_NAME, mapping)

    target: List[str] = []  # this list will contain our actual tensors (as N-dimensional numpy arrays)
    images: List[np.ndarray] = []  # this list will contain our classes (male or female)

    for filename, gender in mappings_dict.items():

        if "-14" in filename or "-10" in filename:  # these images are blurry or obscured
            continue

        local_filename: str = os.path.join(cwd, LOCAL_IMAGES_FOLDER, filename)
        try:
            if not os.path.isfile(local_filename):  # if file does not exist locally
                print(f"Downloading {filename}, saving as {local_filename}")
                s3.download_file(S3_BUCKET_NAME, filename, local_filename)
            else:
                logging.debug(f"Found a local copy of {local_filename}")

            # use the Keras image API to load in an image
            img = load_img(local_filename)
            img = img.convert('L')  # convert to gray scale
            # report details about the image
            images.append(np.array(img))
            target.append(gender)
            if len(images) == IMAGE_LIMIT:
                print("Breaking after reaching image limit.")
                break
        except Exception:
            print(f"Error downloading {filename}")

    binary_target = np.array(list(map(lambda gender: 0 if gender == 'male' else 1, target)))
    encoded_target = to_categorical(binary_target)

    print(f"One-hot encoding target vector {binary_target.shape} -> {encoded_target.shape}")
    NUM_CLASSSES = encoded_target.shape[1]
    print(f"There are {NUM_CLASSSES} classes to predict.")

    indices = np.linspace(0, len(binary_target) - 1, len(binary_target))
    validation_indices = np.random.choice(indices, size=int(len(binary_target) * 0.15), replace=False).astype(int)
    training_indices = set(indices).difference(set(validation_indices))
    training_indices = np.array(list(training_indices)).astype(int)

    combined: List[Any] = list(zip(images, binary_target))
    random.shuffle(combined)

    images[:], binary_target[:] = zip(*combined)

    images: np.ndarray = np.array(images)
    X_train = images[training_indices]
    y_train = binary_target[training_indices]
    X_test = images[validation_indices]
    y_test = binary_target[validation_indices]
    X_train_expanded: np.ndarray = expand_tensor_shape(X_train)
    X_test_expanded: np.ndarray = expand_tensor_shape(X_test)
    images_expanded = expand_tensor_shape(images)

    print(f"The shape of X_train_expanded is {X_train_expanded.shape}")
    print(f"The shape of X_test_expanded is {X_test_expanded.shape}")
    print(f"The shape of X_train is {X_train.shape}")
    print(f"The shape of y_train is {y_train.shape}")
    print(f"The shape of X_test is {X_test.shape}")
    print(f"The shape of y_test is {y_test.shape} - some example targets:\n {y_test[:5]}")

    input_dims = (480, 640, 1)
    model: Model = build_vanilla_cnn(32, 16, 4, input_dims)
    RUN_VANILLA = True  # set this to True to actually run the model
    if RUN_VANILLA:
        history: History = model.fit(X_train_expanded, y_train, epochs=10, batch_size=16)