# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
import tensorflow as tf #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization# type: ignore
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import metrics

# %%
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])
# dataset, dataset_info = tfds.load('malaria', with_info=True)

# %% [markdown]
# <h1>Data Preparation</h1>

# %%
for data in dataset[0].take(3):
    print(data)

# %%
def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

    val_test_dataset = dataset.skip(int(TEST_RATIO*DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))

    test_dataset = val_test_dataset.skip(int(VAL_RATIO*DATASET_SIZE))
    return train_dataset, val_dataset, test_dataset

# %%
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# dataset = tf.data.Dataset.range(10)
train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(list(train_dataset.take(1).as_numpy_iterator()))
print(list(val_dataset.take(1).as_numpy_iterator()))
print(list(test_dataset.take(1).as_numpy_iterator()))

# %% [markdown]
# <h1>Dataset Visualization</h1>

# %%
for i, (image,label) in enumerate (train_dataset.take(16)):
    ax = plt.subplot(4,4, i+1)
    plt.imshow(image)
    plt.title(dataset_info.features['label'].int2str(label))
    

# %% [markdown]
# <h1>Data Preprocessing</h1>

# %%
IM_SIZE=224
def resizing_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

# %%
train_dataset = train_dataset.map(resizing_rescale)
val_dataset = val_dataset.map(resizing_rescale)
test_dataset = test_dataset.map(resizing_rescale)

# %%
# # Resize and batch the datasets
# train_dataset = train_dataset.map(resize_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
# # val_ds = val_ds.map(resize_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# %%
for image, label in train_dataset.take(2):
    print(image, label)

# %%
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# <h1>Model Creation<h1>

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2),
    Flatten(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid'),
])
model.summary()

# %%
# y_true = [0]
# y_pred = [0.4]

# # Convert lists to tensors
# y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
# y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)

# bce = tf.keras.losses.BinaryCrossentropy()
# bce(y_true_tensor, y_pred_tensor)

# %%
model.compile(optimizer=Adam(learning_rate=0.01),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# %%
history=model.fit(train_dataset, validation_data=val_dataset, epochs=20, verbose=1)

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train loss', 'val loss'])
plt.show()

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train accuracy', 'val accuracy'])
plt.show()

# %% [markdown]
# <h1>Model Evaluation and testing<h1>

# %%
'''
# Assuming test_dataset is your dataset
# Define a function to reshape each element of the dataset
def reshape_function(image, label):
    # Reshape image tensor to (224, 224, 3)
    image = tf.reshape(image, [-1, 224, 224, 3])  # Use -1 to preserve batch size dimension
    return image, label

# Apply the reshape function to your dataset using map
test_dataset = test_dataset.map(reshape_function)
'''


# %%
import tensorflow as tf

# Example reshape function to reshape the dataset
def reshape_dataset(image, label):
    image = tf.reshape(image, [-1, 224, 224, 3])  # Reshape image to (None, 224, 224, 3)
    label = tf.reshape(label, [-1])  # Reshape label to (None,)
    return image, label

# Assuming test_dataset is your dataset with the specified element_spec
# Apply reshape function using map
test_dataset = test_dataset.map(reshape_dataset)

# Now, evaluate the model on the reshaped dataset
model.evaluate(test_dataset)


# %%
def par_or_not(x):
    if(x>0.5): 
        return 'U'
    else:
        return 'P'

# %%
par_or_not(model.predict(test_dataset.take(1))[0][0])

# %%
for i, (image, label) in enumerate(test_dataset.take(9)):
    
    ax = plt.subplot(3,3,i+1)
    plt.imshow(image[0])
    plt.title(str(par_or_not(label.numpy()[0])+" : "+str(par_or_not(model.predict(image)[0][0]))))
    
    plt.axis('off')

# %%



