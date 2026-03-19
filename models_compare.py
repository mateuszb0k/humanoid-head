import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import json
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

batch_size = 128
Epochs = 15
Input_shape = (224, 224, 3)
num_emotions = 7 # angry, disgust, fear, happy, sad, suprise + neutral
LR = 0.001 # learning rate


data = Path("/kaggle/input/datasets/baranittto/datasetv2/FINAL_bez_dzieci/FINAL_bez_dzieci")
os.makedirs("plots", exist_ok=True)
trainVal_data = tf.keras.utils.image_dataset_from_directory(data, labels='inferred', label_mode='int',
                                                            color_mode='rgb', batch_size=batch_size,
                                                            image_size=(224, 224), validation_split=0.1,
                                                            subset='training', seed=42)

test_data = tf.keras.utils.image_dataset_from_directory(data, labels='inferred', label_mode='int',
                                                        color_mode='rgb', batch_size=batch_size,
                                                        image_size=(224,224), validation_split=0.1, subset='validation',
                                                        seed=42)

class_names = trainVal_data.class_names
total_batches = tf.data.experimental.cardinality(trainVal_data).numpy()
val_batches = int(total_batches * (20 / 90))
valid_data = trainVal_data.take(val_batches)
train_data = trainVal_data.skip(val_batches)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
valid_data = valid_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)


weights_file = '/kaggle/working/class_weights.json'
if os.path.exists(weights_file):
    with open(weights_file, 'r') as f:
        class_weight_dict = {int(k):float(v) for k,v in json.load(f).items()}
else:
    train_labels = np.concatenate([y.numpy() for x, y in train_data])
    class_weight = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weight))
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)
    with open(weights_file, 'w') as f:
        json.dump(class_weight_dict, f,indent=4)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

models = {
    "InceptionV3": lambda: tf.keras.applications.InceptionV3(input_shape=Input_shape, include_top=False,weights='imagenet'),
    "ResNet50":lambda:tf.keras.applications.ResNet50(input_shape=Input_shape, include_top=False,weights='imagenet'),
    "ResNet101":lambda : tf.keras.applications.ResNet101(input_shape=Input_shape, include_top=False,weights='imagenet'),
    "VGG16":lambda : tf.keras.applications.VGG16(input_shape=Input_shape, include_top=False,weights='imagenet'),
    "VGG19":lambda :tf.keras.applications.VGG19(input_shape=Input_shape, include_top=False,weights='imagenet')
}

for model_name,model_build in models.items():
    print(f"\nBuilding model: {model_name}\n")
    base_model = model_build()
    base_model.trainable = False
    inputs = tf.keras.Input(shape=Input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1. / 255)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_emotions, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(train_data,
                        epochs=Epochs,
                        validation_data=valid_data,
                        class_weight=class_weight_dict)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.show()


    base_model.trainable = True
    num_layers = len(base_model.layers)
    fine_tune_at = int(num_layers * 0.8)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR/10),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    fine_tune_epochs = 10
    total_epochs = fine_tune_epochs+Epochs

    history_fine = model.fit(train_data,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=valid_data,
                             class_weight=class_weight_dict)
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([Epochs - 1, Epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([Epochs - 1, Epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_fine_tuning.png")
    plt.show()
    print(f"Evaluation: {model_name} on test data\n")
    test_loos, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc:.4f}\n")

