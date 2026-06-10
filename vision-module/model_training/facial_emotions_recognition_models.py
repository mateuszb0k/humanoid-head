import tensorflow as tf
import os
import numpy as np
import datetime

from scipy.conftest import num_parallel_threads
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from tensorboard.plugins.hparams import api as hp
from pathlib import Path
from models.tfboard import get_confusion_matrix, plot_confusion_matrix
import json

# disable logging output from TensorFLow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# define our consts for training
batch_size = 32
Epochs = 20
Input_shape = (96, 96, 1) # our imgs are 96x96 in grayscale
num_emotions = 7 # angry, disgust, fear, happy, sad, suprise + neutral
LR = 0.001 # learining rate

data = Path("./dataset_kaggle")


gpus = tf.config.experimental.list_physical_devices('GPU') # returns gpu deveices on our local machine
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # prevent TensorFlow from immediatly allocating all available VRAM on detected GPUs

# creating a Train + Val dataset from image files in a directory
trainVal_data = tf.keras.utils.image_dataset_from_directory(data, labels='inferred', label_mode='int',
                                                            color_mode='grayscale', batch_size=batch_size,
                                                            image_size=(96, 96), validation_split=0.1,
                                                            subset='training', seed=42)

# creating a Test dataset from image files in a directory
test_data = tf.keras.utils.image_dataset_from_directory(data, labels='inferred', label_mode='int',
                                                        color_mode='grayscale', batch_size=batch_size,
                                                        image_size=(96, 96), validation_split=0.1, subset='validation',
                                                        seed=42)

class_names = trainVal_data.class_names # draw out our class names

# in this section we calculate total batches to separate validation and train dataset
total_batches = tf.data.experimental.cardinality(trainVal_data).numpy()
val_batches = int(total_batches * (20 / 90))
valid_data = trainVal_data.take(val_batches)
train_data = trainVal_data.skip(val_batches)

# Normalize pixels values from [0, 255] to [0, 1] for faster convergance
def normalize(x, y):
    """
    Scales the pixel values of the image so they are between 0 and 1.
    This helps the neural network learn faster.
    """
    return x / 255.0, y

# Apply normalization and prefetch (pipeline fetch the next batch while GPU trains on the current one
# batches to overlap I/O with training


augumentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


def aug_train_data(x,y):
    """
    Normalizes the image and applies random visual changes like flipping or rotating.
    This creates more variety in the training data to prevent overfitting.
    """
    x,y = normalize(x,y)
    x = augumentation(x,training=True)
    return x,y


train_data = train_data.map(aug_train_data,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
valid_data = valid_data.map(normalize,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(normalize,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# class weight compensate for imbalanced data, some emotions have more samples than other
# weights are saved in a JSON file to avoid recomputing on every run 
weights_file = 'class_weights.json'
if os.path.exists(weights_file):
    # load pre-computed weights from JSON file
    with open(weights_file, 'r') as f:
        class_weight_dict = {int(k):float(v) for k,v in json.load(f).items()} 
else:
    # extracting labels from the training set
    train_labels = np.concatenate([y.numpy() for x, y in train_data])
    
    # compute balanced weights
    class_weight = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weight))

    with open(weights_file, 'w') as f:
        json.dump(class_weight_dict, f,indent=4)

# convert class weights dict to a tensor
class_weight_tensor = tf.constant(list(class_weight_dict.values()), dtype=tf.float32)

# tracks hyperparameters like LR, dropout,model_names
HP_LR = hp.HParam('learning_rate', hp.Discrete([LR]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.5))
HP_MODEL = hp.HParam('model_name', hp.Discrete(['first_model', 'second_model', 'third_model', 'fourth_model']))
METRIC_ACCURACY = 'val_accuracy'

# save the HP configs in file so TensorBoard can display it
with tf.summary.create_file_writer("newkaggle_logs/hparam_tuning").as_default():
    hp.hparams_config(hparams=[HP_LR, HP_DROPOUT, HP_MODEL],
                      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Val Accuracy')])


def Res(conv, filters, downsample=False):
    """
    Residual block, If downsample=True, stride = 2 halves spatial dims and 1x1 cpnv aligns the skip connection.
    If filter size changes, the skip connection is also projected to match.
    """
    input_x = conv
    if downsample:
        stride = 2
    else:
        stride = 1

    conv = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(conv)
    conv = layers.BatchNormalization()(conv)

    # skip projection : project input if spatial size or filter count changed
    if downsample or input_x.shape[-1] != filters:
        input_x = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(input_x)
        input_x = layers.BatchNormalization()(input_x)
    #Merge main path + skip, then activate
    conv = layers.Add()([conv, input_x])
    conv = layers.Activation('relu')(conv)
    return conv


def se_block(conv, ratio=8):
    """
    Squeeze and Exicitation block. Recalibrates channel wise feature respones.
    GlobalAvgPool squeezes spatial info (two dense layers leran per channel weights then sigmoid
    gate scales each channel) Ratio controls bottleneck size.
    """
    filters = conv.shape[-1]
    se = layers.GlobalAvgPool2D()(conv)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([conv, se])


def conv_se_block(conv, filters, downsample=False):
    """
    Conv block with SE attention. Conv -> BN -> ReLU -> Conv-> BN -> SE-> ReLU.
    This function is used by fourth_model as an attention enchaned alternative to plain residual blocks.
    """
    if downsample:
        stride = 2
    else:
        stride = 1

    conv = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = se_block(conv) # apply channel attention before final actiavation
    conv = layers.Activation('relu')(conv)
    return conv


def first_model():
    """
    This model has 4 conv layers with increasing filters from 32 to 256,
    L2 regulaization, dropout after each conv, and MaxPooling for downsampling.
    We decide to use Relu activation function and softmax ( )
    """

    inputs = Input(Input_shape)

    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(inputs)
    conv1 = layers.Dropout(0.1)(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
    conv2 = layers.Dropout(0.1)(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
    conv3 = layers.Dropout(0.1)(conv3)
    conv3 = layers.Activation('relu')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
    conv4 = layers.Dropout(0.1)(conv4)
    conv4 = layers.Activation('relu')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Classifier
    flatten = layers.Flatten()(pool4)
    dense = layers.Dense(128, activation='relu')(flatten)
    drop_1 = layers.Dropout(0.2)(dense)
    outputs = layers.Dense(num_emotions, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=outputs, name="first_model")
    model.summary()
    return model


def second_model():
    """
    Deeper CNN model than first one. 4 blocks of double conv layers. All with fixed 32 filters
    Two dense layers with BatchNormalization and dropout in the heaa. 
    Tests whether depth without filter growth improves over first_model.
    """
    inputs = Input(Input_shape)

    conv1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    #Classifier with BatchNormalization betwwen dense layers
    flatten = layers.Flatten()(pool4)
    dense = layers.Dense(256, activation='relu')(flatten)
    dense = layers.BatchNormalization()(dense)
    drop_1 = layers.Dropout(0.4)(dense)
    dense_2 = layers.Dense(128, activation='relu')(drop_1)
    drop_2 = layers.Dropout(0.3)(dense_2)
    outputs = layers.Dense(num_emotions, activation='softmax')(drop_2)

    model = Model(inputs=inputs, outputs=outputs, name='second_model')
    model.summary()
    return model


def third_model():

    """
    This model is an ResNet-style model. It has initial Conv stem followed by six residual blocks
    with progressive filter growth from 64 to 256. and stride based downsampling.
    GlobalAvgPool replaces Flatten to reduce parameters and overfitting.
    """
    inputs = Input(Input_shape)

    # Stem - initial conv before residual blocks
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    # Residual Block: pairs at each resolution, downsamples at first block of each pair
    res_block1 = Res(conv1, 64, downsample=True)
    res_block2 = Res(res_block1, 64)
    res_block3 = Res(res_block2, 128, downsample=True)
    res_block4 = Res(res_block3, 128)
    res_block5 = Res(res_block4, 256, downsample=True)
    res_block6 = Res(res_block5, 256)

    #GlobalAvgPool collapses spatial dims (fewer params than Flatten)
    pool1 = layers.GlobalAvgPool2D()(res_block6)
    dense = layers.Dense(256, activation='relu')(pool1)
    drop_1 = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(num_emotions, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=outputs, name='third_model')
    model.summary()
    return model


def fourth_model():
    """
    This model is an SE-Net style model. Same structure as third_model but uses conv_se_blocks
    instead of residual blocks. Channel attention (SE) lets model focus on the most 
    informative feature maps at each stage.
    """
    inputs = Input(shape=Input_shape)

    #Stem
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.001))(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    # SE blocks with progressive filter growth
    conv_se_1 = conv_se_block(conv1, 64, downsample=True)
    conv_se_2 = conv_se_block(conv_se_1, 64)
    conv_se_3 = conv_se_block(conv_se_2, 128, downsample=True)
    conv_se_4 = conv_se_block(conv_se_3, 128)
    conv_se_5 = conv_se_block(conv_se_4, 256, downsample=True)

    pool1 = layers.GlobalAvgPool2D()(conv_se_5)
    dense = layers.Dense(256, activation='relu')(pool1)
    drop_1 = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(num_emotions, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=outputs, name='fourth_model')
    model.summary()
    return model

# Section below applies training

# mapping model names to their build functions
models = {
    "first_model": first_model,
    "second_model": second_model,
    "third_model": third_model,
    "fourth_model": fourth_model,
}

# shared loss and metrics. Reset each epoch, reused across all models
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
loss_metric = tf.keras.metrics.Mean()

history = {} # stores final val metrics per model
model_params = {} # stores parameter counts per model

for name, build_func in models.items():
    print(f"\nTrening {name}")

    model = build_func()
    model_params[name] = model.count_params()
    optimizer = Adam(LR)

    @tf.function # compiled to graph for faster execution
    def train_step(x, y, class_weight_tensor):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            # APply per sample class weights to address class imbalance
            sample_weights = tf.gather(class_weight_tensor, y)
            loss = loss_fn(y, y_pred, sample_weight=sample_weights)

        #Backprop and weight update
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc_metric.update_state(y, y_pred)
        loss_metric.update_state(loss)
        return loss, y_pred

    @tf.function
    def val_step(x, y):
        y_pred = model(x, training=False) # False training disables dropout/BN updates
        loss = loss_fn(y, y_pred)
        acc_metric.update_state(y, y_pred)
        loss_metric.update_state(loss)
        return y_pred

    # separate TensorBoard writers for train and val curves
    log_dir = f"newkaggle_logs/{name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(log_dir + '/train')
    val_writer = tf.summary.create_file_writer(log_dir + '/val')

    # Early sopping and LR reduction state
    best_val_loss = float('inf')
    patience_count = 0   # epochs withou improvement (triggers early stop at 7)
    reduce_count = 0    # epochs without improvement (triggers LR reduction at 3)
    v_acc = 0.0


    # Training pass
    for epoch in range(Epochs):
        print(f"\n Epoka: {epoch + 1}/{Epochs}")
        acc_metric.reset_state()
        loss_metric.reset_state()

        #log train metrics and current LR to TensorBoard
        for batch_idx, (x, y) in enumerate(train_data):
            train_step(x, y, class_weight_tensor)

            # if batch_idx % 100 == 0:
            #     print(f"  Batch {batch_idx} - Loss: {loss_metric.result():.4f}, Acc: {acc_metric.result():.4f}")

        t_loss = float(loss_metric.result())
        t_acc = float(acc_metric.result())

        with train_writer.as_default():
            tf.summary.scalar('loss', t_loss, step=epoch)
            tf.summary.scalar("accuracy", t_acc, step=epoch)
            tf.summary.scalar('learning_rate', float(optimizer.learning_rate), step=epoch)

        acc_metric.reset_state()
        loss_metric.reset_state()

        #Validation pass
        all_val_y = []
        all_val_preds = []

        for x, y in valid_data:
            y_pred = val_step(x, y)
            all_val_y.append(y.numpy())
            all_val_preds.append(y_pred.numpy())

        v_loss = float(loss_metric.result())
        v_acc = float(acc_metric.result())
        # Build and log confusion matrix image to TensorBoard
        all_val_y = np.concatenate(all_val_y)
        all_val_preds = np.concatenate(all_val_preds)
        cm = get_confusion_matrix(all_val_y, all_val_preds, class_names)
        cm_image = plot_confusion_matrix(cm, class_names)

        with val_writer.as_default():
            tf.summary.scalar('loss', v_loss, step=epoch)
            tf.summary.scalar("accuracy", v_acc, step=epoch)
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        print(f"Koniec treningu: {epoch + 1} - Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")

        # Callbacks : save best, reduce LR, early stop
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_count = 0
            reduce_count = 0
            model.save(f'best_{name}.keras') # save checkpt for improvement
        else:
            patience_count += 1
            reduce_count += 1
            # Reduce LR by 80% if no improvement for 3 epochs (min LR: 1e-4)
            if reduce_count >= 3:
                new_lr = max(float(optimizer.learning_rate) * 0.2, 1e-4)
                optimizer.learning_rate.assign(new_lr)
                reduce_count = 0
        # Early stopping: halt if no improvement for 7 consecutive epochs
        if patience_count >= 7:
            break

    #Records this model's hyperparameters and final val accuracy for comparison        
    # history[name] = {'val_acc': v_acc, 'best_val_loss': best_val_loss}
    hparams = {HP_LR: LR, HP_DROPOUT: 0.4, HP_MODEL: name}

    with tf.summary.create_file_writer(f'newkaggle_logs/hparam_tuning/{name}').as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_ACCURACY, v_acc, step=1)

    # evaluate best saved model on test set
    if os.path.exists(f'best_{name}.keras'):
        best_model = tf.keras.models.load_model(f'best_{name}.keras')
        acc_metric.reset_state()
        for x, y in test_data:
            y_pred = best_model(x, training=False)
            acc_metric.update_state(y, y_pred)

        del best_model

    # Free memory before next model to avoid GPU OOm
    del model
    tf.keras.backend.clear_session()
