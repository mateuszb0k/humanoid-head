import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics

def plot_to_image(figure):
    """
    Converts a Matplotlib figure to a PNG tensor suitable for TensorBoard logging.
    Returns a 4D tensor of shape (1, H, W, 4) with RGBA channels.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png') # render figure into in-memory buffer
    plt.close(figure) # free memory
    buf.seek(0) # rewind buffer to start before reading

    # Decode PNG bytes
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image



def get_confusion_matrix(y_labels, logits, class_names):
    """
    Computes a confusion matrix from raw model outputs where:
    y_labels - are ground truth class indices,
    logits - are model output probabilities,
    class_names - are List of class names,
    cm - Confusion matrix of shape (num_classes, num_classes)
    """

    # Convert probabilities to predicted class indices
    preds = np.argmax(logits, axis=1)


    cm = sklearn.metrics.confusion_matrix(y_labels,preds,labels=np.arange(len(class_names)))
    return cm

def plot_confusion_matrix(cm, class_names):
    """
    Plots a normalized confusion matrix and returns it as a TensorBoard-ready image tensor.
    cm - Raw confusion matrix from get_confusion_matrix()
    class_names - List of class names for axis ticks,
    cm_image - PNG tensor of shape(1,H,W,4) ready for tf.summary.image()
    """
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    # Set axis ticks to class names; rotate x labels to avoid overlap
    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize rows to [0, 1] so each cell shows recall per class
    # +1e-7 avoids division by zero for classes with no true samples
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]+1e-7,decimals=3)

    # Annotate each cell with its normalized value
    # Use white text on dark cells, black text on light cells for contrast
    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j,i,cm[i,j],horizontalalignment="center",color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Convert figure to tensor for TensorBoard logging
    cm_image = plot_to_image(figure)
    return cm_image
