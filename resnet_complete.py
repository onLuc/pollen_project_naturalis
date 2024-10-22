import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Create directories to save plots, confusion matrices, and runtimes
os.makedirs("nn_plots/", exist_ok=True)
os.makedirs("confusion_matrices/", exist_ok=True)
os.makedirs("models/", exist_ok=True)

# Set seeds for consistent results between runs.
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


def scale(image, label):
    """
    Scales an entire dataset containing images from range [0,255] to [0,1]

    Parameters
    ----------
    image : Numpy Array
        A cv2 representation of an image.
    label : -
        Needs to be there for the function to work.

    Returns
    -------
    image : TYPE
        A cv2 representation of an image, scaled to range [0,1].
    label : -
        Needs to be there for the function to work.

    """
    image = tf.cast(image, tf.float32) / 255.0  # Scale image values
    return image, label


def generate_dataset(train_dir):
    """
    All images are loaded into the script, processed as a Keras object and
    separated into three subsets.

    Parameters
    ----------
    train_dir : String
        Folder where all the pictures are for training the network are located.

    Returns
    -------
    train_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for training.
    val_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for validation.
    test_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for testing.
    num_classes : Int
        The number of classes (6).
    dataset.class_names : List
        Names of all the classes.

    """
    # Load the dataset using a TensorFlow Keras method.
    dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical',
        shuffle=True)
    
    num_classes = len(dataset.class_names)
    scaled_dataset = dataset.map(scale)

    # Parameters for splitting the dataset into three parts: train, val, and
    # test. (should equal 1 when cumulated)
    train_size = 0.8
    val_size = 0.15
    test_size = 0.05
    
    # Dataset is split into the size of the parts defined above.
    train_ds = scaled_dataset.take(int(len(dataset) * train_size))
    test_val_ds = scaled_dataset.skip(int(len(dataset) * train_size))
    val_ds = test_val_ds.take(int(len(test_val_ds) * (val_size / (val_size + test_size))))
    test_ds = test_val_ds.skip(int(len(test_val_ds) * (val_size / (val_size + test_size))))
    
    
    return train_ds, val_ds, test_ds, num_classes, dataset.class_names


def create_model(base_model, num_classes):
    """
    Builds a model with a pre-trained ResNet backbone, global average pooling, 
    and fully connected layers.

    Parameters
    ----------
    base_model : ResNet model (Keras class)
        One of the 6 ResNet models.
    num_classes : Int
        The number of classes (6).

    Returns
    -------
    model : ResNet model 
        With two layers added for optimization and adaptation to the number of
        classes.

    """
    # The ResNet model gets inserted into a preset model.
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model, train_ds, val_ds):
    """
    The model is fitted to the available training data. 2 callbacks are
    implemented: reduced learning rate and early stopping.
    
    Parameters
    ----------
    model : Keras neural network sequential model
        Contains the ResNet model in the custom sequential model.
    train_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for training.
    val_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for validation.

    Returns
    -------
    history : dictionary
        Contains the performance of the model.

    """
    # Callbacks for learning rate reduction and early stopping.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=[reduce_lr, early_stopping]
    )    

    return history
    
    
def plot_acc_loss(history, name, tot_time):
    """
    Accuracy and Loss data are collected, plotted, and saved using pyplot.

    Parameters
    ----------
    history : dictionary
        Contains the performance of the model.
    name : String
        Name of the CNN being trained.
    tot_time : Float
        Time spent training the script.

    Returns
    -------
    None.

    """
    # Get accuracy and loss history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # Plot accuracy and loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, color='blue', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='red', label='Validation Accuracy')
    plt.plot(epochs, loss, color='green', label='Training Loss')
    plt.plot(epochs, val_loss, color='orange', label='Validation Loss')
    plt.ylim(0, 1)
    plt.title(f'Accuracy and Loss of {name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'nn_plots/{name}_time_{tot_time}_valacc_{val_acc[-1]}_valloss_{val_loss[-1]}_plot.png')
    plt.close()
    

def generate_cm(test_ds, model, name, class_names):
    """
    

    Parameters
    ----------
    test_ds : <class 'tensorflow.python.data.ops.take_op._TakeDataset'>
        Part of the dataset used for testing.
    model : Keras neural network sequential model
        Contains the ResNet model in the custom sequential model.
    name : String
        Name of the CNN being trained.
    class_names : List
        Names of all the species.

    Returns
    -------
    None.

    """
    y_pred = []
    y_true = []

    # Iterate over the test dataset and collect predictions and true labels.
    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # Compute the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot and save the confusion matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f'Confusion Matrix of {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices/{name}_confusion_matrix.png')
    plt.close()


def main():
    """
    Main function that calls all major function and structures the process of
    training all newtorks in base_models from data generation to results.

    Returns
    -------
    None.

    """
    # Data set in chunks is saved as train, val, and test dataset.
    train_ds, val_ds, test_ds, num_classes, class_names = generate_dataset('nn_input')
    
    # All pre-trained neural networks to be used for training.
    base_models = {
        "ResNet 50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 101": ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 152": ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 50 v2": ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 101 v2": ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 152 v2": ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    }
    
    # Takes every model from base_models to train the model and generate
    # results.
    for name, base_model in base_models.items():
        
        start_time = time.time()
        
        # Model is created with right hyperparameters and trained.
        model = create_model(base_model, num_classes)
        history = train_model(model, train_ds, val_ds)
        
        end_time = time.time()
        tot_time = end_time - start_time
        
        model.save(f"models/{name}.keras")
        
        # All results are generated.
        plot_acc_loss(history, name, tot_time)
        generate_cm(test_ds, model, name, class_names)
        
        
main()