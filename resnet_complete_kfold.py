import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
import time

# Create directories to save cross-validation results and training times
os.makedirs("kfold", exist_ok=True)
os.makedirs("kfold_times", exist_ok=True)

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
    Loads the dataset, scales the images, and converts it into numpy arrays 
    for use in K-fold cross-validation.

    Parameters
    ----------
    train_dir : String
        Directory where training images are located.

    Returns
    -------
    X : Numpy Array
        Scaled images.
    y : Numpy Array
        Corresponding labels.
    num_classes : Int
        The number of classes.
    class_names : List
        Names of all the classes.

    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical')

    num_classes = len(dataset.class_names)
    scaled_dataset = dataset.map(scale)

    # Convert the dataset to numpy arrays for easier KFold processing
    X, y = [], []
    for images, labels in scaled_dataset:
        X.append(images.numpy())
        y.append(labels.numpy())

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y, num_classes, dataset.class_names


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


def train_model_kfold(X, y, num_classes, base_model, name, k_folds=5):
    """
    Trains a model using K-fold cross-validation and calculates performance metrics 
    (precision, recall, F1 score) across all folds.

    Parameters
    ----------
    X : Numpy Array
        Scaled images.
    y : Numpy Array
        Corresponding labels.
    num_classes : Int
        The number of classes (6).
    base_model : ResNet model (Keras class)
        One of the 6 ResNet models.
    name : String
        Name of the CNN being trained.
    k_folds : Int, optional
        Number of cross-validation folds (5).

    Returns
    -------
    None.

    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=2)

    # Lists to store performance metrics for each fold
    all_precision, all_recall, all_f1 = [], [], []
    
    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"Fold {fold}/{k_folds}")
        
        # Split data into train and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to tf.data.Dataset for training
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

        # Create and compile model
        model = create_model(base_model, num_classes)

        # Callbacks for learning rate reduction and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model on the current fold
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=40,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )

        # Predict on the validation set
        y_pred = np.argmax(model.predict(X_val), axis=1)
        y_true = np.argmax(y_val, axis=1)

        # Calculate precision, recall, and F1 score for the current fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        fold += 1
    
    # Compute mean metrics across all folds
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)
    mean_f1 = np.mean(all_f1)

    # Save metrics to file
    with open(f"kfold/{name}.txt", 'w') as file:
        file.write(f"precision:\t{mean_precision}\n"
                   f"recall:\t\t{mean_recall}\n"
                   f"f1-score:\t{mean_f1}")


def main():
    """
    Main function that handles K-fold cross-validation for multiple pre-trained 
    ResNet models, measuring performance and saving results.

    Returns
    -------
    None.

    """
    # Load and prepare dataset
    X, y, num_classes, class_names = generate_dataset('nn_input')

    # All pre-trained neural networks to be used for training.
    base_models = {
        "ResNet 50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 101": ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 152": ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 50 v2": ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 101 v2": ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "ResNet 152 v2": ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    }

    # Train each model using K-fold cross-validation
    for name, base_model in base_models.items():
        start_time = time.time()
        print(f"Training {name} with 5-fold Cross-Validation")
        
        # Perform K-fold cross-validation
        train_model_kfold(X, y, num_classes, base_model, name)
        
        # Save training time
        end_time = time.time()
        tot_time = end_time - start_time
        with open(f"kfold_times/5-fold_{name}.txt", 'w') as file:
            file.write(f"{tot_time}")


main()
