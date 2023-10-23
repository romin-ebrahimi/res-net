import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import layers
from keras import metrics
from keras import regularizers
from sklearn.metrics import average_precision_score
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def valid_test_split(X, y, valid_index: int = None, valid_size: float = 0.5):
    """
    Given X and y, split the data into a validation and test sets
    Args:
        X: Input covariate data.
        y: Input target data.
        valid_index: Ending index of the validation set (optional)
        valid_size: Percentage of input data to split into the
            validation set.
    Returns:
        Four arrays including validation inputs, testing inputs,
        validation labels, and testing labels.
    """
    if valid_index is None:
        valid_index = int(X.shape[0] * valid_size)

    X_valid = X[:valid_index,]
    X_test = X[valid_index:,]

    if y is not None:
        Y_valid = y[:valid_index,]
        Y_test = y[valid_index:,]
    else:
        Y_valid = None
        Y_test = None

    return X_valid, X_test, Y_valid, Y_test


def time_series_gen(
    data,
    sampling_rate: int = 1,
    train_size: float = 0.8,
    sequence_length: int = 64,
    target=None,
):
    """
    Given a sequence of data-points gathered at equal intervals,
    along with time series parameters such as the length of the
    sequences used to produce batches of timeseries inputs and
    targets. Note: first element of tensor is the most recent
    data point i.e. X_train[0,][0] will be the most recent data
    observation. This is because the keras utility discards
    sampling_rate observations at the end when a full sequence
    can't be evenly generated. Also, the data and target must be
    reversed prior to being passed into the keras function to
    avoid the misalignment caused by the discarding and bias
    associated with the way the function handles a sequence of
    indices [1,2,3,...]. After processing, the data is reversed
    again to preserve the original temporal relationship where
    row index 0 is the oldest date.
    Args:
        data: Input dataframe.
        sampling_rate: Downsample rate of data.
        train_size: Percentage of data to use for training split.
        sequence_length: Number of lags to include in each row.
        target: Dataframe of target class labels.

    Returns:
        Four arrays including training inputs, testing inputs,
        training labels, and testing labels.
    """
    # Reverse data prior to passing to keras.
    data = data[::-1]

    if target is not None:
        # Reverse target to align with data.
        target = target[::-1]
        X_train = timeseries_dataset_from_array(
            data=data.astype("float32"),
            targets=target.astype("int32"),
            sequence_length=sequence_length,
            sampling_rate=sampling_rate,
            batch_size=len(data),
            shuffle=False,
        )

        for batch in X_train:
            X_train, Y_train = batch
    else:
        X_train = timeseries_dataset_from_array(
            data=data.astype("float32"),
            targets=None,
            sequence_length=sequence_length,
            sampling_rate=sampling_rate,
            batch_size=len(data),
            shuffle=False,
        )

        for batch in X_train:
            X_train = batch

    # Undo the original reversal prior to returning data.
    X_train = X_train[::-1]
    # Index to split train and test data.
    test_index = int(len(X_train) * train_size)
    X_test = X_train[test_index:,]
    X_train = X_train[:test_index,]
    if target is not None:
        # Undo the reversal.
        Y_train = Y_train[::-1]
        Y_test = Y_train[test_index:,]
        Y_train = Y_train[:test_index,]
    else:
        Y_test = None
        Y_train = None

    return X_train, X_test, Y_train, Y_test


def compute_class_weight(arr: np.array) -> dict:
    """
    Args:
        arr: Input np.array of categorical classes where
            each column i represents binary labels for
            class i and the rows represent samples.

    Returns:
        A dictionary of the class weights needed for keras.
    """
    cw = dict()
    n_samples = arr.shape[0]
    n_classes = arr.shape[1]

    for i in range(n_classes):
        # Count of class occurrence.
        class_cnt = float(sum(arr[:, i]))
        cw[int(i)] = n_samples / (n_classes * class_cnt)

    return cw


def residual_block(x, params: dict, regularizer):
    """
    Given an input tensor and parameters for convolutional
    filter, create a residual block with convolutional filters,
    an identity skip connection, and preactivation units.

    Args:
        x: initial input.
        params: dict of parameters for neural network layers.
        regularizer: L1 or L2 regularization.

    Returns:
        Residual block containing 2 convolutions and a skip
        connection using batch normalization and ReLU.
    """
    x_skip = x  # skip connection
    x = layers.BatchNormalization()(x)  # pre-activation BN/ReLU
    x = layers.ReLU()(x)
    x = layers.Conv1D(
        params["filter_out"],
        kernel_size=params["kernel_size"],
        strides=params["strides"],
        padding="same",  # same output dimensions as previous layer
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization()(x)  # pre-activation BN/ReLU
    x = layers.ReLU()(x)
    x = layers.Conv1D(
        params["filter_out"],
        kernel_size=params["kernel_size"],
        strides=params["strides"],
        padding="same",  # same output dimensions as previous layer
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    # Equal weight skip connection and convolutions.
    x = layers.Add()([x_skip, x])

    return x


# Callback that dynamically reduces learning rate.
reduce_lr = ReduceLROnPlateau(
    monitor="loss",  # Use 'val_' prefix for validation set.
    factor=0.5,
    cooldown=0,
    patience=1,
    min_lr=1e-6,
)

early_stop = EarlyStopping(restore_best_weights=True, patience=10)


def get_model(params: dict = {}):
    """
    Using the Keras functional API, intialize the
    convolutional neural network model with residual
    blocks utilizing skip connections (ResNet).
    """
    mirrored_strategy = (
        tf.distribute.MirroredStrategy()
    )  # for utilizing multiple GPUs

    with mirrored_strategy.scope():
        default_params = {}  # params input modifies the defaults
        default_params["n_blocks"] = 3  # number of residual blocks
        default_params["n_classes"] = 3
        default_params["input_length"] = 64
        default_params["l2"] = 0.1
        default_params["dropout_rate"] = 0.1
        default_params["kernel_size"] = 3  # size of the Conv1D filter
        default_params["strides"] = 1  # stride length of the filter
        default_params["loss"] = "categorical_crossentropy"
        default_params["metrics"] = [
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.CategoricalCrossentropy(name="categorical_crossentropy"),
            metrics.CategoricalAccuracy(name="categorical_accuracy"),
        ]

        params = {**default_params, **params}
        params["filter_out"] = int(
            (
                (params["input_length"] - params["kernel_size"])
                / params["strides"]
            )
            + 1
        )
        if params["n_classes"] == 2:
            params["loss"] = "binary_crossentropy"  # binary case

        L2_regularizer = regularizers.L2(
            params["l2"]
        )  # regularization for dense layers

        input_0 = layers.Input(
            shape=(
                params["input_length"],
                1,
            ),
            dtype="float32",
        )
        input_1 = layers.Input(
            shape=(
                params["input_length"],
                1,
            ),
            dtype="float32",
        )

        normalize_0 = layers.Normalization(axis=1)(input_0)
        normalize_1 = layers.Normalization(axis=1)(input_1)

        for i in range(params["n_blocks"]):
            if i == 0:
                res_0 = residual_block(
                    x=normalize_0, params=params, regularizer=L2_regularizer
                )
            else:
                res_0 = residual_block(
                    x=res_0, params=params, regularizer=L2_regularizer
                )

        for i in range(params["n_blocks"]):
            if i == 0:
                res_1 = residual_block(
                    x=normalize_1, params=params, regularizer=L2_regularizer
                )
            else:
                res_1 = residual_block(
                    x=res_1, params=params, regularizer=L2_regularizer
                )

        res_0 = layers.ReLU()(res_0)
        res_1 = layers.ReLU()(res_1)
        gap_0 = layers.GlobalAveragePooling1D()(res_0)
        gap_1 = layers.GlobalAveragePooling1D()(res_1)
        concat = layers.concatenate([gap_0, gap_1])
        concat = layers.Flatten()(concat)

        if params["n_classes"] == 2:  # binary case
            output_layer = layers.Dense(
                params["n_classes"], activation="sigmoid"
            )(concat)
        else:
            output_layer = layers.Dense(
                params["n_classes"],  # softmax for n_classes > 2
                activation="softmax",
            )(concat)

        model = Model(inputs=[input_0, input_1], outputs=output_layer)

        optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(
        loss=params["loss"], optimizer=optimizer, metrics=params["metrics"]
    )

    return model


def get_model_lstm(params: dict = {}):
    """
    Using the Keras functional API, intialize the
    convolutional neural network model with lstm.
    """
    # Initialization for utilizing multiple GPUs.
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        default_params = {}  # params input modifies the defaults
        default_params["n_blocks"] = 3  # number of residual blocks
        default_params["n_classes"] = 3
        default_params["dense_size"] = 16
        default_params["input_length"] = 64
        default_params["l2"] = 0.1
        default_params["dropout_rate"] = 0.1
        default_params["kernel_size"] = 3  # size of the Conv1D filters
        default_params["strides"] = 1  # stride length of filters
        default_params["lstm_units"] = 128  # number of units in lstm layers
        default_params["loss"] = "categorical_crossentropy"
        default_params["metrics"] = [
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.CategoricalCrossentropy(name="categorical_crossentropy"),
            metrics.CategoricalAccuracy(name="categorical_accuracy"),
        ]

        params = {**default_params, **params}
        params["filter_out"] = int(
            (
                (params["input_length"] - params["kernel_size"])
                / params["strides"]
            )
            + 1
        )
        # Set binary cross entropy loss for binary case.
        if params["n_classes"] == 2:
            params["loss"] = "binary_crossentropy"
        # Regularization for dense layers.
        L2_regularizer = regularizers.L2(params["l2"])

        input_0 = layers.Input(
            shape=(
                params["input_length"],
                1,
            ),
            dtype="float32",
        )
        input_1 = layers.Input(
            shape=(
                params["input_length"],
                1,
            ),
            dtype="float32",
        )

        normalize_0 = layers.Normalization(axis=1)(input_0)
        normalize_1 = layers.Normalization(axis=1)(input_1)

        for i in range(params["n_blocks"]):
            if i == 0:
                res_0 = residual_block(
                    x=normalize_0, params=params, regularizer=L2_regularizer
                )
            else:
                res_0 = residual_block(
                    x=res_0, params=params, regularizer=L2_regularizer
                )

        for i in range(params["n_blocks"]):
            if i == 0:
                res_1 = residual_block(
                    x=normalize_1, params=params, regularizer=L2_regularizer
                )
            else:
                res_1 = residual_block(
                    x=res_1, params=params, regularizer=L2_regularizer
                )

        res_0 = layers.ReLU()(res_0)
        lstm1_0 = layers.LSTM(
            units=params["lstm_units"],
            return_sequences=False,
            kernel_regularizer=L2_regularizer,
            recurrent_regularizer=L2_regularizer,
            bias_regularizer=L2_regularizer,
            activity_regularizer=L2_regularizer,
        )(res_0)
        lstm1_0 = layers.BatchNormalization()(lstm1_0)

        res_1 = layers.ReLU()(res_1)
        lstm1_1 = layers.LSTM(
            units=params["lstm_units"],
            return_sequences=False,
            kernel_regularizer=L2_regularizer,
            recurrent_regularizer=L2_regularizer,
            bias_regularizer=L2_regularizer,
            activity_regularizer=L2_regularizer,
        )(res_1)
        lstm1_1 = layers.BatchNormalization()(lstm1_1)

        concat = layers.concatenate([lstm1_0, lstm1_1])
        concat = layers.Flatten()(concat)

        dense_layer = layers.Dense(
            params["dense_size"],
            kernel_regularizer=L2_regularizer,
            bias_regularizer=L2_regularizer,
            activity_regularizer=L2_regularizer,
            activation="relu",
        )(concat)
        dense_layer = layers.BatchNormalization()(dense_layer)
        dense_layer = layers.ReLU()(dense_layer)
        dense_layer = layers.Dropout(params["dropout_rate"])(dense_layer)

        # Binary case.
        if params["n_classes"] == 2:
            output_layer = layers.Dense(
                params["n_classes"], activation="sigmoid"
            )(dense_layer)
        # Use softmax for case when n_classes > 2.
        else:
            output_layer = layers.Dense(
                params["n_classes"], activation="softmax"
            )(dense_layer)

        model = Model(inputs=[input_0, input_1], outputs=output_layer)

        optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(
        loss=params["loss"], optimizer=optimizer, metrics=params["metrics"]
    )

    return model


def learning_curves(history, metric: str = "accuracy") -> None:
    """
    Given the history object from training a keras model summarize
    the training and validation learning curves for a given metric.
    Burn the first 5 epochs as they are very noisy.
    """
    plt.plot(history[metric][5:], color="#2164F3")
    leg = ["train"]
    if history.get(f"val_{metric}") is not None:
        plt.plot(history[f"val_{metric}"][5:], color="#FF6600")
        leg.append("valid")

    plt.title(f"{metric} learning curves")
    plt.ylabel(f"{metric}")
    plt.xlabel("epoch")
    plt.legend(leg, loc="lower right")
    plt.show()

    return None


def pairwise_compare(y_true, y_pred0, y_pred1, threshold=0.5) -> float:
    """
    Function for pairwise statistical comparison of two models
    using the same test set. Returns the wald test statistic
    testing the difference in classification performance.
    """
    y_label0 = y_pred0
    y_label1 = y_pred1
    y_label0[y_label0 >= threshold] = 1
    y_label1[y_label1 >= threshold] = 1
    y_label0[y_label0 != 1] = 0
    y_label1[y_label1 != 1] = 0

    # Label should be 1 when correct.
    y_label0[((y_true - y_label0) == 0)] = 1
    y_label1[((y_true - y_label1) == 0)] = 1
    y_label0[((y_true - y_label0) != 0)] = 0
    y_label1[((y_true - y_label1) != 0)] = 0

    # Non-parameteric label accuracy delta.
    delta = y_label0 - y_label1
    delta_se = np.std(delta)

    # Wald test statistic.
    wald = np.abs(np.mean(delta) / (delta_se / np.sqrt(delta.shape[0])))

    return wald


def rolling_precision(target, pred, increment: int = 1000) -> None:
    """
    Calculate rolling precision from multiclass target array
    and predicted class probabilities given by pred.
    """
    target_arr = to_categorical(target, num_classes=pred.shape[1])

    pr = []  # List of rolling precision.
    for i in range(increment, (pred.shape[0] - increment), increment):
        pr.append(
            average_precision_score(
                y_true=target_arr[i - increment : i,],
                y_score=pred.iloc[i - increment : i,],
            )
        )

    plt.plot(pr, color="#2164F3")
    plt.title(f"Rolling {increment} Precision Window")
    plt.xlabel("time")
    plt.ylabel("average precision")
    plt.show()

    return None
