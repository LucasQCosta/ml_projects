'''
Performs hyperparameter search using Optuna (https://github.com/optuna/optuna-examples/blob/main/keras/keras_simple.py)
Uses datagen.flow_from_dataframe instead of datagen.flow_from_directory.
In this example, we optimize the validation accuracy using
Keras. We optimize hyperparameters such as the filter and kernel size, and layer activation.

References:
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning
'''

from math import exp
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam
# import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.metrics
# from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, f1_score, precision_score, precision_recall_curve
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ModelCheckpoint,
                                        ReduceLROnPlateau)
import os
import sys
import shutil
import pickle
# import argparse
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
import optuna
from optuna.integration import TFKerasPruningCallback
from keras.backend import clear_session
from keras.datasets import mnist
# To avoid the warning in
# https://github.com/tensorflow/tensorflow/issues/47554
from absl import logging
logging.set_verbosity(logging.ERROR)


# Global variables
# identifier for this simulation - use effnet as backend using val acc instead of AUC (23)
EPOCHS = 10  # maximum number of epochs
IMAGESIZE = (240, 180)      # Define the input shape of the images
INPUTSHAPE = (240, 180, 3)  # NN input
# BEST_MODEL = None # Best NN model
# CURRENT_MODEL = None
VERBOSITY_LEVEL = 1  # use 1 to see the progress bar when training and testing

# Important: output folder
auxId = 1
while os.path.isdir('../../outputs/optuna_no_backend_outputs/id_" + ID + "/' + str(auxId) + '/'):
    auxId = auxId + 1
ID = str(auxId)
OUTPUT_DIR = '../../outputs/optuna_no_backend_outputs/id_' + ID + '/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Created folder ", OUTPUT_DIR)

# additional global variables
num_desired_negative_train_examples = 324


def get_data_generators(num_desired_negative_train_examples, batch_size):
    # Define the folders for train, validation, and test data
    train_folder = '../../data_ham1000/HAM10000_images_part_1/'
    validation_folder = '../../data_ham1000/HAM10000_images_part_1/'
    test_folder = '../../data_ham1000/HAM10000_images_part_1/'

    train_csv = "../../data_ham1000/train.csv"
    test_csv = "../../data_ham1000/test.csv"
    validation_csv = "../../data_ham1000/validation.csv"

    # do not remove header
    traindf = pd.read_csv(train_csv, dtype=str)
    testdf = pd.read_csv(test_csv, dtype=str)
    validationdf = pd.read_csv(validation_csv, dtype=str)
    traindf = decrease_num_negatives(traindf, num_desired_negative_train_examples)
    testdf = decrease_num_negatives(testdf, 184)
    validationdf = decrease_num_negatives(validationdf, 76)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=train_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    # Loading and preprocessing the training, validation, and test data
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validationdf,
        directory=validation_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=test_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    return train_generator, validation_generator, test_generator


def objective(trial):  # uses effnet
    # Clear clutter from previous Keras session graphs.
    clear_session()

    num_output_neurons = 1

    batch_size = trial.suggest_int("batch_size", 1, 15)
    train_generator, validation_generator, test_generator = get_data_generators(
        num_desired_negative_train_examples, batch_size)
    # test_generator = None #not used here

    # Define the CNN model
    model = Sequential()

    # Load the respective EfficientNet model but exclude the classification layers
    trainable = False
    model_url = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2'
    extractor = hub.KerasLayer(model_url, input_shape=INPUTSHAPE, trainable=trainable)

    num_dense_layers = trial.suggest_int(
        "num_dense_layers", 0, 3)  # number of layers
    if num_dense_layers > 0:
        num_neurons_per_layer = np.zeros(num_dense_layers, dtype=np.int64)
        num_neurons_per_layer[0] = trial.suggest_int("num_neurons_L1", 10, 300)
        for i in range(num_dense_layers - 1):
            # force number of neurons to not increase
            num_neurons_per_layer[i + 1] = trial.suggest_int("num_neurons_L{}".format(
                i + 2), num_neurons_per_layer[i] // 4, num_neurons_per_layer[i])
        print("num_neurons_per_layer =", num_neurons_per_layer)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.9)

    # trial.suggest_categorical("batch_nor", [True, False])
    use_batch_normalization = False
    use_regularizers = trial.suggest_categorical("regul", [True, False])
    if use_regularizers:
        l1_weight = trial.suggest_categorical("l1_weight", [0, 1e-4, 1e-2])
        l2_weight = trial.suggest_categorical("l2_weight", [0, 1e-4, 1e-2])

    model.add(extractor)
    for i in range(num_dense_layers):
        if use_regularizers:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[i],
                    activation=trial.suggest_categorical(
                        "activation", ["tanh", "elu", "swish"]),
                    # input_shape=INPUTSHAPE,
                    kernel_regularizer=regularizers.L1L2(l1=l1_weight, l2=l2_weight),
                    bias_regularizer=regularizers.L2(l2_weight),
                    activity_regularizer=regularizers.L2(l2_weight)
                )
            )
        else:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[i],
                    activation=trial.suggest_categorical(
                        "activation", ["tanh", "elu", "swish"]),
                    # input_shape=INPUTSHAPE
                )
            )
        # first and most important rule is: don't place a BatchNormalization after a Dropout
        # https://stackoverflow.com/questions/59634780/correct-order-for-spatialdropout2d-batchnormalization-and-activation-function
        if use_batch_normalization:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    if use_regularizers:
        model.add(Dense(num_output_neurons,
                        activation="sigmoid",
                        kernel_regularizer=regularizers.L1L2(
                            l1=l1_weight, l2=l2_weight),
                        bias_regularizer=regularizers.L2(l2_weight),
                        activity_regularizer=regularizers.L2(l2_weight)
                        ))
    else:
        model.add(Dense(num_output_neurons, activation="sigmoid"))

    model.summary()

    # We compile our model with a sampled learning rate.
    learning_rate = 1e-3  # trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Define the metric for callbacks and Optuna
    if True:
        # trial.suggest_categorical("metric_to_monitor", ['val_accuracy', 'val_auc']),
        metric_to_monitor = ('val_accuracy',)
    metric_mode = 'max'
    early_stopping = EarlyStopping(
        monitor=metric_to_monitor[0], patience=3, mode=metric_mode, restore_best_weights=True)

    # look at https://www.tensorflow.org/guide/keras/serialization_and_saving
    # do not use HDF5 (.h5 extension)
    best_model_name = 'optuna_best_model_' + str(trial.number)
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    best_model_save = ModelCheckpoint(
        best_model_name, save_best_only=True, monitor=metric_to_monitor[0], mode=metric_mode)

    reduce_lr_loss = ReduceLROnPlateau(
        monitor=metric_to_monitor[0], factor=0.5, patience=3, verbose=VERBOSITY_LEVEL, min_delta=1e-4, mode=metric_mode)
    # Define Tensorboard as a Keras callback
    tensorboard = TensorBoard(
        log_dir='../outputs/tensorboard_logs',
        # log_dir= '.\logs',
        histogram_freq=1,
        write_images=True
    )

    print("")
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print("  Hyperparameters of Optuna trial # ", trial.number)
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    model.compile(
        loss="binary_crossentropy",
        # optimizer=RMSprop(learning_rate=learning_rate),
        optimizer=Adam(learning_rate=learning_rate),
        # always use both metrics, and choose one to guide Optuna
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )

    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=VERBOSITY_LEVEL,
        # callbacks=[early_stopping,reduce_lr_loss, tensorboard]
        # callbacks=[TFKerasPruningCallback(trial, metric_to_monitor), early_stopping]
        # callbacks=[early_stopping, best_model_save, reduce_lr_loss]
        callbacks=[early_stopping, best_model_save, reduce_lr_loss, tensorboard,
                   TFKerasPruningCallback(trial, metric_to_monitor[0])]
    )
    # CURRENT_MODEL = tf.keras.models.clone_model(model)

    # add to history
    history.history['num_desired_train_examples'] = train_generator.samples

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    pickle_file_path = os.path.join(
        OUTPUT_DIR, 'optuna_best_model_' + str(trial.number), 'trainHistoryDict.pickle')
    with open(pickle_file_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    if True:
        # train data
        print('Train loss:', history.history['loss'][-1])
        print('Train accuracy:', history.history['accuracy'][-1])
        print('Train AUC:', history.history['auc'][-1])

    if True:  # test data cannot be used in model selection. This is just sanity check
        test_loss, test_accuracy, test_auc = model.evaluate(
            test_generator, verbose=VERBOSITY_LEVEL)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)
        print('Test AUC:', test_auc)

    # Evaluate the model accuracy on the validation set.
    # val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=VERBOSITY_LEVEL)
    val_accuracy = history.history['val_accuracy'][-1]
    val_auc = history.history['val_auc'][-1]
    # print('Val loss:', val_loss)
    # print('Val accuracy:', val_accuracy)
    # print('Val AUC:', val_auc)
    # avoid above by using pre-calculated:
    print('Val loss:', history.history['val_loss'][-1])
    print('Val accuracy:', val_accuracy)
    print('Val AUC:', val_auc)

    # Optuna needs to use the same metric for all evaluations (it could be val_accuracy or val_auc but one cannot change it for each trial)

    # trial.suggest_categorical("metric_to_monitor", ['val_accuracy', 'val_auc']),
    if metric_to_monitor[0] == 'val_accuracy':
        return val_accuracy
    elif metric_to_monitor[0] == 'val_auc':
        return val_auc
    else:
        raise Exception("Metric must be val_auc or val_accuracy")


def decrease_num_negatives(df, desired_num_negative_examples):
    '''
    Create dataframe with desired_num_rows rows from df
    '''
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df['target'] == '0'].copy()
    neg_examples = neg_examples.head(round(desired_num_negative_examples)).copy()

    pos_examples = shuffled_df[shuffled_df['target'] == '1'].copy()
    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True)  # shuffle again
    return newdf


if __name__ == '__main__':
    print("=====================================")
    print("Model selection")

    # copy script
    copied_script = os.path.join(OUTPUT_DIR, os.path.basename(sys.argv[0]))
    shutil.copy2(sys.argv[0], copied_script)
    print("Just copied current script as file", copied_script)

    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                study_name="Skin_Problem",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner())
    # study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=7)

    print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trial
    print("Best trial is #", trial.number)
    print("  Value: {}".format(trial.value))

    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(OUTPUT_DIR, 'best_optuna_trial.txt'), 'w') as f:
        f.write("Best trial is #" + str(trial.number))
        f.write('\n')
        f.write("  Value: {}".format(trial.value))
        f.write('\n')
        f.write("  Hyperparameters: ")
        f.write('\n')
        for key, value in trial.params.items():
            f.write("    {}: {}".format(key, value))
            f.write('\n')

    pickle_file_path = os.path.join(OUTPUT_DIR, 'study.pickle')
    with open(pickle_file_path, 'wb') as file_pi:
        pickle.dump(study, file_pi)
    print("Wrote", pickle_file_path)

    # https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html
    plt.close("all")
    plt.figure()
    # Using optuna.visualization.plot_optimization_history(study) invokes the other Optuna's backend. To use matplotlib, use:
    optuna.visualization.matplotlib.plot_optimization_history(
        study)  # optimization history
    # Save the figure to a file (e.g., "optimization_history.png")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_optimization_history.png'))
    plt.close("all")
    # fig.show()
    optuna.visualization.matplotlib.plot_intermediate_values(
        study)  # Visualize the loss curves of the trials
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_loss_curves.png'))
    # fig.show()
    plt.close("all")
    optuna.visualization.matplotlib.plot_contour(study)  # Parameter contour plots
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_contour_plots.png'))
    # fig.show()
    plt.close("all")
    optuna.visualization.matplotlib.plot_param_importances(
        study)  # parameter importance plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_parameter_importance.png'))
    # fig.show()
