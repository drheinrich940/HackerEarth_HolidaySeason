import matplotlib.pyplot as plt
from constants.trainingConstants import *
from constants.loggingConstants import *
import numpy as np
import pandas as pd


def plot_training_results(history, epochs, model, save=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    fig.subplots_adjust(wspace=.35)

    # As we plot validation acc & loss after their training counterparts, we use the corresponding curve color
    #   to pain the box color of the values of interest of each curve. Second plotted curve's color can be got
    #   using the following call.
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]

    # For the first subplot dedicated to Accuracy
    ax[0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].set_yticks(np.arange(0, 1.05, 0.05))
    ax[0].legend(loc='lower right')

    # Annotate helps us draw text at a custom location corresponding to a chosen value. It allows us to select our
    #   points of interests in the curve and add a custom labelled bbox displaying the exact value of the given point.
    #   Here we display the last epoch's accuracy value, and the best accuracy encountered during training.
    ax[0].annotate('Max :%0.2f' % max(val_acc), xy=(1, max(val_acc)), xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    ax[0].annotate('Last :%0.2f' % val_acc[-1], xy=(1, val_acc[-1]), zorder=1, xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    # For the second subplot dedicated to Loss
    ax[1].plot(epochs_range, loss, label='Training Loss')
    ax[1].plot(epochs_range, val_loss, label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].set_yticks(np.arange(0, max(val_loss) + 0.05, 0.05))
    ax[1].legend(loc='upper right')

    # Annotate helps us again to display last epoch's loss value, and the minimal loss value ever computed during train.
    ax[1].annotate('Min :%0.2f' % min(val_loss), xy=(1, min(val_loss)), xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    ax[1].annotate('Last :%0.2f' % val_loss[-1], xy=(1, val_loss[-1]), zorder=1, xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    # Plot saving procedure.
    # Handles several variables to build a representative file name, including :
    #   - Model name
    #   - Optional SE
    #   - Number of epochs
    #   - Batch size
    #   - Data augmentation used, including horizontal and vertical flips
    #   - Stages of the model if it has any
    #   - Filters if it has stages
    #   - Training iteration of the model during the project.
    if save:
        data_aug = ''
        se = ''
        if H_FLIP:
            data_aug = data_aug + 'H_FLIP'
        if V_FLIP:
            data_aug = data_aug + 'V_FLIP'
        if SE_MODULES:
            se = se + 'SE_'

        text_file = open('trainingCounter.txt', 'r')
        train_cpt = text_file.readline()
        text_file.close()

        stages = 's_' + str(STAGES).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')
        filters = 'f_' + str(FILTERS).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')

        model_name = se + model.name
        path = 'TrainingResults'
        plt.savefig(path+'/'+model_name+'_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+'_'+str(data_aug)+'_'+stages+'_'+filters+'_'+train_cpt+'.png')

    plt.show()


def log_training_results(history, _model):
    # For each training, log :
    #   - model used : 'model',
    #           example : ResNet_v1
    #   - improvements used : ['imp1', 'imp2', ...],
    #           example : ['SE', 'ASPP']
    #   - loss used : 'loss',
    #           example : 'SCCE'
    #   - optimizer used : 'optim',
    #           example : 'Nadamax'
    #   - validation accuracy best, last, and history array : [best, last, [history]],
    #           example : [0.981, 0.973, [0, 0.1, 0.4, 0.981, 0.973]]
    #   - validation loss best, last, and history array : [best, last, [history]],
    #           example : [0.1, 0.12, [0.9, 0.8, 0.5, 0.1, 0.12]]
    #   - data aug used : ['data_aug1', 'data_aug2', ... ],
    #           example : ['V_FLIP', 'H_FLIP']
    #   - training iteration id (from file) : n,
    #           example : 5
    #   ? - train validation split ratio : n,
    #           example : 0.2
    #   ? - train validation split seed : n,
    #           example : 123
    text_file = open('trainingCounter.txt', 'r')
    train_cpt = text_file.readline()
    text_file.close()

    _improvements = []
    if SE_MODULES:
        _improvements.append('SE')
    if ASPP:
        _improvements.append('ASPP')
    if SAM:
        _improvements.append('SAM')

    _data_aug = []
    if V_FLIP:
        _data_aug.append('V_FLIP')
    if H_FLIP:
        _data_aug.append('H_FLIP')

    df = pd.DataFrame([], columns=HEADERS)
    df = df.append({MODEL_FIELD: _model.name,
                    IMPROVEMENTS_FIELD: _improvements,
                    LOSS_FIELD: LOSS,
                    OPTIMIZER_FIELD: OPTI,
                    VAL_ACC_BEST_FIELD: max(history.history['val_accuracy']),
                    VAL_ACC_LAST_FIELD: history.history['val_accuracy'][-1],
                    VAL_ACC_HIST_FIELD: history.history['val_accuracy'],
                    VAL_LOSS_BEST_FIELD: max(history.history['val_loss']),
                    VAL_LOSS_LAST_FIELD: history.history['val_loss'][-1],
                    VAL_LOSS_HIST_FIELD: history.history['val_loss'],
                    DATA_AUG_FIELD: _data_aug,
                    TRAIN_ITER_ID_FIELD: train_cpt,
                    BATCH_ITER_ID_FIELD: None,
                    SPLIT_RATIO_FIELD: SPLIT_RATIO,
                    SPLIT_SEED_FIELD: SEED}, ignore_index=True)

    df.to_csv(TRAIN_LOGS_CSV, mode='a', header=False)

# TODO: Finish implementation
def display_training_results_from_csv():
    df = pd.read_csv(TRAIN_LOGS_CSV)
    print(df)


#display_training_results_from_csv()
