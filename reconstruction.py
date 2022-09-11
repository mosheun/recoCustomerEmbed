import keras
import pandas as pd
import Process
# MODEL PARAM #
OPTIMIZER = 'adam'
LOSS = 'mse'
MF_LOSS = 'mse'
VERBOSE = 2
MF_EPOCS = 10
RECONSTRCUT_LENGTH = 15

def getStaticModel(numOutput):
    embed1_input = keras.layers.Input(shape=(15,), name='Embed1')
    embed1_input_vec = keras.layers.Dropout(0.2)(embed1_input)

    embed2_input = keras.layers.Input(shape=(15,), name='Embed2')
    embed2_input_vec = keras.layers.Dropout(0.2)(embed2_input)

    embed3_input = keras.layers.Input(shape=(15,), name='Embed3')
    embed3_input_vec = keras.layers.Dropout(0.2)(embed3_input)

    concat = keras.layers.concatenate([embed1_input_vec, embed2_input_vec, embed3_input_vec], name='Concat')
    concat_dropout = keras.layers.Dropout(0.2)(concat)

    dense = keras.layers.Dense(80, name='FullyConnected')(concat_dropout)
    dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
    dropout_1 = keras.layers.Dropout(0.2, name='Dropout-1')(dense_batch)
    dense_2 = keras.layers.Dense(40, name='FullyConnected-1')(dropout_1)
    dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)

    dropout_2 = keras.layers.Dropout(0.2, name='Dropout-2')(dense_batch_2)
    dense_3 = keras.layers.Dense(20, name='FullyConnected-2')(dropout_2)
    dense_4 = keras.layers.Dense(10, name='FullyConnected-3', activation='relu')(dense_3)

    reconstructedEmbedding = keras.layers.Dense(numOutput, activation='relu', name='Prediction')(dense_4)
    model = keras.Model([embed1_input, embed2_input, embed3_input], reconstructedEmbedding)
    model.compile(optimizer=OPTIMIZER, loss=MF_LOSS)
    return model

def fitModel(model, embed1, embed2, embed3, reconstructedEmbedding):
    history = model.fit([embed1, embed2, embed3], reconstructedEmbedding, epochs=MF_EPOCS, verbose=VERBOSE,
                            validation_split=0.1)
    return history

def trainModel(name, folder):
    print('name: ' + str(name))
    file = str(folder) + '17.7.19_Artist_User_1M.csv'
    groundTruthEmbed = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values

    model = getStaticModel(numOutput=RECONSTRCUT_LENGTH)
    h = fitModel(model, Process.getStatistics(folder), groundTruthEmbed)
