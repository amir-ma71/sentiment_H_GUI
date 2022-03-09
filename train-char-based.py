import codecs
import pickle

from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers

from keras import backend as K

def load_data(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    x = np.asarray(list(x))
    y = to_categorical(y, 3)

    return x, y


x_token_train, y_token_train = load_data('src/data/token_train.tsv')
x_token_test, y_token_test = load_data('src/data/token_test.tsv')
x_morph_train, y_morph_train = load_data('src/data/morph_train.tsv')
x_morph_test, y_morph_test = load_data('src/data/morph_test.tsv')

# print('X token train shape: {}'.format(x_token_train.shape))
# print('X token test shape: {}'.format(x_token_test.shape))
#
# print('X morph train shape: {}'.format(x_morph_train.shape))
# print('X morph test shape: {}'.format(x_morph_test.shape))

from keras.preprocessing import text, sequence


def tokenizer(x_train, x_test, vocabulary_size, char_level):
    tokenize = text.Tokenizer(num_words=vocabulary_size,
                              char_level=char_level,
                              filters='')
    tokenize.fit_on_texts(x_train)  # only fit on train
    with open('src/tokens.pickle', 'wb') as handle:
        pickle.dump(tokenize, handle)
    x_train = tokenize.texts_to_sequences(x_train)
    # x_train2 = tokenize.texts_to_matrix(x_train)
    x_test = tokenize.texts_to_sequences(x_test)

    return x_train, x_test


def pad(x_train, x_test, max_document_length):
    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length, padding='post', truncating='post')

    return x_train, x_test


vocabulary_size = 5000

x_token_train, x_token_test = tokenizer(x_token_train, x_token_test, vocabulary_size, True)
# x_morph_train, x_morph_test = tokenizer(x_morph_train, x_morph_test, vocabulary_size, True)

max_document_length = 300

x_token_train, x_token_test = pad(x_token_train, x_token_test, max_document_length)
# x_morph_train, x_morph_test = pad(x_morph_train, x_morph_test, max_document_length)

import matplotlib.pyplot as plt


def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model Accuracy')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.show()

dropout_keep_prob = 0.5
embedding_size = 300
batch_size = 100
lr = 1e-4
dev_size = 0.2

num_epochs = 5

# Create new TF graph
K.clear_session()

# Construct model
convs = []
text_input = Input(shape=(max_document_length,))
x = Embedding(vocabulary_size, embedding_size)(text_input)
for fsz in [10, 30]:
    conv = Conv1D(128, fsz, padding='valid', activation='relu')(x)
    pool = MaxPool1D()(conv)
    convs.append(pool)
x = Concatenate(axis=1)(convs)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(dropout_keep_prob)(x)
preds = Dense(3, activation='softmax')(x)

model = Model(text_input, preds)

adam = optimizers.Adam(lr=lr)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# Train the model
history = model.fit(x_token_train, y_token_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=dev_size)

# Plot training accuracy and loss
plot_loss_and_accuracy(history)

# Evaluate the model
scores = model.evaluate(x_token_test, y_token_test,
                       batch_size=batch_size, verbose=1)
print('\nAccurancy: {:.3f}'.format(scores[1]))

# Save the model
model.save('char_saved_models/CNN-Token-{:.3f}.h5'.format((scores[1] * 100)))
# pickle.dump(model, open('char_saved_models/CNN2-Token-{:.3f}.pkl'.format((scores[1] * 100)), 'wb'))