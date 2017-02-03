from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.data_utils import VocabularyProcessor
from sklearn import model_selection

data = pd.read_csv('ign.csv')

# drop columns that are not used
data = data.drop(['Unnamed: 0','url', 'release_year', 'release_month', 'release_day', 'platform', 'score', 'genre', 'editors_choice'], axis=1)


# convert labels to 1 and 0
data.loc[data["score_phrase"] == "Disaster", "score_phrase"] = 0
data.loc[data["score_phrase"] == "Unbearable", "score_phrase"] = 0
data.loc[data["score_phrase"] == "Painful", "score_phrase"] = 0
data.loc[data["score_phrase"] == "Bad", "score_phrase"] = 0
data.loc[data["score_phrase"] == "Mediocre", "score_phrase"] = 0
data.loc[data["score_phrase"] == "Awful", "score_phrase"] = 0

data.loc[data["score_phrase"] == "Amazing", "score_phrase"] = 1
data.loc[data["score_phrase"] == "Great", "score_phrase"] = 1
data.loc[data["score_phrase"] == "Okay", "score_phrase"] = 1
data.loc[data["score_phrase"] == "Masterpiece", "score_phrase"] = 1
data.loc[data["score_phrase"] == "Good", "score_phrase"] = 1

# tokenize title
# data["token_title"] = data["title"].apply(nltk.word_tokenize)
word_processor = VocabularyProcessor(100)
tmp = np.array(list(word_processor.fit_transform(data["title"])))

# split into train and test data
train_data, test_data = model_selection.train_test_split(data, train_size=0.9)
trainX = np.array(list(word_processor.fit_transform(train_data["title"])))
trainY = train_data.loc[:, ["score_phrase"]].as_matrix()
testX = np.array(list(word_processor.fit_transform(test_data["title"])))
testY = test_data.loc[:, ["score_phrase"]].as_matrix()

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=16762, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY,  n_epoch=12, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
