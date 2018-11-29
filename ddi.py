import csv
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from pathlib import Path

csv_list = []
file_counter = 0
sentence_list = []
word_list = []
tag_list = []
sentence_all = []  #training data for word2vec
EMBEDDING_DIM = 100
idx2tag ={}
pred_label_list = []
test_label_list = []
unique_sentence = []

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
sentence_file_dict = {}
sentence_all_dic = []
mypath = Path().absolute()
path = str(mypath) + '/trainingFiles'
for filename in os.listdir(path):
    sentence_all_dic = []
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
    file_counter += 1
    root = tree.getroot()
    for child in root:
        if child.tag == 'Sentences':
            sentences = child
            break
    snt_temp = 0
    for sentence_ind, sentence in enumerate(sentences.findall('Sentence')):
        sentenceText = sentence.find('SentenceText').text
        if sentenceText not in unique_sentence:
            unique_sentence.append(sentenceText)
            # print(nltk.pos_tag(nltk.word_tokenize(sentenceText)))
            tag_dict = {}
            for mention in sentence.findall('Mention'):
                for mentiounique_words in mention.attrib['str'].split('|'):
                    for ind, mention_word in enumerate(mentiounique_words.split()):
                        if not ind:
                            tag_dict[mention_word] = 'B-' + mention.attrib['type']
                        else:
                            tag_dict[mention_word] = 'I-' + mention.attrib['type']

            temp_list = []
            sentence_each = []
            for word_tag in nltk.pos_tag(nltk.word_tokenize(sentenceText)):
                tag = 'O'
                word = word_tag[0]
                pos = word_tag[1]
                if word in tag_dict:
                    tag = tag_dict[word]
                csv_list.append([file_counter, sentence_ind, word, pos, tag])

                word_list.append(word)
                tag_list.append(tag)
                sentence_each.append(word)
                temp_list.append((word, pos, tag))
            sentence_list.append(temp_list)
            sentence_all.append(sentence_each)
            sentence_all_dic.append(temp_list)
            snt_temp = sentence_ind
    sentence_file_dict[file_counter] = sentence_all_dic





# train word2vec model
w2vmodel = Word2Vec(sentence_all, min_count=1)
# summarize the loaded model
# print(w2vmodel)

# words = list(set(word_list))
words =[]
for word_in_word_list in word_list:
    if word_in_word_list not in words:
        words.append(word_in_word_list)

words.append("ENDPAD")
unique_words = len(words)

# Embedding matrix for word2vec model
embedding_matrix = np.zeros((unique_words, EMBEDDING_DIM))
for i, w in enumerate(words):
    if w in w2vmodel.wv:
        embedding_matrix[i] = w2vmodel[w]

tags = list(set(tag_list))
n_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# lenght of X is the no of sentences
X = [[word2idx[w[0]] for w in s] for s in sentence_list]


# pad all sentences with ENDPAD "O"
X = pad_sequences(maxlen=100, sequences=X, padding="post", value = unique_words - 1)

# labels for training and testing
Y = [[tag2idx[w[2]] for w in s] for s in sentence_list]

# pad labels whose indexes of all sentences equivalent of "O"
Y = pad_sequences(maxlen=100, sequences=Y, padding="post", value=tag2idx["O"])

from sklearn.model_selection import KFold

kf = KFold(n_splits=5,shuffle=False)

# for train_index, test_index in kf.split(X):
#
#     X_train = X[train_index]
#     X_train.tolist()
#     X_test = X[test_index]
#     X_test.tolist()
#     Y_train = Y[train_index]
#     Y_train.tolist()
#     Y_test = Y[test_index]
#     Y_test.tolist()

# code for kfold without without shuffling
data = []
for i in range(1,22+1):
    data.append(i)

data = np.asarray(data)

for train, test in kf.split(data):
    # print('train: %s, test: %s' % (data[train], data[test]))

    new_sentence_list_train = []
    for i in data[train]:
        new_sentence_list_train += sentence_file_dict[i]
    # print(len(new_sentence_list_train))

    X_train = [[word2idx[w[0]] for w in s] for s in new_sentence_list_train]
    X_train = pad_sequences(maxlen=100, sequences=X, padding="post", value=unique_words - 1)
    Y_train = [[tag2idx[w[2]] for w in s] for s in new_sentence_list_train]
    Y_train = pad_sequences(maxlen=100, sequences=Y, padding="post", value=tag2idx["O"])

    new_sentence_list_test = []
    for j in data[test]:
        new_sentence_list_test += sentence_file_dict[j]
    # print(len(new_sentence_list_test))

    X_test = [[word2idx[w[0]] for w in s] for s in new_sentence_list_test]
    X_test = pad_sequences(maxlen=100, sequences=X, padding="post", value=unique_words - 1)
    Y_test = [[tag2idx[w[2]] for w in s] for s in new_sentence_list_test]
    Y_test = pad_sequences(maxlen=100, sequences=Y, padding="post", value=tag2idx["O"])



    # converted to one hot vector
    Y_train  = [to_categorical(i, num_classes=n_tags) for i in Y_train]
    Y_test = [to_categorical(i, num_classes=n_tags) for i in Y_test]

    # split the data into training and testing
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    input = Input(shape=(100,))
    # model = Embedding(input_dim = unique_words, output_dim=EMBEDDING_DIM, input_length=100)(input)
    model = Embedding(input_dim = unique_words, output_dim=EMBEDDING_DIM, weights = [embedding_matrix], input_length=100)(input)
    # print(model)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
    model = Model(input, out)
    # plot_model(model, to_file = 'model.png', show_shapes=True)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # model.summary()
    history = model.fit(X_train, np.array(Y_train), batch_size=32, epochs=30, validation_split=0.1, verbose=1)

    # i = 0
    # p = model.predict(np.array([X_test[i]]))
    # p = np.argmax(p, axis=-1)
    # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
    # for w, pred in zip(X_test[i], p[0]):
    #     if words[w] != "ENDPAD":
    #         print("{:15}: {}".format(words[w], tags[pred]))

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()

    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

    test_pred = model.predict(X_test, verbose=1)
    idx2tag = {}
    idx2tag = {i: w for w, i in tag2idx.items()}

    pred_labels = pred2label(test_pred)
    test_labels = pred2label(Y_test)
    print(classification_report(test_labels, pred_labels))
    pred_label_list += pred_labels
    test_label_list += test_labels


print(classification_report(test_label_list, pred_label_list))

# with open('eval.csv', 'w') as fileObj:
#     wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
#     wr.writerow(['Actual', 'Predicted'])
#     for actual, predict in zip(test_label_list, pred_label_list):
#         row = [actual, predict]
#         wr.writerow(row)
