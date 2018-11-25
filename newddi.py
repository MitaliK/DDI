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
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path



def build_model(path, dataset_number):
    csv_list = []
    file_counter = 0
    sentence_list = []
    word_list = []
    tag_list = []
    sentence_all = []  #training data for word2vec
    EMBEDDING_DIM = 100
    train_test = ''


    for filename in os.listdir(path):
        if not filename.endswith('.xml'): 
            continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        file_counter += 1
        root = tree.getroot()
        for child in root:
            if child.tag == 'Sentences':
                sentences = child
                break

        train_test = 'train'

        for sentence_ind, sentence in enumerate(sentences.findall('Sentence')):
            sentenceText = sentence.find('SentenceText').text
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
                csv_list.append([file_counter, sentence_ind, word, pos, tag, train_test])

                word_list.append(word)
                tag_list.append(tag)
                sentence_each.append(word)
                temp_list.append((word, pos, tag, train_test))
            sentence_list.append(temp_list)
            sentence_all.append(sentence_each)

    # loop for files within test folder
    path = path + '/test'
    file_counter = 0
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): 
            continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        file_counter += 1
        root = tree.getroot()
        for child in root:
            if child.tag == 'Sentences':
                sentences = child
                break

        train_test = 'test'


        for sentence_ind, sentence in enumerate(sentences.findall('Sentence')):
            sentenceText = sentence.find('SentenceText').text
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
                csv_list.append([file_counter, sentence_ind, word, pos, tag, train_test])

                word_list.append(word)
                tag_list.append(tag)
                sentence_each.append(word)
                temp_list.append((word, pos, tag, train_test))
            sentence_list.append(temp_list)
            sentence_all.append(sentence_each)

        # for word in sentenceText.text:
    with open('data.csv', 'a') as fileObj:
        wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
        wr.writerow(['dataset_number', 'sentence_idx', 'word', 'pos', 'tag'])
        for row in csv_list:
            wr.writerow(row)

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

    # converted to one hot vector
    Y = [to_categorical(i, num_classes=n_tags) for i in Y]

    input = Input(shape=(100,))
    # model = Embedding(input_dim = unique_words, output_dim=EMBEDDING_DIM, input_length=100)(input)
    model = Embedding(input_dim = unique_words, output_dim=EMBEDDING_DIM, weights = [embedding_matrix], input_length=100)(input)
    print(model)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
    model = Model(input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # model.summary()
    history = model.fit(X, np.array(Y), batch_size=32, epochs=30, validation_split=0.1, verbose=1)

    test_pred = model.predict(X_test, verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    #print(test_pred)
    return test_pred, idx2tag



def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

if __name__ == '__main__':
    mypath = Path().absolute()
    path = str(mypath) + '/trainingFiles'
    path_list = next(os.walk(path))[1]
    pred_labels = []
    test_labels = []

    for ever_dir in path_list:
        new_path = path + '/' +ever_dir
        test_pred, idx2tag = build_model(new_path, ever_dir)
        pred_labels = pred2label(test_pred, idx2tag)
        test_labels = pred2label(Y_test, idx2tag)
        print(classification_report(test_labels, pred_labels))
