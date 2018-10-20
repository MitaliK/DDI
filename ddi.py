import xml
import xml.etree.ElementTree as ET
import nltk
import csv
import os
import numpy as np
# Check how long sentences are so that we can pad them
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

csv_list = []
file_counter = 0
sentence_list = []
word_list =  []
tag_list = []

path = '/Users/mitalikulkarni/Documents/Sem_III/BMI 598/Project Document/DDI Data/trainingFiles/'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
    file_counter += 1
    root = tree.getroot()
    for child in root:
        if child.tag == 'Sentences':
            sentences = child
            break

    for sentence_ind,sentence in enumerate(sentences.findall('Sentence')):
        sentenceText = sentence.find('SentenceText').text
        # print sentenceText
        # print nltk.pos_tag(nltk.word_tokenize(sentenceText))
        tag_dict ={}
        for mention in sentence.findall('Mention'):
            for mention_words in mention.attrib['str'].split('|'):
                for ind,mention_word in enumerate(mention_words.split()):
                    if not ind:
                        tag_dict[mention_word] ='B-'+mention.attrib['type']
                    else:
                        tag_dict[mention_word] ='I-'+mention.attrib['type']
        # print tag_dict

        temp_list = []
        for word_tag in nltk.pos_tag(nltk.word_tokenize(sentenceText)):
            tag = 'O'
            word = word_tag[0]
            pos = word_tag[1]
            if word in tag_dict:
                tag = tag_dict[word]
            csv_list.append([file_counter,sentence_ind, word, pos, tag])
            word_list.append(word)
            tag_list.append(tag)
            temp_list.append((word, pos, tag))
        sentence_list.append(temp_list)

    #for word in sentenceText.text:
with open('data.csv','w') as fileObj:
    wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
    wr.writerow(['filenumber','sentence_idx','word', 'pos', 'tag'])
    for row in csv_list:
        wr.writerow(row)

maxlen = max([len(s) for s in sentence_list])
print ('Maximum sequence length:', maxlen)

# sentence plot
# plt.style.use("ggplot")
# plt.hist([len(s) for s in sentence_list],bins=100)
# plt.show()

words = list(set(word_list))
words.append("ENDPAD")
n_words = len(words)

tags = list(set(tag_list))
n_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X = [[word2idx[w[0]] for w in s] for s in sentence_list]
#pad with ENDPAD
X = pad_sequences(maxlen=100, sequences=X, padding="post",value=n_words - 1)


Y = [[tag2idx[w[2]] for w in s] for s in sentence_list]
#pad with 2 which is index of capital o
Y = pad_sequences(maxlen=100, sequences=Y, padding="post", value=tag2idx["O"])
#converted to one hot vector
Y = [to_categorical(i, num_classes=n_tags) for i in Y]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

input = Input(shape=(100,))
model = Embedding(input_dim=n_words, output_dim=100, input_length=100)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)# variational biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, np.array(Y_train), batch_size=32, epochs=25, validation_split=0.1, verbose=1)

i = 0
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w,pred in zip(X_test[i],p[0]):
    print("{:15}: {}".format(words[w],tags[pred]))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()