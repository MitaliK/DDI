# maxlen = max([len(s) for s in sentence_list])
# print('Maximum sequence length:', maxlen)

# sentence lenght plot
# plt.style.use("ggplot")
# plt.hist([len(s) for s in sentence_list],bins=100)
# plt.show()



    # for word in sentenceText.text:
# with open('data.csv', 'w') as fileObj:
#     wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
#     wr.writerow(['filenumber', 'sentence_idx', 'word', 'pos', 'tag'])
#     for row in csv_list:
#         wr.writerow(row)

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

# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))