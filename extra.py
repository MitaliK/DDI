# #from sklearn.model_selection import train_test_split
#
# # maxlen = max([len(s) for s in sentence_list])
# # print('Maximum sequence length:', maxlen)
#
# # sentence lenght plot
# # plt.style.use("ggplot")
# # plt.hist([len(s) for s in sentence_list],bins=100)
# # plt.show()
#
#
# # for word in sentenceText.text:
# # with open('data.csv', 'w') as fileObj:
# #     wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
# #     wr.writerow(['filenumber', 'sentence_idx', 'word', 'pos', 'tag'])
# #     for row in csv_list:
# #         wr.writerow(row)
#
# # acc = history.history['acc']
# # val_acc = history.history['val_acc']
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']
#
# # epochs = range(1, len(acc) + 1)
# # plt.plot(epochs, acc, 'bo', label='Training acc')
# # plt.plot(epochs, val_acc, 'b', label='Validation acc')
# # plt.title('Training and validation accuracy')
# # plt.legend()
# # plt.figure()
#
# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.legend()
# # plt.show()
#
#
# # i = 0
# # p = model.predict(np.array([X_test[i]]))
# # p = np.argmax(p, axis=-1)
# # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
# # for w, pred in zip(X_test[i], p[0]):
# #     if words[w] != "ENDPAD":
# #         print("{:15}: {}".format(words[w], tags[pred]))
#
# # acc = history.history['acc']
# # val_acc = history.history['val_acc']
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']
#
# # epochs = range(1, len(acc) + 1)
# # plt.plot(epochs, acc, 'bo', label='Training acc')
# # plt.plot(epochs, val_acc, 'b', label='Validation acc')
# # plt.title('Training and validation accuracy')
# # plt.legend()
# # plt.figure()
#
# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.legend()
# # plt.show()
#
# # scikit-learn k-fold cross-validation
# import csv
#
# # data sample
# import numpy as np
# from sklearn.model_selection import KFold
#
# data = []
# for i in range(1, 22 + 1):
#     data.append(i)
#
# data = np.asarray(data)
# # prepare cross validation
# kf = KFold(n_splits=5, shuffle=False)
# # enumerate splits
# for train, test in kf.split(data):
#     print('train: %s, test: %s' % (data[train], data[test]))
#
# # a = [1, 2, 3, 4, 5]
# # b = [6, 7, 8, 9, 10]
# #
# # with open('eval.csv', 'w') as fileObj:
# #     wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
# #     wr.writerow(['Actual', 'Predicted'])
# #     for actual, predict in zip(a, b):
# #         row = [actual, predict]
# #         wr.writerow(row)
#
#     # split the data into training and testing
#     # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
