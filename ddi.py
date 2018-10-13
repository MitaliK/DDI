import xml
import xml.etree.ElementTree as ET
import nltk
import csv
tree = ET.parse('/Users/darshan/Documents/github/DDI/trainingFiles/ADCIRCA_ff61b237-be8e-461b-8114-78c52a8ad0ae.xml')
root = tree.getroot()
for child in root:
    if child.tag == 'Sentences':
        sentences = child
        break
csv_list = []
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
    for word_tag in nltk.pos_tag(nltk.word_tokenize(sentenceText)):
        tag = 'O'
        word = word_tag[0]
        pos = word_tag[1]
        if word in tag_dict:
            tag = tag_dict[word]
        csv_list.append([pos,sentence_ind,word,tag])

    #for word in sentenceText.text:
with open('data.csv','wb') as fileObj:
    wr = csv.writer(fileObj, quoting=csv.QUOTE_ALL)
    wr.writerow(['pos','sentence_idx','word','tag'])
    for row in csv_list:
        wr.writerow(row)
a = 10