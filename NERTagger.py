import os
import codecs
import unicodedata
from nltk.tag import StanfordNERTagger
import sys
import math
import pickle
class NERTagger():
    def __init__(self,directory):
        os.environ[
            'CLASSPATH'] =  directory+"\Stanford_NER\stanford-ner-2016-10-31"
        os.environ[
            'STANFORD_MODELS'] = directory+'\Stanford_NER\stanford-ner-2016-10-31\classifiers'


    def processNewData(self):
        """
            Process NER Tagging for new data to be added for Training. It updates the existing entities.pkl object and add the new data to that.
        """
        directory = "demoData/newData"
        fileList = []
        with open('objects//entities'+ '.pkl', 'rb') as fr:
            [removeDic, personDic, locationDic, organizationDic] = pickle.load(fr)
        print "Initial size", len(removeDic)
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        cnt = 0
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                cnt += 1
                print file, cnt
                removeList = set()
                personList = set()
                organizationList = set()
                locationList = set()
                for row in f:
                    splits = row.split(",")
                    if (len(splits) > 2):
                        minute = splits[0]
                        if minute != "":
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                for nertag in nertags:
                                    if nertag[1] == "PERSON":
                                        removeList.add(nertag[0].lower())
                                        personList.add(nertag[0].lower())
                                    elif nertag[1] == "ORGANIZATION":
                                        removeList.add(nertag[0].lower())
                                        organizationList.add(nertag[0].lower())
                                    elif nertag[1] == "LOCATION":
                                        removeList.add(nertag[0].lower())
                                        locationList.add(nertag[0].lower())
                removeDic[file] = removeList
                personDic[file] = personList
                locationDic[file] = locationList
                organizationDic[file] = organizationList
        print "Final size", len(removeDic)
        with open('objects//entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def processTestData(self):
        """
              Process NER Tagging for testing data. It overwrites the existing test_entities.pkl object with the new data.
        """
        directory = "demoData/testData"
        fileList = []
        removeDic = {}
        personDic = {}
        locationDic = {}
        organizationDic = {}
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        cnt = 0
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                cnt += 1
                print file, cnt
                removeList = set()
                personList = set()
                organizationList = set()
                locationList = set()
                for row in f:
                    splits = row.split(",")
                    if (len(splits) > 2):
                        minute = splits[0]
                        if minute != "":
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                for nertag in nertags:
                                    if nertag[1] == "PERSON":
                                        removeList.add(nertag[0].lower())
                                        personList.add(nertag[0].lower())
                                    elif nertag[1] == "ORGANIZATION":
                                        removeList.add(nertag[0].lower())
                                        organizationList.add(nertag[0].lower())
                                    elif nertag[1] == "LOCATION":
                                        removeList.add(nertag[0].lower())
                                        locationList.add(nertag[0].lower())
                removeDic[file] = removeList
                personDic[file] = personList
                locationDic[file] = locationList
                organizationDic[file] = organizationList
        print "Final size", len(removeDic)
        with open('objects//test_entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def processEntireData(self,directory):
        """
                Process NER Tagging for entire training data. It overwrites the existing entities.pkl object with the new data for training.
        """
        fileList = []
        removeDic = {}
        personDic = {}
        locationDic = {}
        organizationDic = {}
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        cnt = 0
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                cnt += 1
                print file, cnt
                removeList = set()
                personList = set()
                organizationList = set()
                locationList = set()
                for row in f:
                    splits = row.split(",")
                    if (len(splits) > 2):
                        minute = splits[0]
                        action = splits[1]
                        if minute != "":
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                for nertag in nertags:
                                    if nertag[1] == "PERSON":
                                        removeList.add(nertag[0].lower())
                                        personList.add(nertag[0].lower())
                                    elif nertag[1] == "ORGANIZATION":
                                        removeList.add(nertag[0].lower())
                                        organizationList.add(nertag[0].lower())
                                    elif nertag[1] == "LOCATION":
                                        removeList.add(nertag[0].lower())
                                        locationList.add(nertag[0].lower())
                                    removeDic[file] = removeList
                                    personDic[file] = personList
                                    locationDic[file] = locationList
                                    organizationDic[file] = organizationList
        with open('objects//entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def combineData(self):
        """
            Combines the training data with other training objects.
        """
        with open('objects//entities'+ '.pkl', 'rb') as fr:
            [removeDic, personDic, locationDic, organizationDic] = pickle.load(fr)
        with open('objects//22419'+ '.pkl', 'rb') as fr:
            [removeDic1, personDic1, locationDic1, organizationDic1] = pickle.load(fr)
        for key in removeDic1:
            removeDic[key] = removeDic1[key]
            locationDic[key] = locationDic1[key]
            personDic[key] = personDic1[key]
            organizationDic[key] = organizationDic1[key]

        with open('objects//entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)



# if __name__ == "__main__":
#     object = NERTagger()
#     object.combineData()