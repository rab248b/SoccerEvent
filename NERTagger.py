import os
import codecs
import unicodedata
from nltk.tag import StanfordNERTagger
import sys
import math
import pickle
class NERTagger():
    def __init__(self):
        os.environ[
            'CLASSPATH'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31' + os.pathsep + "C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31"
        os.environ[
            'STANFORD_MODELS'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31\classifiers' + os.pathsep + 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31\stanford-parser-3.7.0-models.jar'

    def processNewData(self):
        directory = "demoData/newData"
        fileList = []
        with open('entities'+ '.pkl', 'rb') as fr:
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
                        action = splits[1]
                        if minute != "":
                            # print minute, action
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                # print nertags
                                # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                #
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

                # print removeList
        # Training step 1
        print "Final size", len(removeDic)
        with open('entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def processTestData(self):
        directory = "demoData/testData"
        fileList = []
        removeDic = {}
        personDic = {}
        locationDic = {}
        organizationDic = {}
        # with open('entities'+ '.pkl', 'rb') as fr:
        #     [removeDic, personDic, locationDic, organizationDic] = pickle.load(fr)
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
                        action = splits[1]
                        if minute != "":
                            # print minute, action
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                # print nertags
                                # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                #
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

                # print removeList
        # Training step 1
        print "Final size", len(removeDic)
        with open('objects//test_entities' + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def processEntireData(self,entityName,directory):

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
                            # print minute, action
                            commentary = splits[2:]
                            if (len(commentary) > 0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                # print nertags
                                # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                #
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
                # print removeList
        # Training step 1
        with open('objects//'+entityName + '.pkl', 'wb') as f1:
            pickle.dump([removeDic,personDic,locationDic,organizationDic], f1, pickle.HIGHEST_PROTOCOL)

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()
    def combineData(self):
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



if __name__ == "__main__":
    object = NERTagger()
    # object.processNewData()
    # object.processTestData()
    # directory = "demoData/223"
    # object.processEntireData('223',directory)
    # directory = "demoData/22418"
    # object.processEntireData('22418', directory)
    # directory = "demoData/225"
    # object.processEntireData('225',directory)
    # directory = "demoData/22830"
    # object.processEntireData('22830', directory)
    # directory = "demoData/228"
    # object.processEntireData('228',directory)
    # directory = "demoData/236+"
    # object.processEntireData('236+',directory)
    # directory = "demoData/224201"
    # object.processEntireData('224201',directory)
    # directory = "demoData/22420"
    # object.processEntireData('22420',directory)
    # directory = "demoData/230"
    # object.processEntireData('230',directory)
    object.combineData()