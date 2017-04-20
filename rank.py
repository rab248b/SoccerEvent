import os
import codecs
import unicodedata
from nltk.tag import StanfordNERTagger
import sys
import math
import pickle
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

class Rank():
    def __init__(self):
        self.stopList = set(self.readFile('stopwords.txt'))
        os.environ[
            'CLASSPATH'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31' + os.pathsep + "C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31"
        os.environ[
            'STANFORD_MODELS'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31\classifiers' + os.pathsep + 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31\stanford-parser-3.7.0-models.jar'

    #Get all the named Entity for each file and save it in a "entities.pkl" object
    def preprocess(self):
        directory = "demoData"
        fileList = []
        removeDic = {}
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        cnt = 0
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                cnt+=1
                print file,cnt
                removeList = set()
                for row in f:
                    splits = row.split(",")
                    if(len(splits)>2):
                        minute = splits[0]
                        action = splits[1]
                        if minute != "":
                            # print minute, action
                            commentary = splits[2:]
                            if(len(commentary)>0):
                                words = ''.join(commentary)
                                words = self.segmentWords(words)
                                nertags = st.tag(words)
                                # print nertags
                                # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                #
                                for nertag in nertags:
                                    if nertag[1] == "PERSON" or nertag[1] == "ORGANIZATION" or nertag[1] == "LOCATION":
                                        removeList.add(nertag[0].lower())
                removeDic[file] = removeList
                # print removeList
        #Training step 1
        with open('entities'+ '.pkl', 'wb') as f1:
            pickle.dump(removeDic, f1, pickle.HIGHEST_PROTOCOL)



    #Calculate probabilites for commentaries in training set. Removed named entities using "entities.pkl".
    def train(self):
        fileList = []
        directory = "demoData"
        removeDic = {}
        #total words in training set
        totalWords = 0
        #Count of number of words in each class
        classCountDic = {}
        #Prior probability of each class
        classProbDic = {}
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        vocab = {}
        # Contains key as classes and values as another dictionary with words and their counts
        classDic = {}
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        #To load saved entity
        with open('entities'+ '.pkl', 'rb') as fr:
            removeDic = pickle.load(fr)
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                print file
                commentaryList = []
                removeList = set()
                for row in f:
                    splits = row.split(",")
                    if(len(splits)>2):
                        minute = splits[0]
                        action = splits[1]
                        if minute != "":
                            if action.encode("utf-8")!="action":
                                commentary = splits[2:]
                                if action not in classDic:
                                    classDic[action] = {}
                                    classCountDic[action] = 0
                                if(len(commentary)>0):
                                    words = ''.join(commentary)
                                    words = self.segmentWords(words)
                                    # nertags = st.tag(words)
                                    # print nertags
                                    # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                    #
                                    # for nertag in nertags:
                                    #     if nertag[1] == "PERSON" or nertag[1] == "ORGANIZATION" or nertag[1] == "LOCATION":
                                    #         removeList.add(nertag[0].lower())
                                    words = [word.lower() for word in words]
                                    words = self.filterStopWords(words)
                                    commentaryList.append((words,action))
                removeList = removeDic[file]
                # removeDic[file] = removeList
                print removeList
        #Training step 1
        # with open('entities'+ '.pkl', 'wb') as f1:
        #     pickle.dump(removeDic, f1, pickle.HIGHEST_PROTOCOL)
        # Training step 2
                tempCount = self.createDictionaries(commentaryList,vocab,classDic,removeList,classCountDic)
                totalWords = totalWords + tempCount

        probDic, unknownProbDic = self.calculateProbabilities(vocab,classDic,classCountDic)
        classProbDic = dict.fromkeys(classCountDic.keys())
        for clas in classCountDic:
            classProbDic[clas] = float(classCountDic[clas])/totalWords
        probList = [probDic, unknownProbDic,classProbDic]
        print classProbDic
        with open('probabilites'+ '.pkl', 'wb') as f1:
            pickle.dump(probList, f1, pickle.HIGHEST_PROTOCOL)




    def createDictionaries(self,commentaryList,vocab, classDic,removeList, classCountDic):
        count = 0
        for tuple in commentaryList:
            action = tuple[1]
            words = tuple[0]
            # print action, words
            localCount = 0
            for word in words:
                if word not in removeList:
                    localCount=+1
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        temp = vocab[word]
                        vocab[word] = temp + 1
                    if word not in classDic[action]:
                        classDic[action][word] = 1
                    else:
                        temp = classDic[action][word]
                        classDic[action][word] = temp + 1
            temp = classCountDic[action]
            classCountDic[action] = localCount + temp
            count = localCount + count
        # print vocab
        # print classDic
        return count

    def calculateProbabilities(self,vocab, classDic, classCountDic):
        probDic = dict.fromkeys(classDic.keys(),{})
        unknownProbDic = dict.fromkeys(classDic.keys(), {})
        # print probDic.keys()
        for action in classDic:
            for word in classDic[action]:
                probDic[action][word] = float(classDic[action][word]+1)/(len(vocab)+classCountDic[action]+1)
            unknownProbDic[action] =float(1)/(len(vocab)+classCountDic[action]+1)
        return probDic,unknownProbDic

    def testData(self):
        fileList = []
        directory = "demoData/test"

        with open('probabilites'+ '.pkl', 'rb') as fr:
            probList = pickle.load(fr)
        probDic = probList[0]
        unknownProbDic = probList[1]
        classProbDic = probList[2]
        print classProbDic
        print probDic
        print unknownProbDic
        for file in os.listdir(directory):
            if file.endswith(".csv") and not (os.stat(directory + "\\" + file).st_size == 0):
                fileList.append(file)
        st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', encoding="utf-8")
        for file in fileList:
            with codecs.open(directory + "\\" + file, 'r', encoding='utf8') as f:
                print file
                commentaryList = []
                removeList = set()
                for row in f:
                    splits = row.split(",")
                    if(len(splits)>2):
                        minute = splits[0]
                        action = splits[1]
                        if minute != "":
                            # print minute, action
                            if action.encode('utf-8') !="action":
                                commentary = splits[2:]
                                # if action not in classDic:
                                    # classDic[action] = {}
                                if(len(commentary)>0):
                                    words = ''.join(commentary)
                                    words = self.segmentWords(words)
                                    nertags = st.tag(words)
                                    print nertags
                                    # # List of indices of PERSON, LOCATION and ORGANIZATION to be removed.
                                    for nertag in nertags:
                                        if nertag[1] == "PERSON" or nertag[1] == "ORGANIZATION" or nertag[1] == "LOCATION":
                                            removeList.add(nertag[0].lower())
                                    words = [word.lower() for word in words]
                                    words = self.filterStopWords(words)
                                    commentaryList.append((words, action))
                classDic = dict.fromkeys(classProbDic.keys(),0)

                for tuple in commentaryList:
                    words = tuple[0]
                    action = tuple[1]
                    for clas in classDic:
                        # classDic[clas] = math.log(classProbDic[clas])
                        classDic[clas] = 0
                    for word in words:
                        if word not in removeList:
                            for clas in classDic:
                                if word not in probDic[clas]:
                                    tempProb = classDic[clas]
                                    classDic[clas] = tempProb + math.log(unknownProbDic[clas])
                                else:
                                    tempProb = classDic[clas]
                                    classDic[clas] = tempProb + math.log(probDic[clas][word])
                    print classDic
                    predictedAction = max(classDic, key=classDic.get)
                    print action, predictedAction





    def filterStopWords(self, words):
        """Filters stop words."""
        tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                            if unicodedata.category(unichr(i)).startswith('P'))
        filtered = []
        for word in words:
            word = word.translate(tbl)
            # print word
            if not word.encode("utf-8") in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered

    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def sumyTest(self):
        text = '"Substitution sub-out Wilfried Zaha sub-in James McArthur . Palace\'s second change sees the arrival of sub-in James McArthur, with Zaha surprisingly giving way."'
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        print summarizer(parser.document,1)


if __name__ == "__main__":
    object = Rank()
    #Training
    # object.preprocess()
    # object.train()
    #Testing
    # object.testData()
    object.sumyTest()
