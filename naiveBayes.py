import os
import codecs
import unicodedata
from nltk.tag import StanfordNERTagger
import sys
import math
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from numpy  import array
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.preprocessing import DenseTransformer
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from pandas import DataFrame

class NaiveBayes():
    def __init__(self,modelType1="Logistic Regression", modelType2="Logistic Regression"):
        self.stopList = set(self.readFile('stopwords.txt'))
        os.environ[
            'CLASSPATH'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31' + os.pathsep + "C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31"
        os.environ[
            'STANFORD_MODELS'] = 'C:\Users\Rahul\Desktop\Soccer\Stanford_NER\stanford-ner-2016-10-31\classifiers' + os.pathsep + 'C:\Users\Rahul\Desktop\Soccer\Stanford_Parser\stanford-parser-full-2016-10-31\stanford-parser-3.7.0-models.jar'
        self.initializeModels(modelType1,modelType2)
        self.directory = "demoData"


    def initializeModels(self,modelType1, modelType2):
        if (modelType1 == "Gaussian"):
            self.model1 = GaussianNB()
        elif (modelType1 == "Bernoulli"):
            self.model1 = BernoulliNB()
        elif(modelType1 == "Multinomial"):
            self.model1 = MultinomialNB()
        else:
            self.model1 = linear_model.LogisticRegression(solver='sag', C=10)

        if (modelType2 == "Gaussian"):
            self.model2 = GaussianNB()
        elif (modelType2 == "Bernoulli"):
            self.model2 = BernoulliNB()
        elif(modelType2 == "Multinomial"):
            self.model2 = MultinomialNB()
        else:
            self.model2 = linear_model.LogisticRegression(solver='sag', C=10)

        self.count_vectorizer1 = CountVectorizer()
        self.count_vectorizer2 = CountVectorizer()

    def gatherData(self):
        fileList = []
        self.directory = "demoData\\trainingData"
        for file in os.listdir(self.directory):
            if file.endswith(".csv") and not (os.stat(self.directory + "\\" + file).st_size == 0):
                fileList.append(file)
        #To load saved entity
        commentaryList = []
        actionList = []
        data1 = DataFrame({'words': [], 'class1': [],'class2': [], 'commentary': [], 'minute': []})
        data2 = DataFrame({'words': [], 'class1': [], 'class2': [], 'commentary': [], 'minute': []})
        with open('objects//entities'+ '.pkl', 'rb') as fr:
            [removeDic,personDic,locationDic,organizationDic] = pickle.load(fr)
        for file in fileList:
                print file
                removeList = removeDic[file]
                dataFrame1, dataFrame2 = self.build_data_frame(file,removeList)
                data1 = data1.append(dataFrame1)
                data2 = data2.append(dataFrame2)
        print "Data1 Size : " , len(data1)
        print "Data2 Size : " , len(data2)
        with open('objects//dataFrame'+ '.pkl', 'wb') as f1:
            pickle.dump([data1,data2], f1, pickle.HIGHEST_PROTOCOL)

    def build_data_frame(self, file, removeList):
        """
        Builds data set into DataFrame format for 2 different training set.
        :param file:
        :param removeList:
        :return: DataFrame instances for two different training sets.
        """
        rows1 = []
        index1 = []
        rows2 = []
        index2 = []
        r = 0
        r1 = 0
        with codecs.open(self.directory + "\\" + file, 'r', encoding='utf8') as f:
            for row in f:
                splits = row.split(",")
                if(len(splits)>2):
                    class1 = "action"
                    minute = splits[0]
                    action = splits[1]
                    if minute != "":
                        # print minute
                        commentary = splits[2:]
                        if (len(commentary) > 0):
                            words = ''.join(commentary)
                            commentaryWords = ''.join(commentary)
                            words = self.segmentWords(words)
                            words = [word.lower() for word in words]
                            words = [x for x in words if x not in removeList]
                            words = self.filterStopWords(words)
                            words =" ".join(words)
                            class2 = action
                            if action.encode("utf-8") != "action":
                                class1 = "Not action"
                                rows2.append({'words': words, 'class1': class1,'class2': class2, 'commentary': commentaryWords, 'minute': minute})
                                if((file+minute) in index2):
                                    index2.append(file+minute+str(r))
                                    r+=1
                                else:
                                    index2.append(file + minute)
                            rows1.append({'words': words, 'class1': class1,'class2': class2, 'commentary': commentaryWords, 'minute': minute})
                            if ((file + minute) in index2):
                                index1.append(file + minute + str(r1))
                                r1 += 1
                            else:
                                index1.append(file + minute)
        data_frame1 = DataFrame(rows1, index=index1)
        data_frame2 = DataFrame(rows2, index=index2)
        return data_frame1, data_frame2




    def createModel(self):
        """
            Creates Model for testing data.
        """
        with open('objects//dataFrame'+'.pkl','rb') as fd:
            [data1, data2] = pickle.load(fd)
        counts1 = self.count_vectorizer1.fit_transform(data1['words'].values)
        self.model1.fit(counts1, data1['class1'].values)
        counts2 = self.count_vectorizer2.fit_transform(data2['words'].values)
        print "Counts1", len(self.count_vectorizer1.get_feature_names()), "Counts2", len(self.count_vectorizer2.get_feature_names())
        self.model2.fit(counts2, data2['class2'].values)
        with open('objects//model'+ '.pkl', 'wb') as fm:
            pickle.dump([self.model1,self.count_vectorizer1,self.model2,self.count_vectorizer2], fm, pickle.HIGHEST_PROTOCOL)

    def test(self, summaryAlgo = "LexRank"):
        """
            Runs the algorithm on test set data
        """
        with open("objects//model"+'.pkl','rb') as fm:
            [self.model1,self.count_vectorizer1,self.model2,self.count_vectorizer2] =pickle.load(fm)
        with open("objects//test_entities" + '.pkl', 'rb') as fe:
            lst = pickle.load(fe)
        removeDic = lst[0]
        self.directory = "demoData/testData"
        fileList= []
        label1 = array(['action','Not action'])
        label12 = array([u'action',u'yellow-card',u'substitution',u'assist',u'goal',u'penalty-goal',u'red-card',u'own-goal',u'missed-penalty',u'penalty-save',u'yellow-red'])
        confusion1 = np.array([[0 for x in range(len(label1))] for y in range(len(label1))])
        confusion12 = np.array([[0 for x in range(len(label12))] for y in range(len(label12))])
        scores1 = []
        scores12 = []
        if(summaryAlgo == "TextRank"):
            summarizer = TextRankSummarizer()
        else:
            summarizer = LexRankSummarizer()

        for file in os.listdir(self.directory):
            if file.endswith(".csv") and not (os.stat(self.directory + "\\" + file).st_size == 0):
                fileList.append(file)
        for file in fileList:
            removeList = removeDic[file]
            [data1,data2] =self.build_data_frame(file,removeList)
            testCount1 = self.count_vectorizer1.transform(data1['words'].values)
            predicted1 = self.model1.predict(testCount1)
            testy1 = data1['class1'].values
            data12 =  data1[predicted1 == "Not action"]
            testy12 = data1['class2'].values
            testCount12 = self.count_vectorizer2.transform(data12['words'].values)
            predicted12 = self.model2.predict(testCount12)
            score1 = f1_score(testy1, predicted1, pos_label="Not action")
            confusion1 += confusion_matrix(testy1, predicted1, labels=label1)
            scores1.append(score1)
            predicted1[predicted1 == "Not action"] = predicted12
            score12 = f1_score(testy12, predicted1, average='weighted')
            print score1, score12
            confusion12 += confusion_matrix(testy12, predicted1, labels=label12)
            scores12.append(score12)
            for index in data12.index:
                commentary = data12['commentary'][index]
                commentary = commentary[1:-1]
                minute = data12['minute'][index]
                parser = PlaintextParser.from_string(commentary, Tokenizer("english"))
                summary = summarizer(parser.document,1)
                print minute, [sentence for sentence in summary]
        print('Total commentary classified:', len(data1.commentary.values))
        print('Score1:', sum(scores1) / len(scores1))
        print('Confusion matrix1:')
        print(confusion1)
        print('Score2:', sum(scores12) / len(scores12))
        print('Confusion matrix2:')
        print(confusion12)


    def kfold(self, folds,classifier = 3):
        """
            Run k fold cross validation for different classifier.
            classifier =1 corressponds to "action - not just action" classifier
            classifier =2 corressponds to event type classifier
            classifier =3 corressponds to combined classifier
        """
        with open('objects//dataFrame' + '.pkl', 'rb') as fr:
            [data1,data2] = pickle.load(fr)

        if classifier==1:
            self.kfold1(data1,folds)
        elif classifier ==2:
            self.kfold2(data2,folds)
        else:
            self.kfold3(data1,folds)

    def kfold1(self,data,folds):
        """
            Claassifier 1
        :param self:
        :param data:
        :param folds:
        :return:
        """
        k_fold = KFold(n=len(data.words), n_folds=folds)
        scores = []
        classSet = data.class2.unique()
        print classSet
        confusion = np.array([[0 for x in range(len(classSet))] for y in range(len(classSet))])
        testResults = []
        predictedResults = []
        for train_indices, test_indices in k_fold:
            train_text = data.iloc[train_indices]['words'].values
            train_y = data.iloc[train_indices]['class1'].values
            test_text = data.iloc[test_indices]['words'].values
            test_y = data.iloc[test_indices]['class1'].values
            counts = self.count_vectorizer2.fit_transform(train_text)
            self.model1.fit(counts.toarray(), train_y)
            test_counts = self.count_vectorizer2.transform(test_text)
            predictions = self.model1.predict(test_counts.toarray())
            confusion += confusion_matrix(test_y, predictions, labels=classSet)
            score = f1_score(test_y, predictions, average='weighted')
            scores.append(score)
            testResults.extend(test_y)
            predictedResults.extend(predictions)
        print('Total commentary classified:', len(data.words))
        print('Score:', sum(scores) / len(scores))
        print('Confusion matrix:')
        print(confusion)
        print metrics.classification_report(testResults, predictedResults,
                                            target_names=classSet)

    def kfold2(self,data,folds):
        k_fold = KFold(n=len(data.words), n_folds=folds)
        scores = []
        classSet = data.class2.unique()
        print classSet
        confusion = np.array([[0 for x in range(len(classSet))] for y in range(len(classSet))])
        testResults=[]
        predictedResults =[]
        for train_indices, test_indices in k_fold:
            train_text = data.iloc[train_indices]['words'].values
            train_y = data.iloc[train_indices]['class2'].values
            test_text = data.iloc[test_indices]['words'].values
            test_y = data.iloc[test_indices]['class2'].values
            counts = self.count_vectorizer2.fit_transform(train_text)
            self.model1.fit(counts.toarray(), train_y)
            test_counts = self.count_vectorizer2.transform(test_text)
            predictions = self.model1.predict(test_counts.toarray())
            confusion += confusion_matrix(test_y, predictions,labels= classSet)
            score = f1_score(test_y, predictions,average='weighted')
            scores.append(score)
            testResults.extend(test_y)
            predictedResults.extend(predictions)
        print('Total commentary classified:', len(data.words))
        print('Score:', sum(scores) / len(scores))
        print('Confusion matrix:')
        print(confusion)
        print metrics.classification_report(testResults, predictedResults,
                                            target_names=classSet)

    def kfold3(self,data1,folds):
        k_fold = KFold(n=len(data1.words), n_folds=folds)
        scores1 = []
        scores2 = []
        class1Set = data1.class1.unique()
        class2Set = data1.class2.unique()
        confusion1 = np.array([[0 for x in range(len(class1Set))] for y in range(len(class1Set))])
        confusion2 = np.array([[0 for x in range(len(class2Set))] for y in range(len(class2Set))])
        testResults1 = []
        predictedResults1 = []
        testResults2 = []
        predictedResults2 = []
        for train_indices, test_indices in k_fold:
            train_text = data1.iloc[train_indices]['words'].values
            train_y1 = data1.iloc[train_indices]['class1'].values
            train_y2 = data1.iloc[train_indices]['class2'].values
            test_text = data1.iloc[test_indices]['words'].values
            test_y1 = data1.iloc[test_indices]['class1'].values
            test_y2 = data1.iloc[test_indices]['class2'].values
            counts = self.count_vectorizer1.fit_transform(train_text)
            self.model1.fit(counts.toarray(), train_y1)
            self.model2.fit(counts.toarray(), train_y2)
            test_counts = self.count_vectorizer1.transform(test_text)
            predictions1 = self.model1.predict(test_counts.toarray())
            predictions2 = self.model2.predict(test_counts.toarray())
            confusion1 += confusion_matrix(test_y1, predictions1, labels=class1Set)
            confusion2 += confusion_matrix(test_y2, predictions2, labels=class2Set)
            score1 = f1_score(test_y1, predictions1, pos_label="Not action".decode('utf-8'))
            score2 = f1_score(test_y2, predictions2,average='weighted')
            scores1.append(score1)
            scores2.append(score2)
            testResults1.extend(test_y1)
            predictedResults1.extend(predictions1)
            testResults2.extend(test_y2)
            predictedResults2.extend(predictions2)

        print('Total commentary classified:', len(data1.commentary.values))
        print('Score1:', sum(scores1) / len(scores1))
        print('Confusion matrix1:')
        print(confusion1)
        print metrics.classification_report(testResults1, predictedResults1,
                                            target_names=class1Set)
        print('Score2:', sum(scores2) / len(scores2))
        print('Confusion matrix2:')
        print(confusion2)
        print metrics.classification_report(testResults2, predictedResults2,
                                            target_names=class2Set)


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


# if __name__ == "__main__":
#     object = NaiveBayes()
#     #Training
#     # object.preprocess()
#     # object.gatherData()
#     # object.gatherDataActionNotAction()
#     # object.createModel()
#     #Testing
#     object.test()
    # object.kfold()