from NERTagger import NERTagger
from naiveBayes import NaiveBayes
import scrapping_New
import datetime
import os
import sys
import getopt

class Run():
    def __init__(self):
        self.testingFlag = True

    def scrapData(self, date):
        self.validate(date)
        scrapping_New.scrap(date)
        sourceDirectory = "matchData"
        if self.testingFlag == True:
            targetDirectory = "demoData//testData"
        else:
            targetDirectory = "demoData//newData"
        self.transferData(sourceDirectory,targetDirectory)

    def validate(self,date_text):
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    def runTagger(self):
        tagObject = NERTagger(os.getcwd())
        if(self.testingFlag==True):
            tagObject.processTestData()
        else:
            tagObject.processNewData()
            self.transferData(sourceDirectory="demoData/newData", targetDirectory="demoData/trainingData")

    def test(self):
        classifierObject = NaiveBayes()
        classifierObject.test()

    def trainTest(self, folds = 6,classifier =3,model1 = "Logistic Regression",model2 ="Logisitc Regression"):
        classifierObject = NaiveBayes(model1,model2)
        # classifierObject.gatherData()
        classifierObject.kfold(folds,classifier)
        classifierObject.createModel()

    def transferData(self,sourceDirectory, targetDirectory):
        # sourceDirectory = "matchData"
        fileList = []
        targetFileList = []
        # if self.testingFlag == True:
        #     targetDirectory = "demoData//testData"
        # else:
        #     targetDirectory = "demoData//newData"
        for file in os.listdir(sourceDirectory):
            if file.endswith(".csv") and not (os.stat(sourceDirectory + "\\" + file).st_size < 4000):
                fileList.append(file)
        for file in os.listdir(targetDirectory):
            if file.endswith(".csv") and not (os.stat(targetDirectory + "\\" + file).st_size < 4000):
                targetFileList.append(file)
        abs_path =  os.getcwd()
        print os.getcwd()
        for file in fileList:
            if file not in targetFileList:
                os.rename(abs_path+"\\"+sourceDirectory+"\\"+file, abs_path+"\\"+targetDirectory+"\\"+file)

if __name__ == "__main__":
    object = Run()
    if (len(sys.argv)==1):
        print "Invalid Arguments"
    folds = None
    classifier = None
    date = None
    if(len(sys.argv)>=1):
        (options,remainder) = getopt.getopt(sys.argv[1:],'td:f:c:m1:m2:',["date=","folds=","classifier=","model1=","model2="])
        for o,a in options:
            print o,a
            if o in ("-t",""):
                object.testingFlag = False
            if o in ("-d","date="):
                date = a
            if o in ("-f","folds="):
                folds = int(a)
            if o in ("-c","classifier="):
                classifier = int(a)
            if o in ("m1","model1="):
                model1 = a
            if o in ("m2","model2="):
                model2 = a
    if(object.testingFlag == True):
        print "Inside Testing"
        if(date != None):
            print "Scrapping"
            object.scrapData(date)
            object.runTagger()
        print "Testing"
        object.test()
    else:
        print "Inside Training"
        if(date!=None):
            print "Scrapping"
            # object.scrapData(date)
            # object.runTagger()
        if(folds != None):
            if(classifier!=None):
                print "Training with folds and classifier"
                object.trainTest(folds,classifier)
            else:
                print "Training with folds"
                object.trainTest(folds)
        else:
            print "Training with classifier"
            if(classifier!=None):
                object.trainTest(classifier)
            else:
                print "Training with"
                object.trainTest()






    # object.scrapData("2017-04-15")
    # object.transferData()
    # object.runTagger()
    # object.test()