from bs4 import BeautifulSoup
import urllib
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

r = urllib.urlopen("http://www.goal.com/en-us/match/manchester-united-vs-afc-bournemouth/2242021/live-commentary?ICID=MP_MS_3 ").read()

soup = BeautifulSoup(r)
ofile  = open('ttest.csv', "wt")
writer = csv.writer(ofile,delimiter=',')
ul=  soup.find('ul',{'class' :'commentaries '})
for lis in ul.find_all('li'):
    div = lis.find('div')
    tagcontent = div.text
    tagcontent = tagcontent.rstrip('\n')
    tagcontent = tagcontent.strip()
    print tagcontent
    writer.writerow([tagcontent])
ofile.close()
