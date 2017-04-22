import urllib2
import time
from bs4 import BeautifulSoup
import sys
import re
import concurrent.futures
from HTMLParser import HTMLParser
# from pymongo import MongoClient
# mongo_client = MongoClient()
# db = mongo_client.football_commentary
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import csv
previousContent=""
count=0

def scrap_commentary(match_link):
	link_split = match_link.split('/')
	match_id = link_split[len(link_split) -1]
	page=urllib2.urlopen('http://www.goal.com' + match_link+'/live-commentary')
	page_text = page.read()
	new_text = re.sub('<br>', '. ', page_text)
	soup=BeautifulSoup(new_text, "html.parser")
	i=0
	global count
	print match_link
	global previousContent
	fullContent = ''
	ofile = open('matchData/'+match_id+'.csv', "wt")
	writer = csv.writer(ofile)
	ul = soup.find('ul', {'class': 'commentaries '})
	tags = ul.find_all('li')
	# print tags
	h = HTMLParser()
	if len(tags) > 0:
		for tag in tags:
			div = tag.find('div')
			event = tag["data-event-type"]
			minute = div.find('div',{'class':'time'}).text
			divContent= div.find('div', {'class': 'text'})
			#To check if there is <br> tag
			brtags = div.find('div', {'class': 'text'}).findAll('br')
			if len(brtags) >0:
				tagcontent = brtags[0].previousSibling
				print "Previous BR tag",tagcontent
				for brtag in brtags:
					brContent =  brtag.text
					print brContent
					tagcontent = tagcontent+ "; " + brContent
			else:
				tagcontent = div.find('div',{'class':'text'}).text
			if event == 'substitution':
				spans =div.find('div',{'class':'text'}).findAll('span')
				tagcontent = tagcontent.replace('Substitution', 'Substitution ')
				tagcontent = tagcontent.replace('\n', ' ')
				for span in spans:
					tagcontent = tagcontent.replace(span.text,''.join(span['class']).encode('utf-8')+' '+span.text)

			tagcontent = tagcontent.rstrip('\n')
			tagcontent = tagcontent.strip()
			tagcontent = h.unescape(tagcontent)
			minute =minute.rstrip("\n\r")[1:-1]
			if(bool(re.search(r'\d', minute))):
				print minute
				writer.writerow([minute+"'",event.encode('utf-8') ,tagcontent])
			else:
				writer.writerow(["", event.encode('utf-8') , tagcontent])
			fullContent = fullContent+tagcontent
			if(tagcontent):
				print(tagcontent)
		# db.goal_commentary.update({'match_id':match_id}, {"$set" : {'commentary' : fullContent}}, upsert=True)
		print ('inserted')
	ofile.close()
	try:
		lineUpPage = urllib2.urlopen('http://www.goal.com' + match_link+'/lineups')
		soup = BeautifulSoup(lineUpPage, "html.parser")
		lineupfile = open('matchData/lineups/' + match_id + 'lineup.csv', "wt")
		writerLineup = csv.writer(lineupfile)
		div = soup.find('div',{'class':'main-content lineups'})
		homeTeam = div.find('h2',{'class':'home'}).text
		awayTeam = div.find('h2',{'class':'away'}).text
		writerLineup.writerow([homeTeam,awayTeam])
		writerLineup.writerow(["Starting XI"])
		start11Div = div.find('div',{'class':'players'})
		homeLis = start11Div.find('div',{'class':'home'}).find('ul').find_all('li',{'data-side':'home'})
		awayLis = start11Div.find('div', {'class':'away'}).find('ul').find_all('li',{'data-side':'away'})
		homecolumn = []
		awaycolumn = []
		for homeli in homeLis:
			playerNumber = homeli['data-number']
			playerName = homeli.find('a').find('span',{'class':'name'}).text
			eventLis = homeli.find('a').find('ul',{'class':'events'}).find_all('li')
			events = []
			if len(eventLis) > 0:
				for eventLi in eventLis:
					eventAction = eventLi['class']
					eventTime = str(eventLi.text)
					print eventTime
					events.append((eventAction[0].encode('utf-8'),eventTime))
			tempLst = [playerNumber,playerName,events]
			homecolumn.append(tempLst)
			# writerLineup.writerow([playerNumber,playerName,str(events)[1:-1]])

		for awayLi in awayLis:
			playerNumber = awayLi['data-number']
			playerName = awayLi.find('a').find('span',{'class':'name'}).text
			eventLis = awayLi.find('a').find('ul',{'class':'events'}).find_all('li')
			events = []
			if len(eventLis)>0:
				for eventLi in eventLis:
					eventAction = eventLi['class']
					eventTime = str(eventLi.text)
					events.append((eventAction[0].encode('utf-8'),eventTime))
			tempLst = [playerNumber, playerName, events]
			awaycolumn.append(tempLst)
			# writerLineup.writerow([playerNumber,playerName,str(events)[1:-1]])
		for i in range(len(awaycolumn)):
			print homecolumn[i],awaycolumn[i]
			writerLineup.writerow(homecolumn[i]+awaycolumn[i])

		# time.sleep(120)
		lineupfile.close()
		print "Line up inserted"
	except:
		e = sys.exc_info()[0]
		print e
	sys.exit()
	return

def scrap(date):
	# root_url = "http://www.goal.com/en-us/fixtures"
	root_url = "http://www.goal.com/en-us/results/" + date + "?ICID=FX_CAL_1"
	# root_url ="http://www.goal.com/en-us/results/2017-03-04?ICID=FX_CAL_1"
	# root_url ="http://www.goal.com/en-us/results/2017-03-05?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-03-11?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-03-12?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-03-13?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-01?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-02?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-05?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-08?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-09?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-11?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-15?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-04-16?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-02-04?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-02-11?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-02-18?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-14?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-21?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-22?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-29?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-28?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-21?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-22?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-15?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-14?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-08?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-07?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2017-01-01?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-31?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-26?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-27?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-28?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-18?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-17?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-10?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-04?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-12-03?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-11-27?ICID=FX_CAL_1"
	# root_url = "http://www.goal.com/en-us/results/2016-11-26?ICID=FX_CAL_1"
	root_page = urllib2.urlopen(root_url)
	soup = BeautifulSoup(root_page)
	live_links = soup.find_all('a' , 'match-btn')
	# while (True):
	with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
		future_to_url = dict((executor.submit(scrap_commentary, link.get('href')), link.get('href'))
					 for link in live_links)

	for future in concurrent.futures.as_completed(future_to_url):
		url = future_to_url[future]
		if future.exception() is not None:
			print('%r generated an exception: %s' % (url,future.exception()))
		else:
			print('%r page is %d bytes' % (url, len(future.result())))

		# time.sleep(120)
	




# scrap()
    
