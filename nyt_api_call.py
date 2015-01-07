__author__ = 'simranjitsingh'
import urllib
import time
import datetime
import json
import mysql.connector
import sys
import re
import os

cnx = mysql.connector.connect(user='root', password='simran',
                              host='127.0.0.1',
                              database='comment_iq')
cursor = cnx.cursor()

COMMUNITY_API_KEY = "5f933be26992203507b0963c96c653f1:4:66447706"
COMMUNITY_API_KEY2 = "6adcef7a975045db61389446ca15283e:1:30173638"
COMMUNITY_API_KEY3 = "5a3d3ff964c9975c0f23d1ad3437dd45:0:70179423"

key1_limit = 4999
key2_limit = 9998
key3_limit = 14997

global g_offset
global g_day

def error_name(d,offset):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    msg = str(exc_type)
    error = re.split(r'[.]',msg)
    error = re.findall(r'\w+',error[1])
    error_msg = str(error[0]) + "occured in line " + str(exc_tb.tb_lineno) + " ,Last API call date: " + str(d) + " , offset: " + str(offset)
    return error_msg

def escape_string(string):
    res = string
    res = res.replace('\\','\\\\')
    res = res.replace('\n','\\n')
    res = res.replace('\r','\\r')
    res = res.replace('\047','\134\047') # single quotes
    res = res.replace('\042','\134\042') # double quotes
    res = res.replace('\032','\134\032') # for Win32
    return res

class NYTCommunityAPI (object):
    URL = "http://api.nytimes.com/svc/community/v2/comments/by-date/"
    def __init__(self,key):
        self.nQueries = 0
        self.api_key = key
        self.QUERY_LIMIT = 30
        self.LAST_CALL = 0
        self.nCalls = 0;

    def apiCall(self, date, offset=0):
        interval = self.LAST_CALL - time.time()
        if interval < 1:
            self.nQueries += 1
            if self.nQueries >= self.QUERY_LIMIT:
                time.sleep (1)
                self.nQueries = 0

        params = {}
        params["api-key"] = self.api_key
        params["offset"] = str(offset)
        params["sort"] = "oldest"

        url = self.URL + date + ".json?" + urllib.urlencode (params)
        print url
        response = json.load(urllib.urlopen(url))
        self._LAST_CALL = time.time()
        self.nCalls += 1
        return response

def CollectComments():
    pagesize = 25
    API_KEY = COMMUNITY_API_KEY
    nytapi = NYTCommunityAPI(API_KEY)
# originally started collection from 20140115
    d_start = datetime.date(2014,04,22)
    d_end = datetime.date(2014,06,05)
    d = d_start
    global g_offset
    global g_day
    count = 0
    while d < d_end:
        g_day = d
        offset = 0
        date_string = d.strftime("%Y%m%d")
# Get the total # of comments for today
        r = nytapi.apiCall(date_string, offset)
        totalCommentsFound = r["results"]["totalCommentsFound"]
        print "Total comments found: " + str(totalCommentsFound)
        count += 1
# Loop through pages to get all comments
        while offset < totalCommentsFound:
            g_offset = offset
            if (count > key1_limit) and (count < key2_limit):
                if API_KEY != COMMUNITY_API_KEY2:
                    API_KEY = COMMUNITY_API_KEY2
                    nytapi = NYTCommunityAPI(API_KEY)
            if (count > key2_limit) and (count < key3_limit):
                if API_KEY != COMMUNITY_API_KEY3:
                    API_KEY = COMMUNITY_API_KEY3
                    nytapi = NYTCommunityAPI(API_KEY)
            if count > key3_limit:
                d_end = d
                print "last call on date: " + str(d)
                print "last offset value: " + str(offset-25)
                break;
            r = nytapi.apiCall(date_string, offset)
            # DB insertion call here.
            if "comments" in r["results"]:
                for comment in r["results"]["comments"]:
                    commentBody = escape_string(str(comment["commentBody"].encode("utf8")))
                    approveDate = int(comment["approveDate"])
                    recommendationCount = int(comment["recommendationCount"])
                    display_name = escape_string(str(comment["display_name"].encode("utf8")))
                    location = ""
                    if "location" in r:
                        location = escape_string(str(comment["location"].encode("utf8")))
                    commentSequence = int(comment["commentSequence"])
                    status = escape_string(str(comment["status"].encode("utf8")))
                    articleURL = escape_string(str(comment["articleURL"].encode('utf8')))
                    editorsSelection = int(comment["editorsSelection"])
                    insert_query = "INSERT INTO comments (status, commentSequence, commentBody," \
                                   " approveDate, recommendationCount, editorsSelection, display_name," \
                                   " location, articleURL)" \
                                   " VALUES('%s', %d, '%s', FROM_UNIXTIME(%d), %d, %d, '%s', '%s', '%s')" % \
                                   (status.decode("utf8"), commentSequence, commentBody.decode("utf8"), approveDate,
                                    recommendationCount, editorsSelection, display_name.decode("utf8"),
                                    location.decode("utf8"), articleURL.decode("utf8"))
                    cursor.execute(insert_query)

            cnx.commit()
            offset = offset + pagesize
            count += 1
            print "#Calls: " + str(nytapi.nCalls)
            print "counter value: " + str(count)
# Go to next day
        d += datetime.timedelta(days=1)
try:
    CollectComments()
except:
    print error_name(g_day,g_offset)
cnx.close()