# Bloomberg API usage adapted from Matthew Fieger's github

import urllib
import re
import json
import csv

def get_ticker_data(ticker, freq):
    try:
        htmltext = urllib.urlopen("http://www.bloomberg.com/markets/chart/data/" + frequency + "/" + ticker + ":US")
	data = json.load(htmltext)
	datapoints = data["data_values"]
        ticker_file = open("./prices/" + ticker + ".txt", "a")
        for point in datapoints:
            ticker_file.write(str(ticker + "," + str(point[0]) + "," + str(point[1]) + "\n"))
	    ticker_file.close()
    except:
        print "Exception: Bloomberg API failed to retrieve data."
