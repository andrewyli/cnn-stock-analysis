from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib
import re
import json
import csv
import http.client

from six.moves import xrange
import tensorflow as tf

# GLOBAL CONFIG PARAMETERS
EXAMPLE_SIZE = 30
DESIGN_WIDTH = EXAMPLE_SIZE - 1

data_exists = {
    "nasdaq": False
    "nyse": False
    "amex": False
}

def get_ticker_data(ticker, freq):
    # Reads Bloomberg stock data into a file labeled with the ticker
    # Heavily adapted from https://github.com/matthewfieger/bloomberg_stock_data
    try:
        ticker_url = "http://www.bloomberg.com/markets/chart/data/" + freq + "/" + ticker + ":US"
        response = urllib.request.urlopen("http://www.bloomberg.com/markets/chart/data/" + str(freq) + "/" + ticker + ":US")
        str_response = response.read().decode("utf-8")
        data = json.loads(str_response)
        datapoints = data["data_values"]
        ticker_file = open("./prices/" + ticker + ".txt", "w")
        for point in datapoints:
            ticker_file.write(str(ticker + "," + str(point[0]) + "," + str(point[1]) + "\n"))
        ticker_file.close()
    except:
        print("Exception: Bloomberg API failed to retrieve data.")

def get_all_data(exchange):
    with open("./tickers/" + exchange + ".csv", "rb") as tickers:
        reader = csv.reader(tickers)
        for row in reader:
            get_ticker_data(row[0])
    data_exists[exchange] = True

def read_ticker_data(ticker):
    # Opens ticker data file and returns second granularity data
    with open("./prices/" + ticker + ".txt", "r") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            # convert to ints
            tokens = [int(val) for val in line.split(",")]
            data.append(tokens)
    return data

def build_ticker_data(raw_data):
    design_matrix = []
    label_vector = []
    for i in xrange(len(raw_data) - EXAMPLE_SIZE):
        design_matrix.append(raw_data[i : i + DESIGN_WIDTH])
        label_vector.add((raw_data[i + DESIGN_WIDTH] > raw_data[i + DESIGN_WIDTH - 1]))
    return [design_matrix, label_vector]

def generate_training_data(exchange):
    training_data = []
    if not (exchange in data_exists and data_exists[exchange] = True):
        get_all_data(exchange)
    with open("./tickers/" + exchange + ".csv", "rb") as tickers:
        reader = csv.reader(tickers)
        for row in reader:
            training_data.append(build_ticker_data(row[0]))
    return training_data





# each data point is a day? See "sliding window"
# what can we classify from each of these?
# - volatility
# - whether the price ended higher or lower
# - actual ending value
# sliding window architecture, over intervals of X for each day... or maybe even just the year.
# are we even using a CNN? Let's consider a 10Y dataset oh but if there are a lot of stocks it's viable
# 10 Y = 300 M seconds, if we call each image 3000 seconds we could do something in the 100000s. What if we just used a normal NN over windwos of 300 (299 + 1) each to predict the next value?
# Need to remember to validate on different sets (tickers)
# Why second-level granularity? Because you can avoid bullshit like the program running too slow in Python and failing to execute the trade on time
# does the second at which the price is updated matter at all? Also are we classifying up/down/constant or a numerical predictive value? I feel like having an output layer of 3 is good or at least simpler
# What if the NN is as deep as DESIGN_WIDTH? That would almost be like hypercurve decision making
# - this kind of defeats the purpose of the NN now that I think about it.
# I should have a layer that literally just changes the importance of each node based on its seconds value, just a copy layer

# If my NN has a boolean output...is there a better way to do this? or like, what's a simple way of looking at the problem?
