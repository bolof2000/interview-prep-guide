# 1. coding interview covering log parsing and HTTP requests.  ---- log and http request
import json
import re
from collections import Counter
from collections import deque
from collections import defaultdict
from collections import OrderedDict
import csv


def fileReader(filename):
    try:
        with open(filename, 'r') as log:
            logData = log.read()

            regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            ip_address = re.findall(regex, logData)

            return ip_address

    except FileNotFoundError:
        print("Invalid file path")

    finally:
        if len(logData) > 0:
            print(len(logData))
        else:
            print("No file processed")


def readLogFilesAndExtractIPs(filename):
    with open(filename) as f:
        logData = f.read()

        regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'

        ipLists = re.findall(regex, logData)

        return ipLists


def countTotalIPAddressInALog(ipLists):
    if ipLists is not None:
        return Counter(ipLists)


def countIPSecondSolution(ipLists):
    count = defaultdict()
    for item in ipLists:
        if item not in count:
            count[item] = 1
        else:
            count[item] += 1
    return count


def countIPSecondSolution2(ipLists):
    dic = {}
    for item in ipLists:
        if item not in dic:
            dic[item] = 1
        else:
            dic[item] += 1
    return dic


def writeIPToFile(ipcounts):
    with open('output.csv', 'w') as file:
        writer = csv.writer(file)

        header = ['IP', 'Frequency']

        writer.writerow(header)

        for item in ipcounts:
            writer.writerow((item, ipcounts[item]))


def regexPatterns(fileName):
    dic = {}
    with open(fileName, 'r') as file:

        for line in file:
            data = line.split(' "GET ')[0].split(' - - ')[0]
            for item in data.split(' '):
                if item in dic:
                    dic[item] += 1
                else:
                    dic[item] = 1

        print(dic)


def logParsingSolutionUsingSplitFunction(fileName):
    dic = dict()
    with open(fileName, 'r') as file:
        for item in file:
            iplists = item.split(' - - ')[0]

            for item in iplists.split(' '):
                if item not in dic:
                    dic[item] = 1
                else:
                    dic[item] += 1

    with open('out.csv', 'w') as ip:
        output = csv.writer(ip)

        header = ['IP','freq']

        output.writerow(header)

        for item in dic:

            output.writerow((item,dic[item]))

    print(json.dumps(dic))



if __name__ == '__main__':
    # print(fileReader("apache.log"))

    # ipLists = readLogFilesAndExtractIPs('apache.log')
    # print(countTotalIPAddressInALog(ipLists))
    # count = countIPSecondSolution2(ipLists)

    # writeIPToFile(count)

    # print(readLogFilesAndExtractIPs('apache.log'))

    # regexPatterns('apache.log')

    logParsingSolutionUsingSplitFunction('apache.log')
