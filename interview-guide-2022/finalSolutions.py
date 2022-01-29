"""
All Solutions to coding interview for Linkedln SRE
Author: Olusegun Bolofinde

The coding round is easy. 5 questions. One had to be done by recursion, others were on text processing.
 It was on coderpad.io. You can use any language you want to code.
coding interview covering log parsing and HTTP requests.
Describe the relationship between a class and an object
He asked 4 coding questions, 1 pretty basic scripting, 2 log processing , and 1 network.
Parse a log file based on the interviewer's choice (String format is huge)
Questions on file-system traversal, log processing and APIs.
Write code to parse generic web logs and format it in different ways
First was a fizz buzz type question. Second and third were both log parsing


"""
import csv
import re
import unittest

import requests


def countEmployeesManagers(response):
    managersCount = {}

    for employee in response:
        manager = response[employee]

        if employee == manager:
            continue

        if employee not in managersCount:
            managersCount[employee] = []

        if manager not in managersCount:
            managersCount[manager] = [employee]
        else:
            managersCount[manager].append(employee)

    print(managersCount)

    output = dict()

    for manager in managersCount:
        output[manager] = len(managersCount[manager])

        for emp in managersCount[manager]:
            output[manager] += len(managersCount[emp])

        print("{}  :  {}".format(manager, output[manager]))


def countIPsInLog(fileName):
    countDict = dict()

    with open(fileName, 'r') as log:

        for ip in log:
            allip = ip.split(' - - ')[0]

            for ips in allip.split(' '):
                if ips in countDict:
                    countDict[ips] += 1
                else:
                    countDict[ips] = 1

    with open('finalIp', 'w') as data:

        ipLIST = csv.writer(data)

        headers = ['IP', 'Freq']

        ipLIST.writerow(headers)

        for item in countDict:
            ipLIST.writerow((item, countDict[item]))

    return countDict


# Write a program which prints out all numbers between 1 and 100. When the program would print out a number exactly
# divisible by 4, print "Linked" instead. When it would print out a number exactly divisible by 6, print "In"
# instead. When it would print out a number exactly divisible by both 4 and 6, print "LinkedIn." conceptual and
# design interview

def fizzBuzz(n):
    results = []

    for i in range(1, n + 1):

        if i % 4 == 0 and i % 6 == 0:
            results.append("LinkedIn")
        elif i % 4 == 0:
            results.append("Linked")
        elif i % 6 == 0:
            results.append("In")
        else:
            results.append(str(i))
    print(results)
    return results


def getEmployee(id, depth=0):
    if not id:
        return
    baseurl = ""

    url = f"{baseurl}/{id}"

    response = requests.get(url).json()
    if not response:
        return
    print(calculateDepthSpace(depth), f"{response['name']} - {response['title']}")

    for emp in response['reports']:
        getEmployee(emp, depth + 1)


def calculateDepthSpace(depth):
    space = ''

    while depth > 0:
        space += '  '
        depth -= 1
    return space


class EmployeeInfomation(object):

    def __init__(self):
        self.base

    url = "http:/www.facebook.com/api/employees"

    def getEmpById(self, id, depth=0):

        if not id:
            return
        url = f"{self.baseurl}/{id}"

        response = requests.get(url).json()

        if not response:
            return

        print(self.calDepth(depth), f"{response['name']}-{response['title']}")

        for emp in response['reports']:
            self.getEmpById(emp, depth + 1)

    def calDepth(self, depth):

        space = ' '
        while depth > 0:
            space += '  '

            depth -= 1
        return space

    def fizzBuzzSolution(self, n):

        result = []

        for i in range(1, n + 1):
            if i % 4 == 0 and i % 6 == 0:

                result.append("LinkedIn")
            elif i % 4 == 0:
                result.append("Linked")
            elif i % 6 == 0:
                result.append("In")
            else:
                result.append(str(i))

        return result


"""
The coding round is easy. 5 questions. One had to be done by recursion, others were on text processing.
coding interview covering log parsing and HTTP requests.
Describe the relationship between a class and an object
He asked 4 coding questions, 1 pretty basic scripting, 2 log processing , and 1 network.
Parse a log file based on the interviewer's choice (String format is huge)
Questions on file-system traversal, log processing and APIs.
Write code to parse generic web logs and format it in different ways
First was a fizz buzz type question. Second and third were both log parsing

classes are models to represent objects 
classes are blueprint consisting of properties and objects 

"""


def getEmployeeIdData(id, depth=0):
    if not id:
        return

    baseurl = "www.linkedln.com/api/employee"

    url = f"{baseurl}/{id}"

    response = requests.get(url).json()

    if not response:
        return

    def calculateDepth(depth):
        space = ' '
        while depth > 0:
            space += '   '

            depth -= 1
        return space

    print(calculateDepth(depth), f"{response['name']}- {response['title']}")
    for empid in response['reports']:
        getEmployeeIdData(empid, depth + 1)


class ProcessAnyFile(object):

    def __init__(self, filePath, func):
        """
        atrributes of files
        """
        self.filePath = filePath
        self.func = func

    def getIPs(self):

        ipDict = dict()

        with open(self.filePath, self.func) as log:
            for data in log:
                ips = data.split(' - - ')[0]
                for item in ips.split(' '):
                    if item not in ipDict:
                        ipDict[item] = 1
                    else:
                        ipDict[item] += 1

        return ipDict

    def IPsCSV(self, dic):
        with open(self.filePath, self.func) as data:
            cvsData = csv.writer(data)

            headers = ['IP', 'freq']

            cvsData.writerow(headers)

            for item in dic:
                cvsData.writerow((item, dic[item]))

    def readFileUsingRegex(self):

        with open(self.filePath, self.func) as file:
            log = file.read()

            regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'

            ip = re.findall(regex, log)

            return ip

    def readFileUsingSplit(self):

        file = open(self.filePath, self.func)
        lines = file.readlines()
        for line in lines:
            print(line.split('"GET ')[0].split(' - - ')[1].split()[0].split('[')[1].split('2015')[1])


if __name__ == '__main__':
    test1 = ProcessAnyFile('apache.log', 'r')
    # print(test1.getIPs())

    test3 = ProcessAnyFile('apache.log', 'r')
    # print(test3.readFileUsingRegex())

    test4 = ProcessAnyFile('apache.log', 'r')
    print(test4.readFileUsingSplit())
