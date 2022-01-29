"""
The coding round is easy. 5 questions. One had to be done by recursion, others were on text processing.
coding interview covering log parsing and HTTP requests.
Describe the relationship between a class and an object
He asked 4 coding questions, 1 pretty basic scripting, 2 log processing , and 1 network.
Parse a log file based on the interviewer's choice (String format is huge)
Questions on file-system traversal, log processing and APIs.
Write code to parse generic web logs and format it in different ways
First was a fizz buzz type question. Second and third were both log parsing

Write a program which prints out all numbers between 1 and 100. When the program would print out a number exactly divisible by 4, print "Linked" instead.
 When it would print out a number exactly divisible by 6, print "In" instead.
 When it would print out a number exactly divisible by both 4 and 6, print "LinkedIn."
conceptual and design interview

classes are models to represent objects
classes are blueprint consisting of properties and objects

"""
import collections
import glob
import os
import re
import unittest
from pathlib import Path

import requests


class Solutions(object):

    def __init__(self):
        pass

    def linkedFizzBuzz(self, n):
        dic = {4: "Linked", 6: "In"}
        result = []

        for i in range(1, n + 1):

            result_str = ""

            for key in dic.keys():

                if i % key == 0:
                    result_str += dic[key]

            if not result_str:
                result_str = str(i)

            result.append(result_str)

        return result

    def getEmpIdById(self, id, depth=0):
        if not id:
            return
        baseurl = "linked.com/api/employee"
        url = f"{baseurl}/{id}"
        response = requests.get(url).json()
        if not response:
            return
        print(self.calDepth(depth), f"{response['title']}-{response['name']}")

        for empid in response['reports']:
            self.getEmpIdById(empid, depth + 1)

        def calDepth(depth):
            space = ''
            while depth > 0:
                space += '  '
                depth -= 1
            return space






    def logProcessing(self, fileName):

        dic = dict()

        with open(fileName, 'r') as file:
            for line in file:
                ipAddresses = line.split(' - - ')[0].split()

                for item in ipAddresses:
                    if item not in dic:
                        dic[item] = 1
                    else:
                        dic[item] += 1
        return dic

    def logProcessingUsingRegex(self, filename):

        with open(filename, 'r') as file:
            log = file.read()

            regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            result = re.findall(regex, log)

            return result

    def fileDirectory(self):

        for directoryPath, dirNames, fileNames in os.walk('/Volumes/dev-env/interview-prep-guide'):
            print(
                f"Root:{directoryPath}\n"
                f"Sub-directories:{dirNames}\n"
                f"Files: {fileNames}\n\n"

            )

    def countUniqueIp(self, file):

        result = []

        with open(file, 'r') as log:
            for item in log:
                data = item.split('- -')[0].split()
                # print(data)
                for d in data:
                    result.append(d)

        print(len(result))
        lst = list(set(result))

        lst2 = list(collections.OrderedDict.fromkeys(result))
        print(len(lst2))
        print(len(lst))

    def findFilesInAdirec(self):
        # for item in glob.glob(r'/Volumes/dev-env/interview-prep-guide/*.py'):

        input_dir = Path.cwd() / "interview-guide-2022"

        files = list(input_dir.glob("*.py"))

        print(files)


"""
class TestLinkedFizz(unittest.TestCase):
    test1 = Solutions()

    def testOut(self):
        result = self.test1.linkedFizzBuzz(4)
        assert result[3] == "Linked"

    def test2(self):
        assert self.test1.linkedFizzBuzz(100)[99] == "Linked"
        
        
        for data in os.listdir():
            print(data)

    def findAllFiles(self):
        dir = Path.cwd() / "interview-prep-guide"

        files = list(dir.glob("*.py"))
        print(files)


"""
if __name__ == '__main__':
    test1 = Solutions()
    test1.findFilesInAdirec()
