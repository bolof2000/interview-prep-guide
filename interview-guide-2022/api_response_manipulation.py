import requests
import json


def getRequest():
    response = requests.get('https://formulae.brew.sh/api/formula.json')

    package_json = response.json()[0]['name']

    print(json.dumps(package_json, indent=2))


def getAllPackage(name):
    url = f'https://formulae.brew.sh/api/formula/{name}.json'

    response = requests.get(url).json()

    if name == response['analytics']['install']['30d'][name]:
        getAllPackage(name)

    print(json.dumps(response, indent=2))


def assignAndPrint(dic):
    result = dict()

    for employee in dic:

        manager = dic[employee]

        if employee == manager:
            continue
        if employee not in result:
            result[employee] = []

        if manager not in result:
            result[manager] = [employee]
        else:
            result[manager].append(employee)

    # count the managers for each employee

    managerDic = dict()
    for item in result:
        managerDic[item] = len(result[item])
        for emp in result[item]:
            managerDic[item] += len(result[emp])

        print("{}  :   {}".format(item,managerDic[item]))


def logParsing(fileName):

    file = open(fileName,'r')
    print(file)


# dict =  {A:C,B:C,C:F,D:E,E:F,F:F}
if __name__ == '__main__':
    #dic = {"A": "C", "B": "C", "C": "F", "D": "E", "E": "F", "F": "F"}
    #assignAndPrint(dic)
    logParsing('apache.log')
