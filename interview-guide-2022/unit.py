import requests
import os

base_uri = "http:/www.facebook.com/api/employees"

"http:/www.facebook.com/api/employees/001"

{
    "name": "Employee1",
    "title": "CEO",
    "report": ["0012", "0013"]
}

data = {
    10: {
        "name": "Olusegun Bolofinde",
        "title": "Senior QA Engineer",
        "report": [101, 201]},
    101: {
        "name": "Toba Bolofinde",
        "title": "Sr Network Engineer",
        "report": [110, 120]},
    110: {
        "name": "Dammy Bolofinde",
        "title": "Cyber Engineer",
        "report": [111, 112]},
    111: {
        "name": "Rere Bolofinde",
        "title": "Medical Doctor",
        "report": [116, 117]},
    117: {
        "name": "Michelle Bolofinde",
        "title": "Sr Cloud Engineer",
        "report": [118, 119]},
    112: {
        "name": "Adriel Bolofinde",
        "title": "Sr Space Engineer",
        "report": [120, 130]},
    120: {
        "name": "Emmanuel Bolofinde",
        "title": "Advosry engineer",
        "report": [121, 122]},
    121: {
        "name": "Timothy Bolofinde",
        "title": "Crypto Engineer",
        "report": [124, 123]},
    210: {
        "name": "Matt Bolofinde",
        "title": "CEO",
        "report": [211, 212]},
    211: {
        "name": "Shola Bolofinde",
        "title": "Mathematicians",
        "report": [216, 217]},
    217: {
        "name": "Bunmi Bolofinde",
        "title": "Scrum Master",
        "report": [218, 219]},
    212: {
        "name": "Solomon Famo",
        "title": "Game Developer Engineer",
        "report": [220, 230]},
    220: {
        "name": "Ayanfe Bolofinde",
        "title": "Artist engineer",
        "report": [221, 222]},
    221: {
        "name": "Grace Bolofinde",
        "title": "House wife engineer",
        "report": [224, 223]},
}


def printFilesInDirectory(folder):
    list_of_files = os.listdir(folder)
    for file in list_of_files:
        if '.py' in file:
            print(file)


def printAllFoldersInADirectory(folder):
    for root, dirs, files in os.walk(folder):
        print(root)


def calculateSpace(depth):
    space = ''
    while depth > 0:
        space += "   "
        depth -= 1
    return space


def getEmployee(id, depth=0):
    if not id:
        return
    response = data.get(id)

    if not response:
        return

    # if not (response and type(response, dict)):
    # return

    print(calculateSpace(depth), f"{response['name']} - {response['title']}")

    for i in response['report']:
        getEmployee(i, depth + 1)


if __name__ == '__main__':
    # print(getEmployee(10))
    printFilesInDirectory("/Volumes/dev-env/interview-prep-guide/interview-guide-2022")
