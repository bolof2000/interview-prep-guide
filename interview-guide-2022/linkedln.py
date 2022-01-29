import json
import random

import requests
from requests.auth import HTTPBasicAuth


class APICall(object):

    def __init__(self):
        self.username = "f2f9f87d-c8af-498c-9f09-4c1256371098"
        self.password = "c4c90ed8-b8b5-403b-aab0-f1a91e3f371d"
        self.baseurl = "https://sandbox-api.marqeta.com/v3"
        self.cardProductEndpoint = "/cardproducts"
        self.headers = {"content-type": "application/json"}
        self.userendpoint = "/users"

    def getCall2(self):
        res = requests.get(self.baseurl + self.cardProductEndpoint, auth=HTTPBasicAuth(self.username, self.password))
        print(res.json())

    def createUser(self):
        data = {
            "token": random.randint(1, 11)
        }
        response = requests.post(self.baseurl + self.userendpoint, data=json.dumps(data), headers=self.headers,
                                 auth=HTTPBasicAuth(self.username, self.password))
        print(response.status_code)
        print(response.json())

    def getAllUsers(self):
        res = requests.get(self.baseurl + self.userendpoint, auth=HTTPBasicAuth(self.username, self.password))
        d = res.json()

        return d

    
if __name__ == '__main__':
    test = APICall()
    print(test.getAllUsers())
