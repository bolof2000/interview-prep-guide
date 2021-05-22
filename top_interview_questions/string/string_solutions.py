from collections import defaultdict
from collections import deque
from typing import List
from heapq import heappop, heappush

class StringSolutions:

    def strStr(self, haystack,needle):

        if len(haystack) == 0 and len(needle) == 0:
            return 0

        for i in range(len(haystack)):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1


test = StringSolutions()
print(test.strStr("",""))
