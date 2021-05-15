from listnode import ListNode
from collections import defaultdict
import pdb


class Solutions:

    def twoSum(self, nums, target):
        from collections import defaultdict
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = {}
        result = []
        for i, num in enumerate(nums):
            temp = target - num
            if temp in dic:
                result.append(i)
                result.append(dic[temp])
            dic[num] = i

        return sorted(result)

    def addTwoNumbers(self, l1, l2):
        output = ListNode(0)
        carry = 0
        dummy = output
        while l1 is not None or l2 is not None:
            while l1 is not None:
                carry += l1.val
                l1 = l1.next
            while l2 is not None:
                carry += l2.val
                l2 = l2.next

            dummy.next = ListNode(carry % 10)
            carry = carry / 10

            dummy = dummy.next

        return output.next

    def lengthOfLongestSubstring(self, s):
        dic = {}
        window_start = 0
        maxLen = 0
        for i in range(len(s)):
            char = s[i]
            if char in dic and dic[char] >= window_start:
                window_start = dic[char] + 1
            dic[char] = i
            maxLen = max(maxLen, i - window_start + 1)

        return maxLen

    def longestPalindrome(self, s):
        longest = ""
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                substring = s[i:j + 1]
                if len(substring) > len(longest) and self.isPalindrome(substring):
                    longest = substring
        return longest

    def isPalindrome(self, s):
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False

            left += 1
            right -= 1
        return True

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        s = str(abs(x))

        result = int(s[::-1])

        if result > 2147483647:
            return 0

        return result if x > 0 else (result * -1)

    def threeSum(self, nums):
        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i == 0 or i > 0 and nums[i] != nums[i - 1]:
                left = i + 1
                right = len(nums) - 1
                while left < right:
                    temp = nums[i] + nums[left] + nums[right]
                    if temp == 0:
                        result.append(nums[i])
                        result.append(nums[left])
                        result.append(nums[right])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1

                        left += 1
                        right -= 1
                    elif temp > 0:
                        right -= 1
                    else:
                        left += 1

        return result

    def isValid(self, s):
        stack = []
        for i in range(len(s)):
            char = s[i]
            if char == '(' or char == '[' or char == '{':
                stack.append(char)
            elif char == ')' and len(stack) > 0 and stack[-1] == '(':
                stack.pop()
            elif char == ']' and len(stack) > 0 and stack[-1] == '[':
                stack.pop()
            elif char == '}' and len(stack) > 0 and stack[-1] == '{':
                stack.pop()
            else:
                return False

        return len(stack) == 0

    def fourSumCount(self, nums1, nums2, nums3, nums4):
        result = 0
        dic = {}
        for i in range(len(nums1)):
            x = nums1[i]

            for j in range(len(nums2)):
                y = nums2[j]
                z = x + y
                if z not in dic:
                    dic[z] = 0
                dic[z] += 1
        for i in range(len(nums3)):
            x = nums1[i]
            for j in range(len(nums4)):
                y = nums2[j]
                z = x + y
                target = -z
                if target in dic:
                    result += dic[target]

        return result

    def firstUniqChar(self, s):
        mapp = {}
        for char in s:
            if char in mapp:
                mapp[char] += 1
            else:
                mapp[char] = 1
        print(mapp)
        for i,char in enumerate(s):
            if mapp.get(char) ==1:
                return i

        return -1