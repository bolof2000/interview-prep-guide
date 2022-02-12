from typing import List

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
        for i, char in enumerate(s):
            if mapp.get(char) == 1:
                return i

        return -1


def reverseWords(s):
    def reverse_helper(left, right):
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    reverse_helper(0, len(s) - 1)

    left = 0
    for i, char in enumerate(s):
        if char == " ":
            reverse_helper(left, i - 1)
            left = i + 1
    reverse_helper(left, len(s) - 1)

    def longestPalindrome2(s):
        result = ""

        for i in range(s):
            current = expand(s, i - 1, i + 1)
            in_btw = expand(s, i, i + 1)
            result = max(result, current, in_btw, key=len)

        return result

    def expand(s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1

        return s[left + 1:right]

    def merge(self, nums1, m, nums2, n):
        # def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """

        index = m + n - 1
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[index] = nums1[m - 1]
                m -= 1
            else:
                nums1[index] = nums2[n - 1]

                n -= 1

            index -= 1

        while n > 0:
            nums1[index] = nums2[n - 1]
            n -= 1
            index -= 1

    def lowestCommonAncestor(root, p, q):
        if not root:
            return None

        left_res = lowestCommonAncestor(root.left, p, q)
        right_res = lowestCommonAncestor(root.right, p, q)
        if (left_res and right_res) or (root in [p, q]):
            return root
        else:
            return left_res or right_res
