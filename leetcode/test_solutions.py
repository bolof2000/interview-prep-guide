import pytest
from cucumber.leetcode.solutions import Solutions


# ------------------------------------------------------------------------------------------
# fixture
# ------------------------------------------------------------------------------------------
# @pytest.fixture
# def test_method_access():
# return Solutions()

# ------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------
class Testsolutions:
    test_methods = Solutions()

    def test_longestPalindrome(self):
        actual_result = self.test_methods.longestPalindrome("babad")
        assert actual_result == "bab"

    def test_length_of_longest_substring(self):
        actual_result = self.test_methods.lengthOfLongestSubstring("abcabcbb")
        assert actual_result == 3

    def test_is_palindrome(self):
        actual_result = self.test_methods.isPalindrome("abba")
        assert actual_result == True

    def test_first_uniq_char(self):
        actual_result = self.test_methods.firstUniqChar("leetcode")
        assert actual_result == 0

    @pytest.mark.parametrize('test_input,expected',
                             [
                                 ('()[]{}', True),
                                 ('()', True),
                                 ('{}())', False)
                             ])
    def test_isValid(self, test_input, expected):
        assert self.test_methods.isValid(test_input) == expected

# Input: s = "aaabb", k = 3
# s = "()[]{}"
# s1 = "([)]"
# s3 = "{[]}"
# s = "abcabcbb"
# Input: nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
# Output: 2
# test = Solutions()
# print(test.lengthOfLongestSubstring(s))
# print(test.longestPalindrome("babad"))
# print(test.isValid(s1))
