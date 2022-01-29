from collections import defaultdict
from typing import List
from math import inf
from collections import deque
from heapq import heappush,heappop


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val


class TreeNode:
    def __init__(self, left=None, right=None, val=0):
        self.left = left
        self.right = right


class Solutions:

    def threeSum(self, nums):
        result = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i == 0 or (i > 0 and nums[i] != nums[i - 1]):

                left = i + 1
                right = len(nums) - 1

                while left < right:

                    temp = nums[left] + nums[right] + nums[i]
                    if temp == 0:
                        result.append([nums[i], nums[left], nums[right]])

                        while left < right and nums[left] == nums[left + 1]:
                            left += 1

                        while left < right and nums[right] == nums[right - 1]:
                            right -=1

                        left += 1
                        right -= 1

                    elif temp > 0:
                        right -= 1
                    else:
                        left += 1

        return result

    def exist(self, board, word):
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == word[0] and self.dfs(board, row, col, word):
                    return True

        return False

    def dfs(self, board, row, col, word):
        if len(word) == 0:
            return True
        if row < 0 or row >= len(board) or col < 0 or col > len(board[0]) or board[row][col] != word[0]:
            return False

        temp = board[row][col]
        board[row][col] = ' '
        if (self.dfs(board, row - 1, col, word[1:])
                or self.dfs(board, row + 1, col, word[1:])
                or self.dfs(board, row, col - 1, word[1:])
                or self.dfs(board, row, col + 1, word[1:])
        ):
            return True

        board[row][col] = temp

        return False

    def productArrayExceptSelf(self, nums):

        result = [1] * len(nums)
        multiply = 1
        for i in range(0, len(nums)):
            result[i] = result[i] * multiply
            multiply = multiply * nums[i]

        multiply = 1

        for i in range(len(nums) - 1, -1, -1):
            result[i] = result[i] * multiply
            multiply = multiply * nums[i]

        return result

    def subArrayEqualSum(self, nums, k):
        mapp = defaultdict(int)
        mapp.setdefault(0, 1)
        summ = 0
        count = 0
        for num in nums:
            summ += num
            temp = summ - k

            if temp in mapp:
                count += mapp.get(temp)

            mapp[summ] += 1

        return count

    def decodeString(self, s):
        stack = []
        for char in s:
            if char != ']':
                stack.append(char)
            else:

                res = []
                while len(stack) > 0 and str.isalpha(stack[-1]):
                    res.insert(0, stack.pop())

                final_string = "".join(res)
                stack.pop()

                digit = []
                while len(stack) > 0 and str.isdigit(stack[-1]):
                    digit.insert(0, stack.pop())

                final_digit = "".join(digit)
                int_digit = int(final_digit)

                while int_digit > 0:
                    for charr in final_string:
                        stack.append(charr)
                    int_digit -= 1

        output = []
        while len(stack) > 0:
            output.insert(0, stack.pop())

        return "".join(output)

    def houseRobber(self, nums):
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])

        dp = [nums[0], max(nums[0], nums[1])]

        for i in range(2, len(nums)):
            dp.append(max(dp[i - 2] + nums[i], dp[i - 1]))

        return dp.pop()

    def removeElements(self, head: ListNode, val: int) -> ListNode:

        dummy = ListNode(0)
        output = dummy
        dummy.next = head
        while head is not None:
            if head.val == val:
                head = head.next
                dummy.next = head
            else:
                head = head.next
                dummy = dummy.next

        return output.next

    @staticmethod
    def longestPalindrome(s: str) -> int:
        stack = []
        count = 0
        for char in s:
            if char not in stack:
                stack.append(char)
            else:
                if len(stack) > 0 and char in stack:
                    stack.pop()
                    count += 1

        result = count * 2
        if len(stack) > 0:
            result += 1

        return result

    @staticmethod
    def longestOnes(nums: List[int], k: int):

        """
         A = [1,1,1,0,0,0,1,1,1,1,0]  k = 2
        """

        window_start = 0
        right = 0
        while right < len(nums):
            if nums[right] == 0:
                k -= 1

            if k < 0:
                if nums[right] == 0:
                    k += 1
                    window_start += 1

            right += 1

        return right - window_start

    def pivotIndex(self, nums: List[int]) -> int:

        summation = sum(nums)

        accum = 0
        for i in range(len(nums)):
            if summation - nums[i] - accum == accum:
                return i
            accum += nums

        return accum

    def fib(self, n):
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)

    def fib(self, n, memo):

        if n < 2:
            return memo[n]

        res = self.fib(n - 1, memo) + self.fib(n - 2, memo)

        memo[n] = res

        return res

    def rotateArray(self, nums, k):

        self.helper(nums, 0, len(nums) - 1)
        self.helper(nums, 0, k - 1)
        self.helper(nums, k, len(nums) - 1)

        return nums

    def helper(self, nums, left, right):

        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = dummy

        while n > 0:
            fast = fast.next
            n -= 1

        while fast and fast.next is not None:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return dummy.next

    def findMaxConsecutiveOnes(self, nums):
        window_start = 0
        max_len = 0
        for i in range(len(nums)):

            if nums[i] == 1:
                max_len = max(max_len, i - window_start + 1)
            else:
                window_start = i + 1

        return max_len

    def removeDuplicates(self, s):
        stack = []
        for char in s:
            if char not in stack:
                stack.append(char)
            else:
                if len(stack) > 0 and char == stack[-1]:
                    stack.pop()
        return "".join(stack)

    def searchMatrix(self, matrix, target):

        if len(matrix) == 0:
            return False

        row = len(matrix)
        col = len(matrix[0])
        left = 0
        right = row * col - 1

        while left <= right:
            mid = (left + right) // 2
            midNum = matrix[mid / col][mid % col]
            if target == midNum:
                return True

            elif target > midNum:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def searchMatrixII(self, matrix, target):

        if len(matrix) == 0:
            return False

        row = len(matrix)
        col = len(matrix[0])

        currentRow = 0
        currentCol = col - 1
        while currentRow < row and currentCol >= 0:
            if matrix[currentRow][currentCol] == target:
                return True
            if matrix[currentRow][currentCol] > target:
                currentCol -= 1
            else:
                currentRow += 1
        return False

    def minSubArrayLen(self, s, nums):
        if len(nums) == 0:
            return 0
        window_start = 0
        minLen = float('inf')
        summation = 0
        for i in range(len(nums)):
            summation += nums[i]
            while summation >= s:
                min(minLen, i - window_start + 1)

                summation -= nums[window_start]

                window_start += 1

        if minLen == float('inf'):
            return 0

        return minLen

    def transposeMatrix(self, matrix):

        row = len(matrix)
        col = len(matrix[0])

        result = [col][row]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[j][i] = matrix[i][j]

        return result

    def maxProfit(self, prices):
        profit = 0
        cheapest_price = prices[0]
        for price in prices:
            cheapest_price = min(price, cheapest_price)
            current_profit = price - cheapest_price
            profit = max(current_profit, profit)

        return profit

    def isSymmetric(self, root: TreeNode) -> bool:

        # what is the return value
        #  what is the state to pass to the child
        # the state here is to compare left and right node

        # the base case needs to do this and then recursive calls do that for the child

        def dfs(left: TreeNode, right: TreeNode) -> bool:

            if left is None and right is None:
                return True
            if left is None or right is None or left.val != right.val:
                return False

            return dfs(left.left, right.right) and dfs(left.right, right.left)

        if root is None:
            return True

        return dfs(root.left, root.right)

    def invertBinaryTree(root: TreeNode):

        if not root:
            return None

        def dfs(root: TreeNode):
            temp = root.left
            root.left = root.right
            root.right = temp

            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return root

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:

        def dfs(t1, t2):
            if t1 is None and t2 is None:
                return None

            if t1 is None:
                return t2
            if t2 is None:
                return t1

            t3 = TreeNode(t1.val + t2.val)

            dfs(t1.left, t2.left)
            dfs(t1.right, t2.right)

            return t3

        return dfs(t1, t2)

    def palidroneLinkedList(self, head: ListNode) -> bool:

        res = []
        current = head
        while head is not None:
            res.append(head.val)
            head = head.next

        left = 0
        right = len(res) - 1
        while left < right:
            if res[left] != res[right]:
                return False

            left += 1
            right -= 1

        return True

    def inOrderTraversal(self, root: TreeNode):

        def dfs(root, result):
            if not root:
                return []

            dfs(root.left, result)
            result.append(root.val)
            dfs(root.right, result)

        res = []
        dfs(root, res)

        return res

    def minimumAddToMakeValid(self, s):

        if len(s) == 0:
            return 0

        stack = []

        for char in s:
            if char == '(':
                stack.append(char)
            elif len(stack) > 0 and stack[-1] == '(':
                stack.pop()
            else:
                stack.append(char)

        return len(stack)

    def maxDepth(self, root: TreeNode) -> int:

        def dfs(root):
            if not root:
                return 0

            return max(dfs(root.left), dfs(root.right)) + 1

        return dfs(root)

    def reverse_string(self, s):
        res = []
        for i in range(len(s) - 1, -1, -1):
            res.append(s[i])

        return res

    def middleNode(self, root: ListNode) -> ListNode:
        slow = root
        fast = root

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

        return slow

    def mergeTwoSortedList(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        output = dummy

        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                dummy.next = l2
                l2 = l2.next

            else:
                dummy.next = l1
                l1 = l1.next

            dummy = dummy.next

        if l1 is not None:
            dummy.next = l1

        if l2 is not None:
            dummy.next = l2

        return output.next

    def threeSumClosest(self, nums, target):
        closest = nums[0] + nums[1] + nums[len(nums) - 1]
        nums.sort()
        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1

            while left < right:

                current = nums[left] + nums[i] + nums[right]

                if abs(target - current) < abs(target - closest):
                    closest = current
                elif target == current:
                    return target
                elif target > current:
                    left += 1
                else:
                    right -= 1

        return closest

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:

        if root.val == val:
            return root

        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    def reverse_only_letters(self, s):
        """
        input :  ab-cd
        """
        stack = []
        res = []
        for char in s:
            if str.isalpha(char):
                stack.append(char)
        print(stack)

        for ca in s:
            if str.isalpha(ca):
                res.append(stack.pop())

            else:
                res.append(ca)
        print(res)

        return "".join(res)

    def rangeSumBST(self, root: TreeNode, l: int, r: int) -> int:

        def dfs(root):
            summation = 0
            if not root:
                return 0
            if l <= root.val <= r:
                summation += root.val

            summation += dfs(root.left)
            summation += dfs(root.right)

            return summation

        return dfs(root)

    def sortArrayByParity(self, nums):

        even = []
        odd = []
        for num in nums:
            if num % 2 == 0:
                even.append(num)
            if num % 2 == 1:
                odd.append(num)
        return even + odd

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        result = dummy

        carry = 0
        while l1 is not None or l2 is not None or carry != 0:
            if l1 is not None:
                carry += l1.val
                l1 = l1.next

            if l2 is not None:
                carry += l2.val
                l2 = l2.next

            dummy.next = ListNode(carry % 10)
            carry = carry / 10

            dummy = dummy.next

        return result.next

    def encode(self, str):
        """
        str = ["Kevin","is","great"]
        output : "5/kevin2/is5/great"
        """
        encoded = ""
        for word in str:
            length = len(word)
            encoded += length + "/" + word

        return encoded

    def decode(self, strs: str) -> List:

        position = 0
        decoded = []
        while position < len(strs):
            slash_position = strs.index("/", position)
            length = int(strs[slash_position - 1])
            position = slash_position + 1

            decoded.append(strs[position:position + length])
            position += length

        return decoded

    def climbingStairs(self, n):

        if n <= 3:
            return n

        ways = [0, 1, 2, 3]
        for i in range(4, n + 1):
            ways.append(ways[i - 1] + ways[i - 2])

        return ways.pop()

    def houseRobber(self, nums):

        if len(nums) == 0:
            return 0

        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[1], nums[2])
        dp = [nums[0], max(nums[1], nums[2])]
        for i in range(2, len(nums)):
            dp.append(max(nums[i] + dp[i - 2], dp[i - 1]))

        return dp.pop()  # or dp[-1]

    def maxSubArray(self, nums):

        maxSum = nums[0]
        currentSum = nums[0]
        for i in range(1, len(nums)):
            currentSum = max(nums[i] + currentSum, nums[i])
            maxSum = max(currentSum, maxSum)

        return maxSum

    def canAttentMeetings(self, intervals):
        intervals.sort()

        for i in range(len(intervals) - 1):
            if intervals[i + 1][0] < intervals[i][1]:
                return False

    def merge(self, intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda interval: interval[0])
        res = [intervals[0]]
        for current in intervals:
            last_interval = [-1]
            if current[0] <= last_interval[1]:
                last_interval[1] = max(current[1], last_interval[1])
            else:
                res.append(current)
        return res

        return True

    def spiralMatrix(self, matrix):
        result = []
        if not matrix:
            return []

        top = 0
        bottom = len(matrix) - 1

        left = 0
        right = len(matrix[0]) - 1
        direction = "right"

        while top <= bottom and left <= right:

            if direction == "right":
                for i in range(left, right + 1):
                    result.append(matrix[top][i])

                top += 1
                direction = "down"
            elif direction == "down":
                for i in range(top, bottom - 1):
                    result.append(matrix[i][right])

                right -= 1

                direction = "right"

            elif direction == "right":
                for i in range(bottom, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1

                direction = "up"

            elif direction == "up":
                for i in range(left, top - 1):
                    result.append(matrix[i][left])
                left -= 1
                direction = "right"
        return result

    """
    Graph Questions here . Delete and write again till you fully understands it 
    """

    def countConnectedNodes(self, N, edges):

        adjacent_list = {}
        visited = {}
        count = 0

        for vertex in range(N):
            adjacent_list[vertex] = []
            visited[vertex] = False

        for edge in edges:
            v1 = edge[0]
            v2 = edge[1]

            adjacent_list[v1].append(v2)
            adjacent_list[v2].append(v1)

        def dfs(vertex):
            visited[vertex] = True

            for neighbor in adjacent_list[vertex]:
                if not visited[neighbor]:
                    dfs(neighbor)

        for vertex in range(N):
            if not visited[vertex]:
                dfs(vertex)
                count += 1

        return count

    def canFinish(self, numCourses, prerequisites):

        adj_list = {}
        visited = {}

        for vertex in range(numCourses):
            adj_list[vertex] = []
            visited[vertex] = "white"

        for edge in prerequisites:
            v1 = edge[0]
            v2 = edge[1]
            adj_list[v1].append(v2)

        def dfs(vertex):
            visited[vertex] = "gray"
            for neighbor in adj_list[vertex]:
                if visited[neighbor] == "gray" and dfs(neighbor):
                    return True

            if visited[neighbor] == "white" and dfs(neighbor):
                return True

            visited[vertex] = "black"
            return False

        for vertex in range(numCourses):
            if visited[vertex] == "white":
                dfs(vertex)
                return False

        return True

    def maxDepthSolution(self, root: TreeNode) -> int:
        def dfs(root):
            if not root:
                return 0
            return max(dfs(root.left), dfs(root.right)) + 1

        return dfs(root)

    def visible_tree_node(self, root: TreeNode) -> int:

        def dfs(root, max_so_far):

            if not root:
                return 0

            total = 0
            if root.val > max_so_far:
                total += 1

            total += dfs(root.left, max(max_so_far, root.val))
            total += dfs(root.right, max(max_so_far, root.val))

            return total

        return dfs(root, -float('inf'))

    def validateBST(self, root: TreeNode) -> bool:

        def dfs(root: TreeNode, min_val: int, max_val: int) -> bool:

            if not root:
                return True
            if root.val <= min_val and root.val >= max_val:
                return False

            return dfs(root.left, min_val, root.val) and dfs(root.right, root.val, max_val)

        return dfs(root, -inf, inf)

    def serialize(self, root: TreeNode):

        res = []

        def dfs(root):
            if not root:
                res.append('x')
                return
            res.append(root.val)

            dfs(root.left)
            dfs(root.right)

        dfs(root)

        return ' '.join(res)

    def deserialize(self, s):

        def dfs(nodes):
            val = next(nodes)
            if val == 'x':
                return

            current = TreeNode(int(val))

            current.left = dfs(nodes)
            current.right = dfs(nodes)

            return current

        return dfs(iter(s.split()))

    def level_order_traversal(self, root):
        res = []
        queue = deque([root])
        while len(queue) > 0:

            n = len(queue)
            new_level = []
            for _ in range(n):
                node = queue.popleft()
                new_level.append(node.val)

            for child in [root.left, root.right]:
                if child is not None:
                    queue.append(child)

        res.append(new_level)

        return res

    def zig_zag(self, root):

        res = []
        if not root:
            return res
        left_to_right = True
        queue = deque([root])

        while len(queue) > 0:
            n = len(queue)
            new_level = []
            for _ in range(n):
                node = queue.popleft()
                new_level.append(node.val)
            for child in [node.left, node.right]:
                if child is not None:
                    queue.append(child)

            if not left_to_right:
                new_level.reverse()
            res.append(new_level)

            left_to_right = not left_to_right

        return res

    def permute(self, s):

        def dfs(path, used, res):

            if len(path) == len(s):
                res.append(path[:])
                return
            for i, letter in enumerate(s):
                if used[i]:
                    continue
                path.append(letter)
                used[i] = True

                dfs(path, used, res)
                path.pop()
                used[i] = False

        res = []
        dfs([], [False] * len(s), res)

        return res

    def wordBreak(self, s, wordDict):

        def dfs(i, memo):
            # base case

            if i == len(s):
                return True

            if i in memo:
                return memo[i]

            ok = False

            for word in wordDict:
                if s[i:].startswith(word):
                    if dfs(i + len(word), memo):
                        ok = True

            memo[i] = ok
            return ok

        return dfs(0, {})
    def wordBreakNoMemo(self,s,dictWord):

        def dfs(i):
            if i == len(s):
                return True

            for word in dictWord:
                if s[i:].startswith(word):
                    if dfs(i+len(word)):
                        return True

            return False

        dfs(0)


    def decode_ways(self,digits):
        prefixes = [str(i) for i in range(1,27)]
        def dfs(i,memo):

            if i in memo:
                return memo[i]
            ways =0
            for prefix in prefixes:
                if digits[i:].startswith(prefix):
                    ways += dfs(i+len(prefix),memo)


            memo[i] = ways

            return ways

        return dfs(0,{})

    def combination_sum(self,nums,target):
        pass

    def validate_ip4(self, ipaddress):
      pass 
       

    def validate_ip4(self, ipaddres):
      pass 
  

def minCost(cost,s):
  result = 0 
  pre= 0 
  for i in range(1,len(s)):
    if s[pre] != s[i]:
      pre = i 

    else:

      result += max(cost[pre],cost[i])
      if cost[pre] < cost[i]:
        pre = i 
  return result


def numIslands(grid):
  if(grid == None or len(grid)==0):
    return 0 
  count =0
  for row in range(len(grid)):
    for col in range(len(grid[0])):
      if grid[row][col] == "1":
        count +=1
        dfs(grid,row,col)

  return count 

def dfs(grid,row,col):
  if row <0 or row>= len(grid) or col <0 or col >=len(grid[0]) or grid[row][col] == "0":
    return
  grid[row][col] = "0"
  dfs(grid,row+1,col)
  dfs(grid,row-1,col)
  dfs(grid,row,col+1)
  dfs(grid,row,col-1)

class MSTInterviewPrepSolutions(object):

  def maxLength(self,arr):

    n = len(arr)
    result = 0

    def hasDuplicate(s):
      return len(s) != len(set(s))

    def backtrack(current_string,index):
      nonlocal result
      result = max(result, len(current_string))
      for i in range(index,n):
        new_str = current_string+arr[i]
        if not hasDuplicate(new_str):
          backtrack(new_str,i+1)

    backtrack("",0)

    return result


  def spiralMatrix(self,matrix):

    top = 0
    bottom = len(matrix)-1
    left =0
    right = len(matrix[0])-1
    direction = "right"
    result = []
    if not matrix:
      return result 

    while top <= bottom and left <= right:
      if direction == "right":
        for i in range(left,right+1):
          result.append(matrix[top][i])
        top +=1
        direction = "down"
      elif direction == "down":
        for i in range(top,bottom+1):
          result.append(matrix[i][right])
        right -=1
        direction = "left"
      elif direction == "left":
        for i in range(right,left-1,-1):
          result.append(matrix[bottom][i])
        bottom -=1
        direction = "up"
      
      elif direction == "up":
        for i in range(bottom,top-1,-1):
          result.append(matrix[i][left])
        left -=1
        direction = "right"

    return result 


    def firstMissingPositive(self,nums):

        """
      :type nums: List[int]
      :rtype: int
      Basic idea:
      1. for any array whose length is l, the first missing positive must be in range [1,...,l+1], 
          so we only have to care about those elements in this range and remove the rest.
      2. we can use the array index as the hash to restore the frequency of each number within 
          the range [1,...,l+1] 
      """


        n = len(nums)
        for i in range(len(nums)):
          num = nums[i]
          if num <0 or num >= n:
            num = 0   # delete those useless elements
        
        for i in range(len(nums)):
          x = nums[i]%n   # use the index as the has to record the freq of each number 
          nums[x] += n

        for i in range(1,len(nums)):
          num = nums[i]
          if num/n ==0:
            return i 

        return n
      

  def maxNetworkRank(self,n,roads):

    adj =[0]*n+1

    for a,b in roads:
      adj[a] +=1
      adj[b] +=1

      max_rank =0 

      for a,b in roads:
        max_rank = max(max_rank,adj[a]+adj[b]-1)
    return max_rank 

  def maxNetworkRank2(self,n,roads):

    adjacent_list = {}

    for vertex in range(n):
      adjacent_list[vertex] = []
    
    for sub in roads:
      v1 = sub[0]
      v2 = sub[1]
      adjacent_list[v1].append(v2)
      adjacent_list[v2].append(v1)


    max_so_far =0
    for i in range(n):
      for j in range(i+1,n):
        max_so_far= max(max_so_far,len(adjacent_list[i])+len(adjacent_list[j])-(i in adjacent_list[j]))
    
    return max_so_far

  def mergeKLists(lists:List[ListNode])-> ListNode: 

    dummy = ListNode(0)
    output = dummy
    queue = []
    for head in lists:
      while head is not None:
        heappush(queue,head.val)
        head = head.next 

    while len(queue) > 0:
      output.next = ListNode(heappop(queue))
      output = output.next 

    return dummy.next 

  
  def maxLength2(self,arr): 

    result = 0
    def hasDup(s):
        return len(s) != len(set(s))
    def dfs(current_str,index):

     # nonlocal result

      #result = max(result,len(current_str))

      for i in range(index,len(arr)):
        new_str = current_str+arr[i]

        if not hasDup(new_str):
          dfs(new_str,i+1)


    dfs("",0)
    return result 
  

  # solution not clear 
  def modifyString(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] == "?": 
                for c in "abc": 
                    if (i == 0 or s[i-1] != c) and (i+1 == len(s) or s[i+1] != c): 
                        s[i] = c
                        break 
        return "".join(s)

def reverseWords(s:str)->int:

  result = []
  str_new = s.split(" ")
  print(str_new)
  
  for i in range(len(str_new)-1,-1,-1):
    word = str_new[i]
    if len(word) >0:
      result.append(word)
  
  return " ".join(result)




class QueueUsingStack():

  def __init__(self):
    self.stack1 = []
    self.stack2 = []
  
  def push(self,x):
    self.stack1.append(x)

  def pop(self): 

    # fill the stack2 first and then pop the top 
    self.peek()
    return self.stack2.pop()

  def peek(self):
    if not self.stack2:
      while self.stack1:
        self.stack2.append(self.stack1.pop())
    

    return self.stack2[-1]

class Queue(object):

  def __init__(self):
    self.queue = []

  def isEmpty(self):
    return len(self.queue) == 0
  
  def enqueue(self,item):
    self.queue.insert(0,item) # add to the front
  
  def dequeue(self):
    return self.queue.pop()

  def size(self):
    return len(self.queue)


def partitionDisjoint(nums):
  n = len(nums)
  maxleft = [None]*n
  minright = [None]*n
  m = nums[0]
  for i in range(n):
    m = max(m,nums[i])
    maxleft[i] = m 
  

  m = nums[-1]
  for i in range(n-1,-1,-1):
    m = min(m,nums[i])
    minright[i]  = m 

  for i in range(1,n):
    if maxleft[i-1] <= minright[i]:
      return i 

def reverseWords2(s):

  def reverse_word_helper(left,right):
    while left < right:
      s[left],s[right] =s[right],s[right]
      left +=1
      right -=1

    reverse_word_helper(0,len(s)-1)

  left =0 
  for i,char in enumerate(s):
    if char == " ":
      reverse_word_helper(left,i-1)
      left = i+1
    reverse_word_helper(left,len(s)-1)





   

    
    





     


        







    
  



        

      








