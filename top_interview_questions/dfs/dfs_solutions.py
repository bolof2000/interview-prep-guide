class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.right = right
        self.left = left

class DFSSolutions:

    def isSameTree(self,p:TreeNode,q:TreeNode)->bool:


        def dfs(left:TreeNode,right:TreeNode):
            if left is None and right is None:
                return True

            if left is None or right is None or left.val != right.val:
                return False

            return dfs(left.left,right.left) and dfs(left.right,right.right)

        return dfs(p,q)
