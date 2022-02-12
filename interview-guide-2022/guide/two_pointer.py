"""
Two pointers is a common interview technique often used to solve certain problems involving an iterable data structure, such as an array. As the name suggests, this technique uses two (or more) pointers that traverses through the structure. It does not have to be physically using two pointers. As long as the other pointer can be easily calculated from existing values, such as the index of the other pointer, it counts as a two pointer question.

Since "two pointers" is kind of a broad topic, there is no singular way to implement it. Depending on the questions you encounter, you need to implement the answer differently. Generally speaking, a two pointer algorithm has these characteristics:

Two moving pointers, regardless of directions, moving dependently or independently;
A function that utilizes the entries pointing at the two pointers that relates to the answer in a way;
An easy way of deciding which pointer to move;
Optionally a way to process the array when the pointers are moved.

"""
from typing import List

class TwoPointerSolutions:

    def __init__(self):
        pass

    def removeDuplicateFromSortedArray(self,nums:List[int])->int:
        pass