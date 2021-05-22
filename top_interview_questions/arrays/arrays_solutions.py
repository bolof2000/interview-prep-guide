from typing import List


class ArraySolutions:

    def plusOne(self, digits: List[int]) -> List[int]:

        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1

                return digits

        return [1] + digits  # in case we have all 9, we need to increase array size by 1 and fill it with 1


test = ArraySolutions()
print(test.plusOne([1,0]))