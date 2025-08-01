import numpy as np


def calculate_index(num, div):
    original_list = [num // div + (1 if x < num % div else 0) for x in range(div)]
    new_list = [original_list[0]]
    for i in range(1, len(original_list)):
        new_element = original_list[i] + new_list[i - 1]
        new_list.append(new_element)
    new_list.insert(0, 1)
    new_list = np.array(new_list) - 1
    return new_list
