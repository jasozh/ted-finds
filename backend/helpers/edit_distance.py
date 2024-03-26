import numpy as np


def edit_distance(query, text, del_cost=1, ins_cost=1, sub_cost=2) -> float:
    arr = np.zeros((len(query) + 1, len(text) + 1))
    for i in range(len(query) + 1):
        arr[i, 0] = i
    for j in range(len(text) + 1):
        arr[0, j] = j

    for i in range(1, len(query) + 1):
        for j in range(1, len(text) + 1):
            if query[i - 1] == text[j - 1]:
                arr[i, j] = arr[i - 1, j - 1]
            else:
                arr[i, j] = min(
                    arr[i - 1, j] + del_cost,
                    arr[i, j - 1] + ins_cost,
                    arr[i - 1, j - 1] + sub_cost
                )
    return arr[len(query), len(text)]
