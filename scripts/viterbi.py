import numpy as np

def print_matrix(matrix):

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print("{}".format(matrix[i][j]), end = '\t')
        print()

def viterbi(sequence, initial_probs, obs_map, t, e):

    v_matrix = np.zeros((len(initial_probs), len(sequence)))
    backtracking_matrix = np.copy(v_matrix)

    # v_matrix = [[0] * len(sequence)] * len(initial_probs)
    # backtracking_matrix = [[0] * len(sequence)] * len(initial_probs)

    for i in range(len(initial_probs)):
        v_matrix[i, 0] = initial_probs[0] * t[i][obs_map[sequence[0]]]
        backtracking_matrix[i, 0] = -1

    for j in range(1, len(sequence)):
        for i in range(len(initial_probs)):
            for k in range(len(initial_probs)):

                curr_cell = v_matrix[k][j - 1] * t[k][i] * e[i][obs_map[sequence[j]]]
                if curr_cell > v_matrix[i][j]:
                    v_matrix[i][j] = curr_cell
                    backtracking_matrix[i][j] = k
                # v_matrix[i][j] = max(v_matrix[k][j - 1] * t[k][i] * e[i][obs_map[sequence[j]]], 
                #                      v_matrix[i][j])
                # backtracking_matrix[i][j] = np.argmax([])
    
    print("V Matrix: ")
    print_matrix(v_matrix)

    print("Backtracking matrix: ")
    print_matrix(backtracking_matrix)

    
if __name__ == "__main__":

    # init = [0.5, 0.5]

    # obs_mapping = {
    #     'A': 0, 
    #     'C': 1, 
    #     'G': 2, 
    #     'T': 3
    # }

    # transition = [
    #     [0.8, 0.2], 
    #     [0.2, 0.8]
    # ]

    # emission = [
    #     [0.3, 0.2, 0.3, 0.2], 
    #     [0.1, 0.4, 0.1, 0.4]
    # ]
    obs_space = ["mary", "jane", "will", "spot", "can", "see", "pat"]
    obs_mapping = {key: value for value, key in enumerate(obs_space)}

    init = [0.75, 0.25, 0]

    transition = [
        [1/9, 1/3, 1/9, 4/9], 
        [0.25, 0, 0.75, 0], 
        [1, 0, 0, 0]
    ]

    emission = [
        [4/9, 2/9, 1/9, 2/9, 0, 0, 0], 
        [0, 0, 3/4, 0, 1/4, 0, 0], 
        [0, 0, 0, 1/4, 0, 1/2, 1/4]
    ]

    viterbi("jane will spot will".split(), init, obs_mapping, transition, emission)