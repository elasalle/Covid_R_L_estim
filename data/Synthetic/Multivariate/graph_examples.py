import numpy as np


def example_choice(example):
    """
    :param example: str between 'Line', 'T-form', 'Sun', 'Caterpillar', 'Fly', 'Frog', 'Butterfly' and 'Flower'.
    :return: depCont : ndarray of shape (5, 5) containing 1 and -1 to indicate edges of the associated network
             pos : dictionary {k: (i, j)} where k in range(5) and (i, j) euclidian position for displaying B_matrix.

    """
    if example == 'Line':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1,  0],
                            [ 0,  0, -1,  1, -1],
                            [ 0,  0,  0, -1,  1]])

    elif example == 'T-form':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1,  0],
                            [ 0,  0, -1,  0,  1]])
    elif example == 'Sun':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1,  0,  0],
                            [ 0, -1,  0,  1,  0],
                            [ 0, -1,  0,  0,  1]])
    elif example == 'Caterpillar':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1, -1],
                            [ 0,  0, -1, -1,  1]])
    elif example == 'Fly':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0, -1],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1,  0],
                            [ 0, -1, -1,  0,  1]])
    elif example == 'Frog':
        depCont = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1, -1,  0],
                            [ 0, -1, -1,  1,  0],
                            [ 0, -1,  0,  0,  1]])
    elif example == 'Butterfly':
        depCont = np.array([[ 1, -1,  0,  0, -1],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1, -1,  0],
                            [ 0, -1, -1,  1,  0],
                            [-1,  0,  0, -1,  1]])
    elif example == 'Flower':
        depCont = np.array([[ 1, -1, -1,  0, -1],
                            [-1,  1, -1, -1, -1],
                            [-1, -1,  1, -1,  0],
                            [ 0, -1, -1,  1, -1],
                            [-1, -1,  0, -1,  1]])

    else:
        ExampleError = ValueError("See data/Synthetic/Multivariate/graph_examples.py for available examples.")
        raise ExampleError

    depsIndexes = range(5)
    # Displaying positions ----
    pos = {}
    if example == 'Line':
        for k in range(5):
            pos[depsIndexes[k]] = (k, 0)
    elif example == 'T-form':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0.5)
        pos[depsIndexes[4]] = (3, -0.5)
    elif example == 'Sun':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (1, 1)
        pos[depsIndexes[4]] = (1, -1)
    elif example == 'Caterpillar':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0.5)
        pos[depsIndexes[4]] = (3, -0.5)
    elif example == 'Fly':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0)
        pos[depsIndexes[4]] = (1.5, -1)
    elif example == 'Frog':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    elif example == 'Butterfly':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    elif example == 'Flower':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    else:
        ExampleError = ValueError("See data/Synthetic/Multivariate/graph_examples.py for available examples.")
        raise ExampleError
    
    return depCont, pos
