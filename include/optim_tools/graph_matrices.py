import numpy as np
from scipy.io import loadmat
from include.load_data import loadingGraph


def create_transpose_incidence_matrix(graphDepMatrix):
    """
    Returns the transposed incidence matrix of any adjacency matrix that has mandatory 1 on the diagonal.
    :param graphDepMatrix: modified adjacency matrix to have 1 on the diagonal
    :return:
    """
    nbDep, mDep = np.shape(graphDepMatrix)  # here nbDep = mDep and should be 2 less than total number of 'départements'
    theoreticalEdges = max(np.shape(np.where(graphDepMatrix == -1)))  # assuming there are + edges than nodes
    GTV_op = np.zeros((max(np.shape(np.where(graphDepMatrix == -1))), nbDep))  # assuming there are + edges than nodes
    edges = 0  # will browse all rows of GTV_op and count the edges
    for row in range(0, nbDep):
        # np.where returns a tuple in which there is an array (for vector use)
        indP1 = np.where(graphDepMatrix[row] == 1)  # browsing through each 'département' once as "starting point"
        indN1 = np.where(graphDepMatrix[row] == -1)[0]  # browsing through each 'département' linked to the previous one
        for j in range(0, max(np.shape(indN1))):  # assuming there are + edges than nodes
            GTV_op[edges, indP1] = 1
            GTV_op[edges, indN1[j]] = -1
            edges += 1
    assert (edges == theoreticalEdges)

    return GTV_op


def create_transpose_incidence_dpt(graphDepMatrix=None):
    """
    Returns the transposed incidence matrix of the contiguity modified adjacency matrix 'depCont', or any sub matrix.
    :param graphDepMatrix: submatrix of a matrix of shape (96, 96) of 1 and -1, giving the links between 'départements'.
    Computes the matrix GTV_op, which is the **transposed incidence matrix**.
    From the matrix graphDepMatrix, GTV_op is the Total Variation operator s.t. for each edge (d1, d2) of the B_matrix,
    the column associated to d1 contains 1 and the one associated to d2 contains -1.

    For now, the computation is suited for a simple contiguity matrix 'departementsContigus.mat' with a defined format.
    :return: GTV_op, ndarray of shape (number of edges, number of 'départements')
    """
    if graphDepMatrix is None:
        departementsContigus = loadmat('include/load_data/departement_contigus.mat')
        graphDepMatrix = departementsContigus['matrice']  # french for matrix
        graphDepMatrix[-1, -1] = 1  # correction of a mistake : every dep verifies depContMatrix[i, i] = 1

    # Corsica is in rows and columns of indexes 28 and 29 from graphDepMatrix & will be added at the end of GTV_op
    graphDepNoCors = np.concatenate((graphDepMatrix[:28], graphDepMatrix[30:]))
    graphDepNoCors = np.concatenate((graphDepNoCors[:, :28], graphDepNoCors[:, 30:]), axis=1)

    nbDep, mDep = np.shape(graphDepNoCors)  # here nbDep = mDep and should be 2 less than total number of 'départements'
    theoreticalEdges = max(np.shape(np.where(graphDepNoCors == -1)))  # assuming there are + edges than nodes
    GTV_op = np.zeros((max(np.shape(np.where(graphDepNoCors == -1))), nbDep))  # assuming there are + edges than nodes
    edges = 0  # will browse all rows of GTV_op and count the edges
    for row in range(0, nbDep):
        # np.where returns a tuple in which there is an array (for vector use)
        indP1 = np.where(graphDepNoCors[row] == 1)  # browsing through each 'département' once as "starting point"
        indN1 = np.where(graphDepNoCors[row] == -1)[0]  # browsing through each 'département' linked to the previous one
        for j in range(0, max(np.shape(indN1))):  # assuming there are + edges than nodes
            GTV_op[edges, indP1] = 1
            GTV_op[edges, indN1[j]] = -1
            edges += 1
    assert(edges == theoreticalEdges)

    # Adding Corsica to the GTV_op matrix : only contiguous to each other
    Gtmp = np.copy(GTV_op)  # of shape (edges, nbDep)
    GTV_op = np.zeros((edges+2, nbDep+2))  # adding 2 rows and 2 columns
    GTV_op[:edges, :nbDep] = Gtmp
    GTV_op[edges, nbDep] = 1
    GTV_op[edges+1, nbDep+1] = 1
    GTV_op[edges, nbDep+1] = -1
    GTV_op[edges+1, nbDep] = -1

    # Checking if the GTV_op matrix is the same as in MATLAB's code
    # testG = scipy.io.loadmat('data/Gmatrixtest.mat')
    # Gmatlab = testG['GTV_op']
    # print(np.where(GTV_op != Gmatlab))

    return GTV_op


def incidenceMatrix_deptCont(deps='all'):
    """
    Incidence matrix B of shape (nodes, territories), sorted by "département code" with Corsica's '2A' and '2B' at the end.
    When chosenDeps='all', should compute the result of create_transpose_incidence_dpt() *transposed*.
    Else, it will compute the incidence matrix of the subgraph composed with nodes labelled in chosenDeps.
    :return: ndarray of shape (vertices, edges) -- usually |vertices| == deps
    """

    # deps = ['64', '40', '33', '17', '2A', '79']
    depCont, labels = loadingGraph.departementsAdjMatrix()

    # Corsica is in rows and columns of indexes 28 and 29 from depCont and will be put at the end.
    depContNoCorsRows = np.concatenate((depCont[:28], depCont[30:]))
    depContNoCorsica = np.concatenate((depContNoCorsRows[:, :28], depContNoCorsRows[:, 30:]), axis=1)
    labelsNoCorsica = np.array(list(labels)[:28] + list(labels)[30:])

    if deps == 'all':
        Corsica2A = True
        Corsica2B = True
        depContCrop = depContNoCorsica
    else:
        chosenDeps = np.array(deps)
        if '2A' in deps:
            Corsica2A = True
            index2A = np.where(chosenDeps == '2A')[0][0]
            chosenDeps = np.concatenate((chosenDeps[:index2A], chosenDeps[index2A + 1:]))
        else:
            Corsica2A = False
        if '2B' in deps:
            Corsica2B = True
            index2B = np.where(chosenDeps == '2B')[0]
            chosenDeps = np.concatenate((chosenDeps[:index2B], chosenDeps[index2B + 1:]))
        else:
            Corsica2B = False

        chosenIndexes = np.sort(np.array([np.where(labelsNoCorsica == chosenDeps[k])[0][0]
                                          for k in range(len(chosenDeps))]))  # Sorting indexes !
        depContCropRows = depContNoCorsica[chosenIndexes]
        depContCrop = depContCropRows[:, chosenIndexes]

    nbDep, mDep = np.shape(depContCrop)
    assert (nbDep == mDep)

    theoreticalEdges = max(np.shape(np.where(depContCrop == -1)))  # assuming there are + edges than nodes
    B_matrix = np.zeros((theoreticalEdges, nbDep))  # assuming there are + edges than nodes
    edges = 0  # will browse all rows of B_matrix and count the edges
    for row in range(0, nbDep):
        # np.where returns a tuple in which there is an array (for vector use)
        indP1 = np.where(depContCrop[row] == 1)  # browsing through each 'département' once as "starting point"
        indN1 = np.where(depContCrop[row] == -1)[0]  # browsing through each 'département' linked to the previous one
        for j in range(0, max(np.shape(indN1))):  # assuming there are + edges than nodes
            B_matrix[edges, indP1] = 1
            B_matrix[edges, indN1[j]] = -1
            edges += 1
    assert (edges == theoreticalEdges)

    # Adding Corsica to the B_matrix matrix : only contiguous to each other
    if Corsica2A and Corsica2B:
        Btmp = np.copy(B_matrix)  # of shape (edges, nbDep)
        B_matrix = np.zeros((edges + 2, nbDep + 2))  # adding 2 rows and 2 columns
        B_matrix[:edges, :nbDep] = Btmp
        B_matrix[edges, nbDep] = 1
        B_matrix[edges + 1, nbDep + 1] = 1
        B_matrix[edges, nbDep + 1] = -1
        B_matrix[edges + 1, nbDep] = -1
    elif Corsica2A or Corsica2B:  # not connected to any other département
        Btmp = np.copy(B_matrix)  # of shape (edges, nbDep)
        B_matrix = np.zeros((edges + 1, nbDep + 1))  # adding 2 rows and 2 columns
        B_matrix[:edges, :nbDep] = Btmp
        B_matrix[edges, nbDep] = 1

    return np.transpose(B_matrix)


def laplacianMatrix_deptCont(deps='all'):
    """
    Laplacian matrix L = B.B^T of shape (deps, deps).
    :return:
    """
    return np.dot(incidenceMatrix_deptCont(deps), np.transpose(incidenceMatrix_deptCont(deps)))

# depCont, labels = loadingGraph.departementsAdjMatrix()
# A = create_transpose_incidence_dpt(depCont)
# B = incidenceMatrix_deptCont(deps='all')
# assert(np.max(np.abs(A - np.transpose(B))) == 0)
