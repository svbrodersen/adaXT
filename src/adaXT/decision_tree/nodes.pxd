cimport numpy as cnp
cimport cython
cdef class Node:
    cdef public:
        int parent
        cnp.ndarray indices
        int depth
        double impurity
        bint visited
        bint is_leaf

cdef class DecisionNode(Node):
    cdef public:
        int left_child
        int right_child
        double threshold
        int split_idx

cdef class LeafNode(Node):
    cdef public:
        double weighted_samples
        int id
        cnp.ndarray value

cdef class LocalPolynomialLeafNode(LeafNode):
    cdef public:
        double theta0, theta1, theta2
