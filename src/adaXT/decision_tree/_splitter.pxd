cimport numpy as cnp

cdef class test_obj:
    cdef:
        double crit
        list[:, :] idx_split
        double[2] imp
        double split_val

cdef class Splitter:
    cdef:
        double[:, ::1] features
        double[:] outcomes
        int n_features
        int[:, ::1] pre_sort
        int[:] indices
        int n_indices
        object criteria
        int n_class
        double* class_labels
        int* n_in_class

    cdef cnp.ndarray sort_feature(self, int[:], double[:])


    cdef (double, double, double, double) test_split(self, int[:], int[:], int)

    cpdef get_split(self, int[:])
    
    cpdef void make_c_lists(self, int)

    cpdef void free_c_lists(self)