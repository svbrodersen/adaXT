# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

# General
import numpy as np
from numpy import float64 as DOUBLE
import sys

# Custom
from .splitter import Splitter
from ..criteria import Criteria
from .DepthTreeBuilder import DepthTreeBuilder
from .Nodes import DecisionNode

cdef double EPSILON = np.finfo('double').eps


class DecisionTree:
    def __init__(
            self,
            tree_type: str,
            criteria: Criteria,
            max_depth: int = sys.maxsize,
            impurity_tol: float = 0,
            min_samples_split: int = 1,
            min_samples_leaf: int = 1,
            min_improvement: float = 0,
            splitter: Splitter | None = None) -> None:

        tree_types = ["Classification", "Regression"]
        assert tree_type in tree_types, f"Expected Classification or Regression as tree type, got: {tree_type}"
        self.max_depth = max_depth
        self.impurity_tol = impurity_tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_improvement = min_improvement
        self.criteria = criteria
        self.tree_type = tree_type
        self.leaf_nodes = None
        self.root = None
        self.n_nodes = -1
        self.n_features = -1
        self.n_classes = -1
        self.n_obs = -1
        self.classes = None
        self.splitter = splitter

    def __check_input(self, X: object, Y: object):
        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        Y = np.ascontiguousarray(Y, dtype=DOUBLE)

        # Check if X and Y has same number of rows
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y should have the same number of rows")

        # Check if Y has dimensions (n, 1) or (n,)
        if 2 < Y.ndim:
            raise ValueError("Y should have dimensions (n,1) or (n,)")
        elif 2 == Y.ndim:
            if 1 < Y.shape[1]:
                raise ValueError("Y should have dimensions (n,1) or (n,)")
            else:
                Y = Y.reshape(-1)

        return X, Y

    def __check_sample_weight(self, sample_weight: np.ndarray, n_samples):

        if sample_weight is None:
            return np.ones(n_samples, dtype=np.double)
        sample_weight = np.array(sample_weight, dtype=np.double)
        if sample_weight.shape[0] != n_samples:
            raise ValueError("sample_weight should have as many elements as X and Y")
        if sample_weight.ndim > 1:
            raise ValueError("sample_weight should have dimension (n_samples,)")
        return sample_weight

    def __check_dimensions(self, double[:, :] X):
        X = np.ascontiguousarray(X, dtype=DOUBLE)
        # If there is only a single point
        if X.ndim == 1:
            if (X.shape[0] != self.n_features):
                raise ValueError(f"Number of features should be {self.n_features}, got {X.shape[0]}")
        else:
            if X.shape[1] != self.n_features:
                raise ValueError(f"Dimension should be {self.n_features}, got {X.shape[1]}")
        return X

    def fit(
            self,
            X,
            Y,
            sample_indices: np.ndarray | None = None,
            feature_indices: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,) -> None:

        X, Y = self.__check_input(X, Y)
        row, col = X.shape

        # If sample_weight is valid it is simply passed through check_sample_weight, if it is None all entries are set to 1
        sample_weight = self.__check_sample_weight(sample_weight=sample_weight, n_samples=row)

        if feature_indices is None:
            feature_indices = np.arange(col, dtype=np.int32)
        builder = DepthTreeBuilder(
            X=X,
            Y=Y,
            sample_indices=sample_indices,
            feature_indices=feature_indices,
            sample_weight=sample_weight,
            criteria=self.criteria(X, Y, sample_weight),
            splitter=self.splitter)
        builder.build_tree(self)

    def predict(self, double[:, :] X):
        cdef:
            int i, cur_split_idx, idx
            double cur_threshold
            int row = X.shape[0]
            double[:] Y = np.empty(row)
            object cur_node
        if not self.root:
            raise AttributeError("The tree has not been fitted before trying to call predict")

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)

        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if self.tree_type == "Regression":
                Y[i] = cur_node.value[0]
            elif self.tree_type == "Classification":
                idx = self.__find_max_index(cur_node.value)
                if self.classes is not None:
                    Y[i] = self.classes[idx]
        return Y

    def predict_proba(self, double[:, :] X):
        cdef:
            int i, cur_split_idx
            double cur_threshold
            int row = X.shape[0]
            object cur_node
            list ret_val = []

        if not self.root:
            raise AttributeError("The tree has not been fitted before trying to call predict_proba")

        if self.tree_type != "Classification":
            raise ValueError("predict_proba can only be called on a Classification tree")

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)

        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            if self.classes is not None:
                ret_val.append(cur_node.value)

        return np.asarray(ret_val)

    def __find_max_index(self, lst):
        cur_max = 0
        for i in range(1, len(lst)):
            if lst[cur_max] < lst[i]:
                cur_max = i
        return cur_max

    def get_leaf_matrix(self, scale: bool = False) -> np.ndarray:
        if not self.root:
            raise ValueError("The tree has not been trained before trying to predict")

        leaf_nodes = self.leaf_nodes
        n_obs = self.n_obs

        matrix = np.zeros((n_obs, n_obs))
        if (not leaf_nodes):  # make sure that there are calculated observations
            return matrix
        for node in leaf_nodes:
            if scale:
                n_node = node.indices.shape[0]
                matrix[np.ix_(node.indices, node.indices)] = 1/n_node
            else:
                matrix[np.ix_(node.indices, node.indices)] = 1

        return matrix

    def predict_leaf_matrix(self, double[:, :] X, scale: bool = False):
        cdef:
            int i
            int row
            dict ht
            int cur_split_idx
            double cur_threshold

        if not self.root:
            raise ValueError("The tree has not been trained before trying to predict")

        # Make sure that x fits the dimensions.
        X = self.__check_dimensions(X)
        row = X.shape[0]
        ht = {}
        for i in range(row):
            cur_node = self.root
            while isinstance(cur_node, DecisionNode):
                cur_split_idx = cur_node.split_idx
                cur_threshold = cur_node.threshold
                if X[i, cur_split_idx] < cur_threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child

            if cur_node.id not in ht.keys():
                ht[cur_node.id] = [i]
            else:
                ht[cur_node.id] += [i]
        matrix = np.zeros((row, row))
        for key in ht.keys():
            indices = ht[key]
            val = 1
            count = len(indices)
            if scale:
                val = 1/count
            matrix[np.ix_(indices, indices)] = val

        return matrix