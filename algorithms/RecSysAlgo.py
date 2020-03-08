from copy import deepcopy
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd as ssvd
import numpy as np


class LowDimEmbeddingAlgo:

    def __init__(self):
        self.item_embeddings = None
        self.user_embeddings = None
        self.name = None

    def fit(self, data, k):
        raise NotImplementedError("fit(data, k) function must be implemented by inherited Algo "
                                  "object")


class ItemsSVDAlgo(LowDimEmbeddingAlgo):

    def __init__(self, replace_zero_by=-1):
        super().__init__()
        self.utility_matrix = None
        self.zero_replacement = replace_zero_by
        self.name = "ItemsSVDAlgo"

    def fit(self, RecSysData, k):

        values = deepcopy(RecSysData.is_selected.values)
        values[np.where(values == 0)] = self.zero_replacement
        rows = RecSysData.user_index.values
        columns = RecSysData.template_index.values
        self.utility_matrix = csc_matrix((values, (rows, columns)))

        _, _, ItemsEmbed = ssvd(self.utility_matrix, k)

        self.item_embeddings = ItemsEmbed


class ItemsUsersSVDAlgo(LowDimEmbeddingAlgo):

    def __init__(self, replace_zero_by=-1):
        super().__init__()
        self.utility_matrix = None
        self.zero_replacement = replace_zero_by
        self.name = "ItemsUsersSVDAlgo"

    def fit(self, RecSysData, k):

        values = deepcopy(RecSysData.is_selected.values)
        values[np.where(values == 0)] = self.zero_replacement
        rows = RecSysData.user_index.values
        columns = RecSysData.template_index.values
        self.utility_matrix = csc_matrix((values, (rows, columns)))

        UsersEmbed, _, ItemsEmbed = ssvd(self.utility_matrix, k)

        self.item_embeddings = ItemsEmbed
        self.user_embeddings = UsersEmbed


class RandomEmbeddingAlgo(LowDimEmbeddingAlgo):

    def __init__(self, seed=None):
        super().__init__()
        self.name = "RandomAlgo"
        if seed:
            # set seed for random mechanism
            pass

    def fit(self, RecSysData, k):

        num_items = RecSysData.template_index.max() + 1
        self.embeddings = 2 * np.random.rand(k, num_items) - 1


available_algorithms = [ItemsSVDAlgo, RandomEmbeddingAlgo]


if __name__ == "__main__":
    import pandas as pd
    alg = ItemsSVDAlgo
    data = pd.read_csv("../data/RecSys data/min_previews 6.csv")
    algo = alg()
    K = 10
    algo.fit(data, K)
    print(algo.embeddings.shape)
