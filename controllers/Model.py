import numpy as np


class ItemSimilarityModel:
    # The Model class abstracts the idea that a model is the combination of an algorithm that
    # learns low-rank embeddings, a dimension k of the embeddings and a similarity metric. The
    # model controls the calls to the algorithm’s fit() functions and computes the similarity
    # matrix from the embeddings.
    #
    # An instance is therefore initiated with a RecSysAlgo object, an integer k and a SimFunc
    # object. The Model class hosts a get_similarities function and a get_embeddings function
    # (both take data as argument).

    def __init__(self, algo, k, SimFunc):
        self.algo = algo
        self.k = k
        self.sim_func = SimFunc(k)
        self.embeddings = None
        self.embeddings_of = None
        self.similarities = None
        self.similarities_of = None

    def GetEmbeddings(self, data):
            if self.embeddings_of is not None and data.shape == self.embeddings_of.shape and \
                    (data.values == self.embeddings_of.values).all():
                return self.embeddings
            else:
                print("fitting algorithm %s with k=%g" % (self.algo.name, self.k))
                self.algo.fit(data, self.k)
                self.embeddings = self.algo.item_embeddings
                self.embeddings_of = data
                return self.embeddings

    def GetSimilarities(self, data):
        if self.similarities_of is not None and data.shape == self.similarities_of.shape and \
                (data.values == self.similarities_of.values).all():
            return self.similarities
        else:
            print("calculating similarities by algorithm %s with k=%g and similarity function "
                  "%s" % (self.algo.name, self.k, self.sim_func.name))
            self.similarities = self.sim_func(self.GetEmbeddings(data))
            self.similarities_of = data
            return self.similarities


class LowDimEmbeddingModel:
    # The Model class hosts a get_item_similarities function, get_item_embeddings and
    # get_user_embeddings functions (both take data as argument).

    def __init__(self, algo, k, SimFunc):
        self.algo = algo
        self.k = k
        self.sim_func = SimFunc(k)
        self.item_embeddings = None
        self.user_embeddings = None
        self.embeddings_of = None
        self.item_similarities = None
        self.similarities_of = None

    def GetEmbeddings(self, data):
        print("fitting algorithm %s with k=%g" % (self.algo.name, self.k))
        self.algo.fit(data, self.k)
        self.item_embeddings = self.algo.item_embeddings
        self.user_embeddings = self.algo.user_embeddings
        self.embeddings_of = data

    def GetItemEmbeddings(self, data):
        if self.embeddings_of is not None and data.shape == self.embeddings_of.shape and \
                (data.values == self.embeddings_of.values).all():
            return self.item_embeddings
        else:
            self.GetEmbeddings(data)
            return self.item_embeddings

    def GetUserEmbeddings(self, data):
        if self.embeddings_of is not None and data.shape == self.embeddings_of.shape and \
                (data.values == self.embeddings_of.values).all():
            return self.user_embeddings
        else:
            self.GetEmbeddings(data)
            return self.user_embeddings

    def GetItemSimilarities(self, data):
        if self.similarities_of is not None and data.shape == self.similarities_of.shape and \
                (data.values == self.similarities_of.values).all():
            return self.item_similarities
        else:
            print("calculating similarities by algorithm %s with k=%g and similarity function "
                  "%s" % (self.algo.name, self.k, self.sim_func.name))
            self.item_similarities = self.sim_func(self.GetItemEmbeddings(data))
            self.similarities_of = data
            return self.item_similarities


if __name__ == "__main__":
    from algorithms.RecSysAlgo import ItemsSVDAlgo
    from similarity_functions.SimFunc import CosineSimilarity
    import pandas as pd
    alg = ItemsSVDAlgo
    simfunc = CosineSimilarity
    data = pd.read_csv("../data/RecSys data/min_previews 6.csv")

    algo = alg()
    k = 10
    model = Model(algo, k, simfunc)

    print(model.GetEmbeddings(data).shape)
    print(model.GetSimilarities(data).shape)
    print(model.GetSimilarities(data).shape)
