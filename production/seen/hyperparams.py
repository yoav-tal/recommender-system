from algorithms.RecSysAlgo import ItemsSVDAlgo
from controllers.Model import ItemSimilarityModel
from similarity_functions.SimFunc import CosineSimilarity

min_previews = 20

zero_replacement = 1
k = 10
simfunc = CosineSimilarity

algo = ItemsSVDAlgo(replace_zero_by=zero_replacement)
model = ItemSimilarityModel(algo, k, simfunc)
