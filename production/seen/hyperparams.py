from algorithms.RecSysAlgo import ItemsSVDAlgo
from controllers.Model import ItemSimilarityModel
from similarity_functions.SimFunc import CosineSimilarity

# parameters for querry settings
min_previews = 20

# parameters for item similarity model
zero_replacement = 1
k = 10
simfunc = CosineSimilarity

# set up of item similarity model
algo = ItemsSVDAlgo(replace_zero_by=zero_replacement)
model = ItemSimilarityModel(algo, k, simfunc)

# parameters for recommendations
scores_chart = {"0": 1, "1": 1}
