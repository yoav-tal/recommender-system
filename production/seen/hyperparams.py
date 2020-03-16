from algorithms.RecSysAlgo import ItemsSVDAlgo
from controllers.Model import ItemSimilarityModel
from similarity_functions.SimFunc import CosineSimilarity

# paths
query_path = "../../data/queries/preview_selections_seen - BigQuery"  # path to query text file
data_path = "../../data/query data/seen/"  # folder to save query data
processed_data_path = "../../data/processing/seen/"  # folder to save processed data
similarities_filename = "testFile_get_similarities"

# parameters for item similarity model
zero_replacement = 1
k = 10
simfunc = CosineSimilarity

# set up of item similarity model
algo = ItemsSVDAlgo(replace_zero_by=zero_replacement)
model = ItemSimilarityModel(algo, k, simfunc)

# parameters for recommendations
scores_chart = {"0": 1, "1": 1}
