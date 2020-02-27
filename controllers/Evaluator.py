import numpy as np
from recommendation.evaluation_module import EvaluationModule
from controllers.data_handler import DataHandler
from controllers.Viewer import ThumbnailViewer


class Evaluator:

    def __init__(self, DataHandler_instance):

        self.data_handler = DataHandler_instance
        self.evaluation_module = EvaluationModule(self.data_handler)
        self.models = []
        self.viewer = ThumbnailViewer(self.data_handler)

    def add_model(self, model, name):
        self.models.append((model, name))

    def evaluate(self, find_nearest_to=[], n_nearest_to=10, print_nearest=True,
                 view_nearest=False, print_category_match_index=False,
                 calc_match_index=False, n_for_match_index=10,
                 get_recom_for=[], n_for_recom=10,
                 recom_scores_chart=None, print_recom=True, view_recom=False,
                 print_recom_category_match_index=False, calc_hitRate=False, n_for_hitRate=10,
                 show_hit_position_index=False, calc_novelty_score=False, n_for_novelty=10):

        if find_nearest_to:
            self.find_nearest_to(find_nearest_to, n_nearest_to, print_nearest, view_nearest,
                                 print_category_match_index)

        if calc_match_index:
            self.calculate_match_index(n_for_match_index)

        if get_recom_for:
            self.recommend(get_recom_for, n_for_recom, recom_scores_chart, print_recom,
                           view_recom, print_recom_category_match_index)

        if calc_hitRate:
            self.calculate_hit_rate(n_for_hitRate, recom_scores_chart, show_hit_position_index)

        if calc_novelty_score:
            self.calculate_novelty_score(n_for_novelty, recom_scores_chart)

    def calculate_hit_rate(self, n, scores_chart, show_hit_position_index):
        LOO_data, left_out = self.data_handler.prepare_LOO_dataset()

        users_list = self.data_handler.getUserIds(dataset="LOO")
        print("number of users in dataset: %g" % len(users_list))

        for model, model_name in self.models:
            similarities = model.GetSimilarities(LOO_data)
            recomms = self.evaluation_module.get_recommendations_for_users(
                users_list, similarities, n, scores_chart, dataset="LOO")
            model_hit_rate = self.evaluation_module.calc_hit_rate(recomms, left_out)
            print("Hit Rate of model %s with %g recommendations per user: %g\n" % (model_name, n,
                                                                                   model_hit_rate))

        if show_hit_position_index:
            # for each user, find the position in the ranked items of the left-out item,
            # and return the min, Q1, median, Q3 and max
            pass

    def calculate_match_index(self, n):
        # For each model, get n nearest neighbors for all items and calc match index (mean match
        # over items) of the model
        match_indices = {}
        data = self.data_handler.data
        n_items = self.data_handler.getNumItems()
        for model, model_name in self.models:
            indices = []
            similarities = model.GetSimilarities(data)
            nearest_neighbors, distances = \
                self.evaluation_module.get_item_neighbors(list(range(n_items)), similarities, n)
            for item in range(n_items):
                nearest_to_item = nearest_neighbors[item]
                indices.append(self.evaluation_module.neighbor_category_match(item,
                                                                              nearest_to_item))
            match_indices[model_name] = indices
            print("\nModel: %s \tmean index: %g" %
                  (model_name, np.mean(match_indices[model_name])))

    def calculate_novelty_score(self, n, scores_chart):

        data = self.data_handler.data

        users_list = self.data_handler.getUserIds()
        print("number of users in dataset: %g" % len(users_list))

        for model, model_name in self.models:
            similarities = model.GetSimilarities(data)
            recomms = self.evaluation_module.get_recommendations_for_users(
                users_list, similarities, n, scores_chart)
            model_novelty_score = self.evaluation_module.calc_novelty_score(recomms, data)
            print("Novely score model %s with %g recommendations per user: %g\n" %
                  (model_name, n, model_novelty_score))

    def find_nearest_to(self, find_nearest_to, n_nearest_to, print_nearest, view_nearest,
                        print_category_match_index):
        nearestToItem = {}
        data = self.data_handler.data
        for model, model_name in self.models:
            similarities = model.GetSimilarities(data)
            nearest_neighbors, distances = \
                self.evaluation_module.get_item_neighbors(find_nearest_to, similarities,
                                                          n_nearest_to)
            nearestToItem[model_name] = \
                {"items": self.data_handler.getItemName(nearest_neighbors),
                 "distances": distances}

        examined_items = self.data_handler.getItemName([find_nearest_to])
        if len(find_nearest_to) > 1:
            examined_items = examined_items.squeeze()
        if len(find_nearest_to) == 1:
            examined_items = np.array([examined_items[0][0]])

        for i, item in enumerate(examined_items):
            # print("nearest to item %s:" % item)
            for _, model_name in self.models:
                item_neighbors = nearestToItem[model_name]["items"][i]
                print("model %s:" % model_name)
                if print_nearest:
                    self.viewer.print_recommendations(
                        [item], item_neighbors, nearestToItem[model_name]["distances"][i])
                if print_category_match_index:
                    match_index = self.evaluation_module.neighbor_category_match(item,
                                                                                 item_neighbors)
                    print("category match index for item %s: %g" % (item, match_index))

                if view_nearest:
                    self.viewer.view_thumbnails([item], item_neighbors)
            print("\n\n")

    def recommend(self, users_ids, n_for_recom, scores_chart, print_recom, view_recom,
                  print_recom_category_match_index):
        recommendations = {}
        data = self.data_handler.data
        for model, model_name in self.models:
            similarities = model.GetSimilarities(data)
            recomms = self.evaluation_module.get_recommendations_for_users(
                users_ids, similarities, n_for_recom, scores_chart)
            recommendations[model_name] = {"items": self.data_handler.getItemName(recomms)}

        for i, user in enumerate(users_ids):
            for _, model_name in self.models:
                user_recomms = recommendations[model_name]["items"][i]
                print("model %s:" % model_name)
                if print_recom:
                    self.viewer.print_user_recommendations(user, user_recomms)

                if view_recom:
                    selected, unselected = self.data_handler.getUserActivitySplit(user)
                    self.viewer.view_thumbnails(np.squeeze(selected), np.squeeze(unselected),
                                                user_recomms)


if __name__ == "__main__":
    from algorithms.RecSysAlgo import SparseSVDAlgo, RandomEmbeddingAlgo
    from similarity_functions.SimFunc import CosineSimilarity
    from controllers.Model import Model, PopularityModel
    import pandas as pd

    #data_specs = {"project_name": "boost", "data_name": "min_previews 5 - ch"}#, "variants": ["",
                                                                                #            "ch",
                                                                                #         "ja"]}

    data_specs = {"project_name": "story", "data_name": "min_previews 20"}
    dh = DataHandler(data_specs)
    E = Evaluator(dh)

    # establish SVD algorithm
    algo = SparseSVDAlgo(replace_zero_by=-1)
    simfunc = CosineSimilarity
    k = 20
    model_1 = Model(algo, k, simfunc)
    #E.add_model(model_1, "SparseSVD_k20_cosine")

    algo = SparseSVDAlgo(replace_zero_by=1)
    simfunc = CosineSimilarity
    k = 20
    model_2 = Model(algo, k, simfunc)
    E.add_model(model_2, "SparseSVD_k20_cosine_forclicks")

    # establish random recommendation algorithm
    rand_algo = RandomEmbeddingAlgo()
    model_rand = Model(rand_algo, k, simfunc)
    # E.add_model(model_rand, "Random_k20_seedDefault")

    popularity_model = PopularityModel()
    # E.add_model(popularity_model, "PopularityModel")

    E.evaluate(find_nearest_to=list(np.random.randint(149, size=10)), n_nearest_to=6,
                  print_nearest=True, view_nearest=True, print_category_match_index=True)  # list(
    # np.random.randint(625, size=15))
    # E.evaluate(find_nearest_to=list(range(64, 72)), n_nearest_to=6, view_nearest=True,
    #            print_category_match_index=True, calc_match_index=True)
    #E.evaluate(calc_match_index=True, n_for_match_index=6)
    # E.evaluate(get_recom_for=[0, 1, 2, 3, 4], n_for_recom=6,
    #           recom_scores_chart={"0": 1, "1": 1}, view_recom=True)
    # E.evaluate(calc_hitRate=True, n_for_hitRate=6, recom_scores_chart={"0": 1, "1": 1})
    # E.evaluate(calc_novelty_score=True, n_for_novelty=6, recom_scores_chart={"0": 1, "1": 1})
