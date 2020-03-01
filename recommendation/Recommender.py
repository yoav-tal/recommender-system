import numpy as np

from controllers.Model import ItemSimilarityModel
from recommendation.evaluation_module import EvaluationModule


class Recommender:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.evaluation_module = EvaluationModule(data_handler)

    def calc_hit_rate(self, n):
        LOO_data, left_out = self.data_handler.prepare_LOO_dataset()

        users_list = self.data_handler.getUserIds(dataset="LOO")
        print("number of users in LOO dataset: %g" % len(users_list))

        recommendations = self.recommend_for_users(users_list, n, dataset="LOO")
        hit_rate = self.evaluation_module.calc_hit_rate(recommendations, left_out)
        return hit_rate

    def calc_match_index(self, n):
        indices = []
        n_items = self.data_handler.getNumItems()
        nearest_neighbors, _ = self.find_nearest_items(list(range(n_items)), n)

        for item in range(n_items):
            nearest_to_item = nearest_neighbors[item]
            indices.append(self.evaluation_module.neighbor_category_match(item,
                                                                          nearest_to_item))
        return np.mean(indices)

    def calc_novelty_score(self, n):
        users_list = self.data_handler.getUserIds()
        recommendations = self.recommend_for_users(users_list, n)
        novelty_score = self.evaluation_module.calc_novelty_score(recommendations)
        return novelty_score

    def recommend_for_users(self, users, n, dataset=""):
        raise NotImplementedError("recommend_for_users function must be implemented by inherited "
                                  "Recommender object")

    def find_nearest_items(self, items, n):
        raise NotImplementedError("recommend_for_items function must be implemented by inherited "
                                  "Recommender object")

    def neighbor_category_match(self, item, item_neighbors):
        return self.evaluation_module.neighbor_category_match(item, item_neighbors)


class ItemSimilarityRecommender(Recommender):

    def __init__(self, dh, algo, k, simfunc, scores_chart):
        super().__init__(dh)
        self.model = ItemSimilarityModel(algo, k, simfunc)
        self.scores_chart = scores_chart

    def recommend_for_users(self, users, n, dataset="all"):
        data = self.data_handler.getData(dataset)
        similarities = self.model.GetSimilarities(data)
        recommendations = self.evaluation_module.get_recommendations_for_users(
            users, similarities, n, self.scores_chart, dataset=dataset)
        return recommendations

    def find_nearest_items(self, items, n):

        data = self.data_handler.getData("all")
        similarities = self.model.GetSimilarities(data)
        nearest_neighbors, distances = \
            self.evaluation_module.get_item_neighbors(items, similarities, n)
        return nearest_neighbors, distances


class RandomRecommender(Recommender):

    def __init__(self, dh):
        super().__init__(dh)


class PopularityRecommender(Recommender):

    def __init__(self, dh):
        super().__init__(dh)
