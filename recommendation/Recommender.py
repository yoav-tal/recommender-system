import numpy as np

from controllers.Model import ItemSimilarityModel, LowDimEmbeddingModel
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
        nearest_neighbors, _ = self.recommend_for_items(list(range(n_items)), n)

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

    def recommend_for_items(self, items, n):
        raise NotImplementedError("recommend_for_items function must be implemented by inherited "
                                  "Recommender object")

    def neighbor_category_match(self, item, recommendations_for_item):
        return self.evaluation_module.neighbor_category_match(item, recommendations_for_item)


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

    def recommend_for_items(self, items, n):

        data = self.data_handler.getData("all")
        similarities = self.model.GetSimilarities(data)
        nearest_neighbors, distances = \
            self.evaluation_module.get_item_neighbors(items, similarities, n)
        return nearest_neighbors, distances


class UserBasedRecommender(Recommender):

    def __init__(self, dh, algo, k, simfunc, scores_chart, n_neighbors=10, d_neighbors=5,
                 d_user=5):
        super().__init__(dh)
        self.model = LowDimEmbeddingModel(algo, k, simfunc)
        self.simfunc = simfunc(k)
        self.scores_chart = scores_chart
        self.recommendation_function = \
            self.evaluation_module.establish_user_based_recommendation_function(n_neighbors,
                                                                                d_neighbors,
                                                                                d_user,
                                                                                self.simfunc)

    def recommend_for_users(self, users, n, dataset="all"):
        data = self.data_handler.getData(dataset)
        user_embeddings = self.model.GetUserEmbeddings(data)
        item_similarity = self.model.GetItemSimilarities(data)
        recommendations = self.recommendation_function(users, user_embeddings, item_similarity, n,
                                                       self.scores_chart, dataset=dataset)
        return recommendations

    def recommend_for_items(self, items, n):
        recommendations, votes = self.evaluation_module.get_user_votes(items, n, dataset="all")
        return recommendations, votes


class RandomRecommender(Recommender):

    def __init__(self, dh, seed=0):
        super().__init__(dh)
        self.seed = seed

    def recommend_for_items(self, items, n):
        num_items = self.data_handler.getNumItems()
        return np.random.randint(num_items, size=(len(items), n)),\
               np.zeros(shape=(len(items), n))

    def recommend_for_users(self, users, n, dataset="all"):
        num_items = self.data_handler.getNumItems(dataset=dataset)
        return np.random.randint(num_items, size=(len(users), n))


class PopularityRecommender(Recommender):

    def __init__(self, dh):
        super().__init__(dh)
        self.popularity = get_popularity_as_similarity_matrix(self.data_handler.data)

    def recommend_for_items(self, items, n):
        recommendations, ranks = \
            self.evaluation_module.get_item_neighbors(items, self.popularity, n)
        return recommendations, ranks

    def recommend_for_users(self, users, n, dataset="all"):
        recommendations = self.evaluation_module.get_recommendations_for_users(
            users, self.popularity, n, scores_chart={"0": 0, "1": 1}, dataset=dataset)
        return recommendations


def get_popularity_as_similarity_matrix(data):
    # get a matrix in which all rows are identical and represent popularity: in the position of
    # the least popular we have 0, the next least popular is 1 and so on, the most popular is N
    # (N being the number of items)
    unique, counts = np.unique(data.template_index.values, return_counts=True)
    temp = np.argsort(counts)
    popularity_ranks = np.empty_like(temp)
    popularity_ranks[temp] = np.arange(len(temp))

    n_items = len(popularity_ranks)
    popularity_as_similarities = np.repeat([popularity_ranks], n_items, axis=0)

    return popularity_as_similarities
