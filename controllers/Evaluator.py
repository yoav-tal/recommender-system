import numpy as np
from controllers.Viewer import ThumbnailViewer


class Evaluator:

    def __init__(self):

        self.recommenders = []
        self.viewer = ThumbnailViewer()

    def add_recommender(self, recommender, name):
        self.recommenders.append((recommender, name))
        if self.viewer.datahandler is None:
            self.viewer.set_datahandelr(recommender.data_handler)

    def evaluate(self, find_nearest_to=None, n_nearest_to=10, print_nearest=True,
                 view_nearest=False, print_category_match_index=False,
                 calc_match_index=False, n_for_match_index=10,
                 get_recom_for=None, n_for_recom=10,
                 print_recom=True, view_recom=False,
                 print_recom_category_match_index=False, calc_hitRate=False, n_for_hitRate=10,
                 show_hit_position_index=False, calc_novelty_score=False, n_for_novelty=10):

        if find_nearest_to:
            self.find_nearest_items(find_nearest_to, n_nearest_to, print_nearest, view_nearest,
                                    print_category_match_index)

        if calc_match_index:
            self.calculate_match_index(n_for_match_index)

        if get_recom_for:
            self.recommend(get_recom_for, n_for_recom, print_recom,
                           view_recom, print_recom_category_match_index)

        if calc_hitRate:
            self.calculate_hit_rate(n_for_hitRate, show_hit_position_index)

        if calc_novelty_score:
            self.calculate_novelty_score(n_for_novelty)

    def calculate_hit_rate(self, n, show_hit_position_index):

        for recommender, recommender_name in self.recommenders:
            hit_rate = recommender.calc_hit_rate(n)
            print("Hit Rate of recommender %s with %g recommendations per user: %g\n" %
                  (recommender_name, n, hit_rate))

        if show_hit_position_index:
            # for each user, find the position in the ranked items of the left-out item,
            # and return the min, Q1, median, Q3 and max
            pass

    def calculate_match_index(self, n):
        # For each model, get n nearest neighbors for all items and calc match index (mean match
        # over items) of the model

        for recommender, recommender_name in self.recommenders:
            match_index = recommender.calc_match_index(n)
            print("\nModel: %s \tcategory match index: %g" % (recommender_name, match_index))

    def calculate_novelty_score(self, n):

        for recommender, recommender_name in self.recommenders:
            novelty_score = recommender.calc_novelty_score(n)
            print("Novely score of recommender %s with %g recommendations per user: %g\n" %
                  (recommender_name, n, novelty_score))

    def find_nearest_items(self, find_nearest_to, n_nearest_to, print_nearest, view_nearest,
                           print_category_match_index):
        nearestToItem = {}

        for recommender, recommender_name in self.recommenders:
            nearest_neighbors, distances = recommender.recommend_for_items(find_nearest_to,
                                                                           n_nearest_to)
            nearestToItem[recommender_name] = {"items": nearest_neighbors, "distances": distances}

        for i, item in enumerate(find_nearest_to):

            for recommender, recommender_name in self.recommenders:
                item_neighbors = nearestToItem[recommender_name]["items"][i]
                print("model %s:" % recommender_name)
                if print_nearest:
                    self.viewer.print_recommendations(
                        [[item]], item_neighbors, nearestToItem[recommender_name]["distances"][i])
                if print_category_match_index:
                    match_index = recommender.neighbor_category_match(item, item_neighbors)
                    print("category match index of recommendations: %g" % (match_index))

                if view_nearest:
                    self.viewer.view_thumbnails([item], item_neighbors)
            print("\n\n")

    def recommend(self, user_ids, n_for_recom, print_recom, view_recom,
                  print_recom_category_match_index):
        recommendations = {}

        for recommender, recommender_name in self.recommenders:
            recomms = recommender.recommend_for_users(user_ids, n_for_recom)
            recommendations[recommender_name] = {"items": recomms}

        for i, user in enumerate(user_ids):
            for _, recommender_name in self.recommenders:
                user_recomms = recommendations[recommender_name]["items"][i]
                print("recommender %s:" % recommender_name)
                if print_recom:
                    self.viewer.print_user_recommendations(user, user_recomms)

                if view_recom:
                    self.viewer.view_user_recommendations(user, user_recomms)


if __name__ == "__main__":
    from controllers.data_handler import DataHandler
    from algorithms.RecSysAlgo import ItemsSVDAlgo, ItemsUsersSVDAlgo
    from similarity_functions.SimFunc import CosineSimilarity
    from recommendation.Recommender import ItemSimilarityRecommender, RandomRecommender, \
        PopularityRecommender, UserBasedRecommender

    # data_specs = {"project_name": "boost", "data_name": "min_previews 20 - us", "variants": ["",
    #                                                                                         "ch",
    #                                                                                      "ja"]}

    data_specs = {"project_name": "story", "data_name": "min_previews 5"}
    dh = DataHandler(data_specs)
    E = Evaluator()

    # establish SVD Recommenders
    k_dimensions = [1, 2, 3, 4]
    for k in k_dimensions:
        algo = ItemsSVDAlgo(replace_zero_by=1)
        simfunc = CosineSimilarity
        scores_chart = {"0": 1, "1": 1}
        SVD_recommender = ItemSimilarityRecommender(dh, algo, k, simfunc, scores_chart)
        # E.add_recommender(SVD_recommender, f"SVD_k%g_cosine" % k)

    # establish random recommendation algorithm
    random_recommender = RandomRecommender(dh)
    E.add_recommender(random_recommender, "Random_seedDefault")

    # popularity-based recommender
    popularity_recommender = PopularityRecommender(dh)
    E.add_recommender(popularity_recommender, "Popularity")

    # user based recommender
    for zero_replacement in [-1, 0, 1]:
        for k in [10, 20, 30]:
            for zero_score in [-1, 0, 0.5, 1]:
                algo = ItemsUsersSVDAlgo(replace_zero_by=zero_replacement)
                simfunc = CosineSimilarity
                scores_chart = {"0": zero_score, "1": 1}
                SVD_user_based = UserBasedRecommender(dh, algo, k, simfunc, scores_chart,
                                                      n_neighbors=10, d_neighbors=5, d_user=5)
                # E.add_recommender(SVD_user_based, "zero_replacements: %g; k: %g; zero_score: %g" %
                #                   (zero_replacement, k, zero_score))

    E.evaluate(calc_hitRate=True, n_for_hitRate=6)
    # E.evaluate(calc_match_index=True, n_for_match_index=6)
    # E.evaluate(calc_novelty_score=True, n_for_novelty=6)
    # E.evaluate(find_nearest_to=list(range(10)), n_nearest_to=6,
    #            print_nearest=True, view_nearest=True, print_category_match_index=True)
    # E.evaluate(find_nearest_to=list(np.random.randint(149, size=2)), n_nearest_to=6,
    #            print_nearest=True, view_nearest=True, print_category_match_index=True)
    # E.evaluate(get_recom_for=list(range(1, 10)), n_for_recom=6, view_recom=True)
    # E.evaluate(get_recom_for=[5], n_for_recom=6, view_recom=True)

    # E.evaluate(calc_hitRate=True, n_for_hitRate=6, calc_match_index=True, n_for_match_index=6,
    #        calc_novelty_score=True, n_for_novelty=6, find_nearest_to=list(range(10)),
    #        n_nearest_to=6, print_nearest=True, view_nearest=True,
    #        print_category_match_index=True, get_recom_for=list(range(6, 10)), n_for_recom=6,
    #        view_recom=True)
    # E.evaluate(find_nearest_to=list(np.random.randint(149, size=1)), n_nearest_to=6,
    #        print_nearest=True, view_nearest=True, print_category_match_index=True,
    #        get_recom_for=[5], n_for_recom=6, view_recom=True)