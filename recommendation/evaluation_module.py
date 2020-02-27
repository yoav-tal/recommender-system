import numpy as np


class EvaluationModule:

    def __init__(self, datahandler):
        self.datahandler = datahandler

    def calc_hit_rate(self, recommendations, left_out):
        print("calculating hit rate\n")
        hits = 0
        for i, item in enumerate(left_out):
            hits += item in recommendations[i]

        return hits / len(left_out)

    def calc_novelty_score(self, recommendations, data):

        # get popularity rank
        unique, counts = np.unique(data.template_index.values, return_counts=True)
        temp = np.argsort(counts)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))
        ranks = len(ranks) - ranks  # np.where(ranks==1) is the most popular,
        # np.where(ranks==ranks.max()) is the least popular

        novelty_score = ranks[recommendations].sum()/np.prod(recommendations.shape)
        return novelty_score

    def get_item_neighbors(self, item_ids, similarities, n):

        similarity_rows = similarities[item_ids, :]
        most_similar = np.argsort(similarity_rows)[:, -n - 1:]  # taking n+1 items since item
        #  is always nearest itself
        nearest_neighbors = np.empty((len(item_ids), n), dtype=np.int)
        for i, item_id in enumerate(item_ids):
            if item_id in most_similar[i]:
                nearest_neighbors[i] = most_similar[i][most_similar[i] != item_id]
            else:
                nearest_neighbors[i] = most_similar[i, -n:]

        distances = [similarity_rows[i, nearest_neighbors[i]] for i in range(len(item_ids))]
        distances = np.fliplr(distances)

        return np.fliplr(nearest_neighbors), distances

    def get_recommendations_by_items(self, item_ids, item_scores, similarities, n):
        # For each item, get the relevant row from the similarity matrix, and multiply it by the
        # score (thus scoring each item by its similarity to used items, weighted by items'
        # scores). Then sum up all the scored rows, thus getting an overall score for each item
        # based on input items and scores. Then get top n of unused items.

        similarity_rows = similarities[item_ids, :]
        scored_rows = similarity_rows * item_scores[:, None]
        weighted_sums = scored_rows.sum(axis=0)
        most_similar = np.argsort(weighted_sums)[-n - len(item_ids):]
        for item in item_ids:
            if item in most_similar:
                most_similar = most_similar[most_similar != item]
        recommended = most_similar[-n:]

        # user may have used more than n_items - n_for_recommendation items. In this case we add
        # previously used items at the end of the recommended items list
        if len(recommended) < n:
            num_missing = n - len(recommended)
            picked_from_used = np.random.choice(item_ids, num_missing)
            recommended = np.insert(recommended, 0, picked_from_used)

        return np.flip(recommended)

    def get_recommendations_for_users(self, users_ids, similarities, n, scores_chart,
                                      dataset="all"):
        # for each user, get the set of previews and their labels (scores). i.e. get the
        # equivalent to a row from the utility matrix. Transform the labels to scores according
        # to the scores chart.
        # Then get recommendations for the set of items with their labels and store them in an
        # array to return
        recommendations = np.empty((len(users_ids), n), dtype=np.int)
        print("setting recommendations for each user")
        for i, user in enumerate(users_ids):
            user_activity = self.datahandler.getUserActivity(user, dataset=dataset)
            items = user_activity[:, 0]
            labels = user_activity[:, 1]
            for label, score in scores_chart.items():
                labels[np.where(labels == int(label))] = score
            recommendations[i] = self.get_recommendations_by_items(items, labels, similarities, n)

        return recommendations

    def neighbor_category_match(self, subject_item, neighboring_items):

        match_index = 0
        item_categories = set(self.datahandler.getItemCategories(subject_item))
        if item_categories == set():
            return 0

        for neighbor in neighboring_items:
            neighbor_categories = set(self.datahandler.getItemCategories(neighbor))
            if neighbor_categories == set():
                category_match = 0
            else:
                category_match = len(item_categories.intersection(neighbor_categories)) / \
                    np.sqrt(len(item_categories) * len(neighbor_categories))
            match_index += category_match
        return match_index / len(neighboring_items)
