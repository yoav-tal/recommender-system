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

    def calc_novelty_score(self, recommendations):

        # get popularity rank
        data = self.datahandler.data
        unique, counts = np.unique(data.template_index.values, return_counts=True)
        temp = np.argsort(counts)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))
        ranks = len(ranks) - ranks  # np.where(ranks==1) is the most popular,
        # np.where(ranks==ranks.max()) is the least popular

        novelty_score = ranks[recommendations].sum()/np.prod(recommendations.shape)
        return novelty_score

    def establish_user_based_recommendation_function(self, n_neighbors, depth_neighbors,
                                                     depth_user, similarity_function):
        # with n_neighbors=0 this is item based recommendation
        # with depth_neighbors=0 and depth_user=0 this is standard user based recommendation of
        # items (without neighboring items)

        def recommendation_function(users, user_embeddings, item_similarity, n, scores_chart,
                                    dataset="all"):
            # for each user, take nearest n_neighbors. For each item used by the neighbors,
            # take depth_neighbors nearest items. Weigh items by distance in the item domain and
            # users domain. For each item used by the user, take depth_user nearest item and
            # weigh them by distance in the item domain. Discard used items and recommend top n.
            # The output should be a numpy array with shape (len(users), n)

            nearest_users, distances = self.find_nearest_users(users, user_embeddings, n_neighbors,
                                                               similarity_function)

            # The variant that calls self.weight_item_based_recommendations is correct, however it
            # implements a weighing that is based on the *recommendations* to the neighbors (
            # weights the top-n items for each user, where n=depth_neighbors). Therefore,
            # if depth_neighbors=0, it will not implement a standard user-based collaborative
            # filtering recommendation (it will give one arbitrary item from each neighbor).
            # This method will be replaced by a more correct one, however
            # self.weight_item_based_recommendations will still be available for future reference.
            # This method is currently not computationally efficient, as it repeatedly computes
            # recommendations for the neighbors of each subject user. It will not undergo the
            # required improvement of computing all recommendations in advance and reading them
            # from memory.

            # weighted_items = self.weight_item_based_recommendations(
            #   nearest_users, distances, item_similarity, scores_chart, depth_neighbors, dataset)

            weighted_items = self.weight_items_by_users(
                nearest_users, distances, item_similarity, scores_chart, depth_neighbors, dataset)

            rearranged_users = np.reshape(np.array(users), (len(users), 1))
            dummy_distances = np.ones(len(users))

            # user_weighted_items = self.weight_item_based_recommendations(
            #     rearranged_users, dummy_distances, item_similarity, scores_chart, depth_user,
            #     dataset)

            user_weighted_items = self.weight_items_by_users(
                rearranged_users, dummy_distances, item_similarity, scores_chart, depth_user,
                dataset)

            weighted_items += user_weighted_items

            recommendations = self.get_recommendations_from_weights(users, weighted_items, n,
                                                                    dataset)

            return recommendations

        return recommendation_function

    def find_nearest_users(self, users, user_embeddings, n_neighbors, similarity_function):

        user_representations = user_embeddings[:, users]
        similarity_rows = similarity_function(user_embeddings,
                                              subject_vectors=user_representations)
        most_similar = np.argsort(similarity_rows)[:, -n_neighbors - 1:]

        nearest_neighbors = np.empty((len(users), n_neighbors), dtype=np.int)
        for i, user in enumerate(users):
            if user in most_similar[i]:
                nearest_neighbors[i] = most_similar[i][most_similar[i] != user]
            else:
                nearest_neighbors[i] = most_similar[i, -n_neighbors:]
        distances = [similarity_rows[i, nearest_neighbors[i]] for i in range(len(users))]
        distances = np.fliplr(distances)

        return np.fliplr(nearest_neighbors), distances

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

    def get_recommendations_by_items(self, item_ids, item_scores, similarities, n,
                                     return_weights=False, discard_input_items=True):
        # For each item, get the relevant row from the similarity matrix, and multiply it by the
        # score (thus scoring each item by its similarity to used items, weighted by items'
        # scores). Then sum up all the scored rows, thus getting an overall score for each item
        # based on input items and scores. Then get top n of unused items.

        similarity_rows = similarities[item_ids, :]
        scored_rows = similarity_rows * item_scores[:, None]
        weighted_sums = scored_rows.sum(axis=0)
        most_similar = np.argsort(weighted_sums)[-n - len(item_ids):]
        if discard_input_items:
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
        if return_weights:
            return np.flip(recommended), np.flip(weighted_sums[recommended] / len(item_ids))
        return np.flip(recommended)

    def get_recommendations_for_users(self, users_ids, similarities, n, scores_chart,
                                      dataset="all", return_weights=False,
                                      discard_self_usage=True):
        # for each user, get the set of previews and their labels (scores). i.e. get the
        # equivalent to a row from the utility matrix. Transform the labels to scores according
        # to the scores chart.
        # Then get recommendations for the set of items with their labels and store them in an
        # array to return
        recommendations = np.zeros((len(users_ids), n), dtype=np.int)
        weights = np.zeros((len(users_ids), n))

        # print("setting recommendations for each user")
        for i, user in enumerate(users_ids):
            user_activity = self.datahandler.getUserActivity(user, dataset=dataset)
            items = user_activity[:, 0]
            labels = user_activity[:, 1]
            for label, score in scores_chart.items():
                labels[np.where(labels == int(label))] = score
            temp_recommendations = self.get_recommendations_by_items(items, labels,
                                                                     similarities, n,
                                                                     return_weights,
                                                                     discard_self_usage)
            if return_weights:
                recommendations[i], weights[i] = temp_recommendations
            else:
                recommendations[i] = temp_recommendations

        if return_weights:
            return recommendations, weights
        else:
            return recommendations

    def get_recommendations_from_weights(self, users, weighted_items, n, dataset="all"):

        # find top-n weighted items and discard already used items
        recommendations = np.zeros((len(users), n), dtype=np.int)
        ordered_items = np.argsort(weighted_items)

        for i, user in enumerate(users):
            used_items = self.datahandler.getUserActivity(user, dataset=dataset)[:, 0]
            top_items = ordered_items[i, -n - len(used_items):]
            for item in used_items:
                if item in top_items:
                    top_items = top_items[top_items != item]
            recommended = top_items[-n:]
            # Add random unused items at the end of the recommended items list if not
            # enough are recommended upon
            if len(recommended) < n:
                num_missing = n - len(recommended)
                picked_from_used = np.random.choice(used_items, num_missing)
                recommended = np.insert(recommended, 0, picked_from_used)

            recommendations[i] = np.flip(recommended)

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

    def weight_item_based_recommendations(self, nearest_users, distances, item_similarity, scores_chart,
                                          depth, dataset="all"):
        # for each user, do something like in recommend_for_user with n_recommendations=depth. But
        # get distances, not only indices. Multiply weights by distances and add together the
        # weights from each row of nearest_users.
        num_items = self.datahandler.getNumItems(dataset)
        weighted_items = np.zeros((nearest_users.shape[0], num_items))
        for i, neighbors_set in enumerate(nearest_users):
            recommended_items, weights = self.get_recommendations_for_users(
                neighbors_set, item_similarity, depth, scores_chart, dataset,
                return_weights=True, discard_self_usage=False)
            # weight each neighbors recommendations by her distance from the subject user
            weights = (weights.T * distances[i]).T
            # sum weights from all neighbors
            for j in range(len(neighbors_set)):
                weighted_items[i, recommended_items[j]] += weights[j]
        return weighted_items

    def weight_items_by_users(self, neighboring_users, distances, item_similarity, scores_chart,
                              depth, dataset="all"):

        num_items = self.datahandler.getNumItems(dataset)
        weighted_items = np.zeros((neighboring_users.shape[0], num_items))
        for i, neighbors_set in enumerate(neighboring_users):
            for neighbor in neighbors_set:
                used_items = self.datahandler.getUserActivity(neighbor, dataset=dataset)
                neighboring_items, weights = self.get_item_neighbors(used_items, item_similarity,
                                                                     depth)
                # weight neighboring items of each neighboring user by the distance of the
                # neighbor user from the subject user


                ###  this is wrong since all neighboring items are from one user!!!
                ### maybe should normalize in some way
                weights = (weights.T * distances[i]).T
                # sum weights from all neighbors
                for j in range(len(neighbors_set)):
                    weighted_items[i, nearest_items[j]] += weights[j]
        return weighted_items