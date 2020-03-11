
import numpy as np

from production.seen.hyperparams import scores_chart


def get_recommendations(selected_items, unselected_items, similarities, n):

    selected_similarity_rows = similarities[selected_items, :]
    selected_similarity_rows = selected_similarity_rows * scores_chart["1"]

    unselected_similarity_rows = similarities[unselected_items, :]
    unselected_similarity_rows = unselected_similarity_rows * scores_chart["0"]

    weighted_sums = selected_similarity_rows.sum(axis=0) + unselected_similarity_rows.sum(axis=0)

    all_items = np.concatenate((selected_items, unselected_items))
    weighted_sums[all_items] = 1e-9

    ordered = np.argsort(weighted_sums)

    recommended = ordered[-n:]

    return np.flip(recommended)


if __name__ == "__main__":
    similarities = [[1, 0.9, 0.8, 0.7],
                    [0.9, 1, 0.5, 0.4],
                    [0.8, 0.5, 1, 0.2],
                    [0.7, 0.4, 0.2, 1]]
    similarities = np.array(similarities)
    selected = np.array([1])
    unselected = np.array([2])
    recommended1 = get_recommendations(selected, unselected, similarities, 1)
    recommended2 = get_recommendations(selected, unselected, similarities, 2)
    recommended3 = get_recommendations(selected, unselected, similarities, 3)
