import argparse
from production.seen.hyperparams import min_previews, model
from production.utils import run_query, run_preprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-start_date", default='2019-12-11')
    parser.add_argument("-end_date", default='2020-02-01')
    parser_args = parser.parse_args()
    dates = (parser_args.start_date, parser_args.end_date)

    query_data = run_query(dates)
    data, row_template_map = run_preprocessing(query_data, min_previews)
    template_similarities = model.GetSimilarities(data)

