import argparse

import numpy as np
import pandas as pd

from google.cloud import bigquery

from production.seen.hyperparams import query_path, data_path, processed_data_path, \
    similarities_filename, model
from production.utils import run_query, run_preprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-start_date", default='2019-11-01')
    parser.add_argument("-end_date", default='2020-01-01')
    parser.add_argument("-min_previews", default=20)
    parser_args = parser.parse_args()
    dates = (parser_args.start_date, parser_args.end_date)

    client = bigquery.Client()
    run_query(query_path, data_path, dates, bigQueryClient=client)

    data, row_template_map = run_preprocessing(data_path, processed_data_path,
                                               parser_args.min_previews, dates)
    template_similarities = model.GetSimilarities(data)

    formatted_output = np.c_[row_template_map.template_name.values, template_similarities]
    dataframe_titles = np.insert("template_name", 0, row_template_map.template_name.values)
    output_dataframe = pd.DataFrame(formatted_output)

    similarities_filename += f" min_previews %g from %s to %s" % (parser_args.min_previews,
                                                                  dates[0], dates[1])

    output_dataframe.to_csv(f"./similarity_matrices/%s" % similarities_filename, index=False,
                            header=False)
