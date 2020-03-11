import argparse

import numpy as np
import pandas as pd

from production.seen.hyperparams import min_previews, model
from production.utils import run_query, run_preprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-start_date", default='2019-12-11')
    parser.add_argument("-end_date", default='2020-02-01')
    parser.add_argument("-filename", default="testing")
    parser_args = parser.parse_args()
    dates = (parser_args.start_date, parser_args.end_date)

    query_data = run_query(dates)
    data, row_template_map = run_preprocessing(query_data, min_previews)
    template_similarities = model.GetSimilarities(data)

    formatted_output = np.c_[row_template_map.template_name.values, template_similarities]
    dataframe_titles = np.insert("template_name", 0, row_template_map.template_name.values)
    output_dataframe = pd.DataFrame(formatted_output)
    output_dataframe.to_csv(f"./similarity_matrices/%s" % parser_args.filename, index=False,
                            header=False)
