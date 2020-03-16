from os.path import isfile
import pandas as pd
import pickle


def run_query(query_path, data_path, dates, bigQueryClient):

    data_file = data_path + f"from %s to %s" % (dates[0], dates[1])
    if isfile(data_file):
        return

    query_text = read_query_text(query_path, dates)
    query_job = bigQueryClient.query(query_text)
    results = query_job.result()

    print("converting query results to dataframe")
    query_data = results.to_dataframe(progress_bar_type='tqdm')
    pickle.dump(query_data, open(data_file, "wb"))


def run_preprocessing(data_path, processed_data_path, min_previews, dates):

    processed_data_file = processed_data_path + f"min_previews %s from %s to %s" % \
        (min_previews, dates[0], dates[1])
    if isfile(processed_data_file):
        output_df = pickle.load(open(processed_data_file, "rb"))
        template_index_map = pickle.load(open(processed_data_file + " templates_map", "rb"))

    else:
        data_file = data_path + f"from %s to %s" % (dates[0], dates[1])

        query_data = pickle.load(open(data_file, "rb"))
        query_data = query_data[["id_for_vendor", "template_name", "is_selected"]]
        query_data.is_selected = query_data.is_selected.astype(int)

        # Discard cases of multiple user-template event. Take the 'selected' label as 1 if at
        # least one preview was selected.
        query_data.drop_duplicates(inplace=True)
        query_data = query_data.groupby(by=["id_for_vendor", "template_name"]).\
            agg({"is_selected": "max"}).reset_index()

        # discard users with too few events
        query_data = query_data.groupby(by="id_for_vendor").\
            filter(lambda x: len(x) >= min_previews)

        # create serial user id and map
        print("creating serial user id...")
        unique_user_id = set(query_data.id_for_vendor)
        unique_user_id = pd.DataFrame(list(unique_user_id), columns=["id_for_vendor"])
        unique_user_id = unique_user_id.sort_values(by="id_for_vendor").reset_index(drop=True)
        unique_user_id["user_index"] = unique_user_id.index

        query_data = query_data.merge(unique_user_id, on="id_for_vendor")

        print("creating serial item id...")
        unique_template_name = set(query_data.template_name)
        unique_template_name = pd.DataFrame(list(unique_template_name), columns=["template_name"])
        unique_template_name = unique_template_name.sort_values(by="template_name").\
            reset_index(drop=True)
        unique_template_name["template_index"] = unique_template_name.index
        template_index_map = unique_template_name[["template_index", "template_name"]]

        output_df = query_data.merge(unique_template_name, on="template_name")
        output_df = output_df.sort_values(by="user_index").reset_index(drop=True)
        output_df = output_df[["user_index", "template_index", "is_selected"]]

        pickle.dump(output_df, open(processed_data_file, "wb"))
        pickle.dump(template_index_map, open(processed_data_file + " templates_map", "wb"))

    return output_df, template_index_map


def read_query_text(query_path, dates):
    query_text = ""

    with (open(query_path, 'r')) as query_lines:
        for line in query_lines:
            if "start_date" in line:
                line = line.replace("{{start_date}}", dates[0])
            elif "end_date" in line:
                line = line.replace("{{end_date}}", dates[1])
            query_text += line

    return query_text
