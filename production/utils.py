import pandas as pd


def run_query(dates):
    query_path = "../../data/query data/demo for story.csv"
    query_data = pd.read_csv(query_path)
    return query_data


def run_preprocessing(query_data, min_previews):

    query_data = query_data[["id_for_vendor", "template_name", "is_selected"]]
    query_data.is_selected = query_data.is_selected.astype(int)

    # Discard cases of multiple user-template event. Take the 'selected' label as 1 if at least one
    # preview was selected.
    query_data.drop_duplicates(inplace=True)
    query_data = query_data.groupby(by=["id_for_vendor", "template_name"]).\
        agg({"is_selected": "max"}).reset_index()

    # discard users with too few events
    query_data = query_data.groupby(by="id_for_vendor").filter(lambda x: len(x) >= min_previews)

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
    unique_template_name = unique_template_name.sort_values(by="template_name").reset_index(drop=True)
    unique_template_name["template_index"] = unique_template_name.index
    template_index_map = unique_template_name[["template_index", "template_name"]]

    output_df = query_data.merge(unique_template_name, on="template_name")
    output_df = output_df.sort_values(by="user_index").reset_index(drop=True)
    output_df = output_df[["user_index", "template_index", "is_selected"]]

    return output_df, template_index_map
