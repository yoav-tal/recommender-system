import pandas as pd
from os import makedirs
pd.set_option('display.max_columns', None)

data_filename = "../query data/demo for boost.csv"
min_previews = 20
project_name = "boost"
variant = "ja"
makedirs("../RecSys data/" + project_name, exist_ok=True)
makedirs("./" + project_name, exist_ok=True)

print("loading raw data...")
query_data = pd.read_csv(data_filename)
query_data = query_data[["id_for_vendor", "template_name", "is_selected"]]
query_data.is_selected = query_data.is_selected.astype(int)

# Discard cases of multiple user-template event. Take the 'selected' label as 1 if at least one
# preview was selected.
print("cleaning data...")
if variant is not None:
    if variant in ["ch", "ja"]:
        # leave only rows in which template_name ends with the variant string
        def is_in_var(name):
            return name.split("-")[-1] == variant
    else:
        def is_in_var(name):
            return name.split("-")[-1] not in ["ch", "ja"]

    query_data["in_variant"] = query_data.template_name.apply(is_in_var)
    query_data = query_data[query_data.in_variant]
    query_data.drop(columns="in_variant", inplace=True)

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
new_df = query_data.merge(unique_template_name, on="template_name")
new_df = new_df.sort_values(by="user_index").reset_index(drop=True)


print("saving files")
data_name = f"min_previews %g" % min_previews
if variant in ["ch", "ja"]:
    data_name += " - " + variant
elif variant == "":
    data_name += " - us"

new_df[["user_index", "template_index", "is_selected"]].to_csv(
    f"../RecSys data/%s/%s.csv" % (project_name, data_name), index=False)
unique_user_id[["user_index", "id_for_vendor"]].to_csv(
    f"./%s/%s users_map" % (project_name, data_name), index=False)
unique_template_name[["template_index", "template_name"]].to_csv(
    f"./%s/%s templates_map" % (project_name, data_name), index=False)
print("process ended")
