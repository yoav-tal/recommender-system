import warnings
import pandas as pd
import numpy as np
import json


class DataHandler:

    def __init__(self, data_specs):
        data_specs["variants"] = [""] if "variants" not in data_specs.keys() else \
            data_specs["variants"]
        self.data_specs = data_specs
        self.data, self.users_map, self.items_map = self.LoadData(data_specs)
        self.metadata, self.thumbnails_path = self.LoadMetadata(data_specs)
        self.json_configuration_field = "jsonConfigurationName" if \
            data_specs["project_name"] == "story" else "configuration"
        self.LOO_data = {"usage_data": None, "left_out": pd.DataFrame()}

    def getData(self, dataset):
        if dataset == "all":
            return self.data
        elif dataset == "LOO":
            return self.LOO_data["usage_data"]

    def getItemCategories(self, item):
        if type(item) != str and type(item) != np.str_:
            item = self.getItemName([[item]])[0][0]
        variant = item.split("-")[-1]
        variant = variant if variant in self.metadata.keys() else ""
        metadata = self.metadata[variant]
        for dat in metadata:
            if dat[self.json_configuration_field] == item:
                return dat["templateCategories"]
        # in case a configuration is not in the json file
        return []

    def getItemName(self, item_ids):
        item_names = []
        for id_list in item_ids:
            item_names.append(self.items_map.iloc[id_list, 1].values)
        return np.array(item_names)

    def getItemThumbnailPath(self, item):
        if type(item) != str and type(item) != np.str_:
            item = self.getItemName([[item]])[0][0]
        variant = item.split("-")[-1]
        variant = variant if variant in self.metadata.keys() else ""
        metadata = self.metadata[variant]
        for dat in metadata:
            if dat[self.json_configuration_field] == item:
                return self.thumbnails_path + dat["templateThumbnail"]

    def getNumItems(self, dataset="all"):
        if dataset == "all":
            return self.items_map.template_index.max() + 1
        if dataset == "LOO":
            item_indices = np.unique(self.LOO_data["usage_data"].template_index.values)
            if item_indices.max() + 1 > len(item_indices):
                warnings.warn("Some items are not represented in the LOO train set. \n"
                              "These missing items may be recommended upon by Usage-independent "
                              "recommenders.")
            return item_indices.max() + 1

    def getUsage(self, user_ids):
        return self.data.loc[self.data.user_index.isin(user_ids)]

    def getUserActivity(self, user_id, dataset="all"):
        if dataset == "all":
            return self.data.loc[self.data.user_index == user_id,
                                 ["template_index", "is_selected"]].values
        elif dataset == "LOO":
            return self.LOO_data["usage_data"].loc[self.LOO_data["usage_data"].
                                                   user_index == user_id,
                                                   ["template_index", "is_selected"]].values
        else:
            raise ValueError("unknown dataset " + dataset)

    def getUserActivitySplit(self, user_id):
        user_previews = self.getUserActivity(user_id)
        selected = user_previews[np.where(user_previews[:, 1] == 1), 0]
        unselected = user_previews[np.where(user_previews[:, 1] == 0), 0]
        return selected.squeeze(0), unselected.squeeze(0)

    def getUserIds(self, dataset="all"):
        if dataset == "all":
            return self.data.user_index.unique()
        elif dataset == "LOO":
            return self.LOO_data["usage_data"].user_index.unique()
        else:
            raise ValueError("unknown dataset " + dataset)

    def LoadData(self, dataspecs):
        data_path = "../data/RecSys data/" + dataspecs["project_name"] + "/" + \
                    dataspecs["data_name"] + ".csv"
        users_map_path = "../data/processing/" + dataspecs["project_name"] + "/" + \
                         dataspecs["data_name"] + " users_map"
        items_map_path = "../data/processing/" + dataspecs["project_name"] + "/" + \
                         dataspecs["data_name"] + " templates_map"
        data = pd.read_csv(data_path)
        users_map = pd.read_csv(users_map_path)
        items_map = pd.read_csv(items_map_path)
        return data, users_map, items_map

    def LoadMetadata(self, dataspecs):
        thumbnails_path = "../data/templates metadata/" + \
                          dataspecs["project_name"] + "/thumbnails/"

        metadata_path = "../data/templates metadata/" + dataspecs["project_name"] \
                        + "/TemplatesMetadata"

        metadata = {}
        for variant in dataspecs["variants"]:
            suffix = "-" + variant + ".json" if len(variant) > 0 else variant + ".json"
            metadata[variant] = json.load(open(metadata_path + suffix))

        return metadata, thumbnails_path

    def prepare_LOO_dataset(self):
        # for each user in the data, randomly remove one selected item. If the user has no items
        # labeled as selected, discard the user. Return the obfuscated dataset and list of the
        # left-out items.
        # Optional: Hold a map that transforms user's index from LOO-dataset to 'all'.

        def loo(user_previews):
            this_user = user_previews.user_index.iloc[0]
            selected = user_previews.loc[user_previews.is_selected == 1]
            if len(selected) < 2:
                return pd.DataFrame()
            else:
                to_remove = selected.iloc[np.random.randint(len(selected))].template_index
                removed_items.append({"user_index": this_user, "template_index": to_remove})
                return user_previews[user_previews.template_index != to_remove]

        removed_items = []
        # for i in range(3):
        #     print("!!! USING 1000 DATA LINES FOR CODE TESTING !!!")
        # self.data = self.data.iloc[:7000]
        users_split = self.data.groupby(by="user_index")

        if self.LOO_data["usage_data"] is None:
            print("Constructing LOO dataset and left-out test set")
            LOO_data = users_split.apply(loo)

            self.LOO_data["usage_data"] = LOO_data.reset_index(drop=True).astype(int)
            self.LOO_data["left_out"] = pd.DataFrame(removed_items[1:])

        return self.LOO_data["usage_data"], self.LOO_data["left_out"]["template_index"].values


if __name__ == "__main__":
    dh = DataHandler("min_previews 20")
    print(dh.items_map.shape)
    print(dh.users_map.shape)
    print(dh.data.shape)
    print(dh.getItemName([[1, 2]]))
    print(dh.getItemName([[1, 2], [3, 4]]))

