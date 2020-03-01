from PIL import Image


class ThumbnailViewer:

    def __init__(self):
        self.datahandler = None
        self.default_image_shape = (400, 400)

    def set_datahandelr(self, data_handler):
        self.datahandler = data_handler
        if data_handler.data_specs["project_name"] == "story":
            self.default_image_shape = (720, 1280)

    def view_thumbnails(self, selected_items, unselected_items, recommended_items=[]):
        num_horizontal = max(len(selected_items), len(unselected_items), len(recommended_items))
        # here set default shape
        thumbnails_array = None
        for item_list in [selected_items, unselected_items, recommended_items]:
            if len(item_list) == 0:
                continue

            thumbnails_row = self.concatItemThumbnails(item_list, num_horizontal)
            if thumbnails_array is None:
                thumbnails_array = thumbnails_row
            else:
                thumbnails_array = img_concat_v(thumbnails_array, thumbnails_row)

        thumbnails_array.show()

    def concatItemThumbnails(self, item_list, num_thumbnails, image_shape=None):
        image_shape = image_shape if image_shape else self.default_image_shape
        for i, item in enumerate(item_list):
            item_thumbnail = self.datahandler.getItemThumbnailPath(item)
            img = Image.new('RGB', image_shape) if item_thumbnail is None \
                else Image.open(item_thumbnail)
            img = img.resize(image_shape)
            if i == 0:
                thumbnails_row = img
            else:
                thumbnails_row = img_concat_h(thumbnails_row, img)
        for i in range(len(item_list), num_thumbnails):
                thumbnails_row = img_concat_h(thumbnails_row, Image.new('RGB', image_shape))

        return thumbnails_row

    def print_recommendations(self, subject_items, recommended_items, *details):
        print("Subject items:")
        for item in subject_items:
            print(self.datahandler.getItemName([item]).squeeze())
        print("Recommended items:")
        if details is not None:
            recommended_items = self.datahandler.getItemName([recommended_items])
            for rec_data in zip(recommended_items[0], details[0]):
                print(rec_data)
        else:
            for rec_data in recommended_items:
                print(rec_data)
        return

    def print_user_recommendations(self, user_id, recommended_items):
        print("User: %g" % user_id)
        selected, unselected = self.datahandler.getUserActivitySplit(user_id)
        print("Selected previews:")
        print(self.datahandler.getItemName(selected))
        print("Unselected Previews:")
        print(self.datahandler.getItemName(unselected))
        print("Recommended_items:")
        print(self.datahandler.getItemName([recommended_items]))

    def view_user_recommendations(self, user, user_recomms):
        selected_items, unselected_items = self.datahandler.getUserActivitySplit(user)
        self.view_thumbnails(selected_items.squeeze(), unselected_items.squeeze(), user_recomms)


def img_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
