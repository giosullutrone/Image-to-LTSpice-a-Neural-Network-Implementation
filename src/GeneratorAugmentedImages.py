class GeneratorAugmentedImages:
    """A class that generates augmented images with random transformations from given images and boxes"""

    def __init__(self, folder_images, folder_boxes,
                 rotation_probability=1.0,
                 zoom_probability=1.0,
                 flip_vertical_probability=0.2,
                 flip_horizontal_probability=0.2,
                 rotation_values=(-0.26, 0.26),
                 zoom_values=(0.99, 1.01),
                 seed=42):

        self.__folder_images = folder_images
        self.__folder_boxes = folder_boxes

        self.__seed = seed

        self.__rotation_probability = rotation_probability
        self.__zoom_probability = zoom_probability
        self.__flip_vertical_probability = flip_vertical_probability
        self.__flip_horizontal_probability = flip_horizontal_probability

        self.__rotation_values = rotation_values

        self.__zoom_values = zoom_values

        self.__image_exts = (".png", ".jpg")
        self.__box_ext = ".txt"

    def get_rotation_probability(self):
        return self.__rotation_probability

    def set_rotation_probability(self, rotation_probability):
        self.__rotation_probability = rotation_probability

    def get_zoom_probability(self):
        return self.__zoom_probability

    def set_zoom_probability(self, zoom_probability):
        self.__zoom_probability = zoom_probability

    def get_flip_vertical_probability(self):
        return self.__flip_vertical_probability

    def set_flip_vertical_probability(self, flip_vertical_probability):
        self.__flip_vertical_probability = flip_vertical_probability

    def get_flip_horizontal_probability(self):
        return self.__flip_horizontal_probability

    def set_flip_horizontal_probability(self, flip_horizontal_probability):
        self.__flip_horizontal_probability = flip_horizontal_probability

    def get_image_exts(self):
        return self.__image_exts

    def get_box_ext(self):
        return self.__box_ext

    def set_image_exts(self, image_ext):
        self.__image_exts = image_ext

    def set_box_ext(self, box_ext):
        self.__box_ext = box_ext

    def set_seed(self, seed):
        self.__seed = seed

    def set_rotation_values(self, values):
        """Set possible rotation values ex. => (-0.26, 0.26) => rotate by (a value from -0.26, 0.26) radians"""
        self.__rotation_values = values

    def set_zoom_values(self, values):
        """Set possible zoom values ex. => (-0.01, 0.01) => size of the image * (a value from 0.99 to 1.01)"""
        self.__zoom_values = values

    def generate_augmented_images_to_folder(self, folder_ouput, number_of_images, image_size=(416, 416), information_loss_checker_method=None):
        """Generates augmented images with the four random transformation and saves them in a specific folder.
        An InformationLossChecker method can be given to check for data loss while still guaranteeing the specified number of images"""
        self.generate_augmented_images_to_folders(folder_ouput, folder_ouput, number_of_images, image_size=image_size, information_loss_checker_method=information_loss_checker_method)

    def generate_augmented_images_to_folders(self, folder_images_ouput, folder_boxes_output, number_of_images, image_size=(416, 416), information_loss_checker_method=None):
        """Generates augmented images with the four random transformation and saves them in specific folders.
        An InformationLossChecker method can be given to check for data loss while still guaranteeing the specified number of images"""

        import random
        from src.AugmentedImage import AugmentedImage
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        import os

        os.makedirs(folder_images_ouput, exist_ok=True)
        os.makedirs(folder_boxes_output, exist_ok=True)

        random.seed(self.__seed)

        images_and_boxes_file_names = AugmentedImagesUtil.get_images_and_boxes_file_names_from_folder(self.__folder_images,
                                                                                                      self.__folder_boxes,
                                                                                                      self.__image_exts,
                                                                                                      self.__box_ext)
        done = 0

        while done < number_of_images:
            image_file, box_file = random.choice(images_and_boxes_file_names)
            augmented_image = AugmentedImage.from_files(self.__folder_images + image_file, self.__folder_boxes + box_file)

            r = random.random()
            if r <= self.__rotation_probability:
                augmented_image.rotate_image(random.uniform(self.__rotation_values[0], self.__rotation_values[1]))

            r = random.random()
            if r <= self.__zoom_probability:
                augmented_image.zoom_image(random.uniform(self.__zoom_values[0], self.__zoom_values[1]))

            r = random.random()
            if r <= self.__flip_vertical_probability:
                augmented_image.flip_image_vertically()

            r = random.random()
            if r <= self.__flip_horizontal_probability:
                augmented_image.flip_image_horizontally()

            if information_loss_checker_method is not None:
                if information_loss_checker_method(augmented_image):
                    continue

            image_path_new = folder_images_ouput + image_file.replace(".", f"_{done}.")
            box_path_new = folder_boxes_output + box_file.replace(".", f"_{done}.")
            augmented_image.to_files(image_path_new, box_path_new, image_size=image_size, grayscale=True)

            done += 1
