class AugmentedImagesUtil:
    @staticmethod
    def get_images_file_names_from_folder(folder_images, image_exts=(".png", ".jpg")):
        """Returns the file names of images in the provided folder"""
        import os
        return [x for x in os.listdir(folder_images) if x.endswith(image_exts)]

    @staticmethod
    def get_images_and_boxes_file_names_from_folder(folder_images, folder_boxes, image_exts=(".png", ".jpg"), box_ext=".txt"):
        """Returns the file names of images and boxes as a list of tuples"""

        import os

        image_files = AugmentedImagesUtil.get_images_file_names_from_folder(folder_images, image_exts)
        box_files = [x.split(".")[0] + box_ext for x in image_files]

        for box_file in box_files:
            assert os.path.isfile(folder_boxes + box_file), "Box file does not exists for the given image..."

        images_and_boxes_files_paths = []
        for image_file, box_file in zip(image_files, box_files):
            images_and_boxes_files_paths.append((image_file, box_file))
        return images_and_boxes_files_paths

    @staticmethod
    def from_folders(folder_images, folder_boxes, grayscale=True, image_exts=(".png", ".jpg"), box_ext=".txt"):
        """Returns a list of augmented_image from the provided folders"""

        from src.AugmentedImage import AugmentedImage

        images_and_boxes_file_names = AugmentedImagesUtil.get_images_and_boxes_file_names_from_folder(folder_images,
                                                                                                      folder_boxes,
                                                                                                      image_exts,
                                                                                                      box_ext)
        augmented_images = []
        for image_and_box_file_name in images_and_boxes_file_names:
            image_file, box_file = image_and_box_file_name
            augmented_images.append(AugmentedImage.from_files(folder_images + image_file, folder_boxes + box_file, grayscale=grayscale))
        return augmented_images

    @staticmethod
    def images_and_boxes_to_folder(folder_images_input, folder_images_output, image_size=(416, 416), grayscale=True, invert=False):
        from src.AugmentedImage import AugmentedImage
        import os

        os.makedirs(folder_images_output, exist_ok=True)

        augmented_images = AugmentedImagesUtil.from_folders(folder_images_input, folder_images_input, grayscale=False)

        for index, augmented_image in enumerate(augmented_images):
            image = augmented_image.get_image()
            boxes = augmented_image.get_boxes()

            AugmentedImage.image_to_file(image,
                                         image_path=folder_images_output + str(index) + ".jpg",
                                         image_size=image_size,
                                         grayscale=grayscale,
                                         invert=invert)

            AugmentedImage.boxes_to_file(boxes,
                                         folder_images_output + str(index) + ".txt",
                                         image_width=AugmentedImage.get_image_width(image),
                                         image_height=AugmentedImage.get_image_height(image))
