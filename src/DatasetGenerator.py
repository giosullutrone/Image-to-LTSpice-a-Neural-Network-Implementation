class DatasetGenerator:
    def __init__(self, folder_images, folder_boxes):
        self.__folder_images = folder_images
        self.__folder_boxes = folder_boxes
        self.__image_exts = (".jpg", ".png")
        self.__box_ext = ".txt"

    def set_exts(self, image_exts=(".jpg", ".png"), box_ext=(".txt")):
        self.__image_exts = image_exts
        self.__box_ext = box_ext

    def generate_pre_tracking_dataset(self, folder_images_output, number_of_images):
        import random
        from src.AugmentedImage import AugmentedImage
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        from src.CircuitObject import CircuitObject
        import cv2
        import os
        import numpy as np

        images_and_boxes_file_names = AugmentedImagesUtil.get_images_and_boxes_file_names_from_folder(self.__folder_images,
                                                                                                 self.__folder_boxes,
                                                                                                 self.__image_exts,
                                                                                                 self.__box_ext)
        done = 0

        while done < number_of_images:
            image_file, box_file = random.choice(images_and_boxes_file_names)
            augmented_image = AugmentedImage.from_files(self.__folder_images + image_file, self.__folder_boxes + box_file)

            #Get sections from the image of each box in the image
            image = augmented_image.get_image()
            boxes = augmented_image.get_boxes()
            rectangles = boxes.get_topleft_boxes_data()

            for rect in rectangles:
                box_class, x, y, width, height = rect

                if CircuitObject.is_an_unknown_object(box_class):
                    continue

                x = max(0, x)
                y = max(0, y)

                #Create a black image with the same resolution
                image_new = np.zeros_like(image)
                #Add the image section, pertain to the box, to the new black image
                image_new[int(y):int(y+height), int(x):int(x+width)] = image[int(y):int(y+height), int(x):int(x+width)]

                #Save it in the correct folder
                from src.CircuitObject import CircuitObject
                image_path_new = folder_images_output + f"{CircuitObject.get_type_from_box_class(box_class)}/"
                os.makedirs(image_path_new, exist_ok=True)

                image_path_new += f"{done}." + image_file.split(".")[1]

                cv2.imwrite(image_path_new, image_new)

            done += 1

    def generate_tracking_dataset(self, folder_images_output, folder_grids_output, number_of_images):
        import random
        from src.AugmentedImage import AugmentedImage
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        from src.GridBoxesUtil import GridBoxesUtil
        import os

        images_and_boxes_file_names = AugmentedImagesUtil.get_images_and_boxes_file_names_from_folder(self.__folder_images,
                                                                                                      self.__folder_boxes,
                                                                                                      self.__image_exts,
                                                                                                      self.__box_ext)
        os.makedirs(folder_images_output, exist_ok=True)
        os.makedirs(folder_grids_output, exist_ok=True)
        done = 0

        while done < number_of_images:
            image_file, box_file = random.choice(images_and_boxes_file_names)
            augmented_image = AugmentedImage.from_files(self.__folder_images + image_file, self.__folder_boxes + box_file)

            grid = GridBoxesUtil.to_grid_from_augmented_image(augmented_image)
            ########################
            # The method to_grid_from_augmented_image returns None if a collision has happened
            # therefore we skip this grid
            ########################
            if grid is None:
                continue

            image_path_new = folder_images_output
            grid_path_new = folder_grids_output

            image_path_new += f"{done}." + image_file.split(".")[1]
            grid_path_new += f"{done}." + box_file.split(".")[1]

            AugmentedImage.image_to_file(augmented_image.get_image(), image_path_new)
            GridBoxesUtil.to_file(grid, grid_path_new)

            done += 1

    def generate_identification_dataset(self, folder_images_output, number_of_images, image_output_size=50, translation_values=(-0.01, 0.01), deformation_values=(-0.01, 0.01)):
        import random
        from src.AugmentedImage import AugmentedImage
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        from src.CircuitObject import CircuitObject
        import cv2
        import os

        images_and_boxes_file_names = AugmentedImagesUtil.get_images_and_boxes_file_names_from_folder(self.__folder_images,
                                                                                                      self.__folder_boxes,
                                                                                                      self.__image_exts,
                                                                                                      self.__box_ext)
        done = 0

        while done < number_of_images:
            image_file, box_file = random.choice(images_and_boxes_file_names)
            augmented_image = AugmentedImage.from_files(self.__folder_images + image_file, self.__folder_boxes + box_file)

            #Get sections from the image of each box in the image
            image = augmented_image.get_image()
            boxes = augmented_image.get_boxes()
            rectangles = boxes.get_topleft_boxes_data()

            for rect in rectangles:
                box_class, x, y, width, height = rect

                if CircuitObject.is_an_unknown_object(box_class) or not CircuitObject.is_an_object(box_class):
                    continue

                import random
                x_translation = int(x + x * random.uniform(translation_values[0], translation_values[1]))
                y_translation = int(y + y * random.uniform(translation_values[0], translation_values[1]))

                width_deformation = int(width + width * random.uniform(deformation_values[0], deformation_values[1]))
                height_deformation = int(height + height * random.uniform(deformation_values[0], deformation_values[1]))

                x_translation = max(0, x_translation)
                y_translation = max(0, y_translation)

                x_translation = min(image.shape[1], x_translation)
                y_translation = min(image.shape[0], y_translation)

                width_deformation = min(image.shape[1] - x_translation, width_deformation)
                height_deformation = min(image.shape[0] - y_translation, height_deformation)

                image_section = image[y_translation: y_translation+height_deformation, x_translation: x_translation+width_deformation]

                #Resize to match output size
                image_section = cv2.resize(image_section, (image_output_size, image_output_size))

                #Save it in the correct folder
                box_type = CircuitObject.get_type_from_box_class(box_class)
                box_class -= CircuitObject.get_box_class_from_type(box_type)

                image_path_new = folder_images_output + f"/{box_type}/{box_class}/"
                os.makedirs(image_path_new, exist_ok=True)

                image_path_new += f"{done}." + image_file.split(".")[1]

                cv2.imwrite(image_path_new, image_section)

            done += 1
