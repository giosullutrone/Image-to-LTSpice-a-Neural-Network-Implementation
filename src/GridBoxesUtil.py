class GridBoxesUtil:
    @staticmethod
    def to_grid(image, boxes, subdivisions=13):
        """Returns an array of shape subdivisions*subdivisions*5 where each cell
        can only contain one object"""

        from src.CircuitObject import CircuitObject
        from src.AugmentedImage import AugmentedImage

        ########################
        # Create a 2D grid of subdivisions*subdivisions*5
        # Where the tuple of size 5 is => (o, x, y, w, h)
        #    Where o => is there an object or not
        #          x => box x in relation with its grid cell
        #          y => box y in relation with its grid cell
        #          w => box width in relation with its grid cell
        #          h => box height in relation with its grid cell
        ########################

        import numpy as np

        assert AugmentedImage.get_image_width(image) == AugmentedImage.get_image_height(image), "Image width and height are not equal..."

        grid = np.full((subdivisions, subdivisions, 5 + CircuitObject.NUMBER_OF_CLASSES), 0.0)

        #Get the size of each grid cell
        step = AugmentedImage.get_image_width(image) / subdivisions

        box_classes = boxes.get_box_classes()
        box_centers = boxes.get_box_centers()
        box_sizes = boxes.get_box_sizes()

        for box_class, box_center, box_size in zip(box_classes, box_centers, box_sizes):
            box_x, box_y = box_center
            box_width, box_height = box_size

            #todo: Fix this problem, both in the generator and here
            # -> skip what is not an object

            #Check if box_class is associated with an unknown object and if it is, skip it
            if CircuitObject.is_an_unknown_object(box_class) or not CircuitObject.is_an_object(box_class):
                continue

            #Check if box_class is associated with a circuit object or not
            box_is_an_object = 1.0
            box_type = CircuitObject.get_type_from_box_class(box_class)

            #Find the cell that contains the center of the box
            cell_x = int(box_x // step)
            cell_y = int(box_y // step)

            #Make sure that there isn't already an object in that cell, else return None
            if grid[cell_y, cell_x, 0] != 0.0:
                return None

            box_x_new = (box_x - (cell_x * step)) / step
            box_y_new = (box_y - (cell_y * step)) / step

            box_width_new = box_width / (AugmentedImage.get_image_width(image) / 2)
            box_height_new = box_height / (AugmentedImage.get_image_height(image) / 2)

            grid[cell_y, cell_x, 0] = box_is_an_object
            grid[cell_y, cell_x, 1:3] = (box_x_new, box_y_new)
            grid[cell_y, cell_x, 3:5] = (box_width_new, box_height_new)
            grid[cell_y, cell_x, 5 + box_type] = 1.0
        return grid

    @staticmethod
    def to_boxes(grid, image_width, image_height, subdivisions=13, confidence=0.5):
        from src.Boxes import Boxes
        from src.Box import Box
        from src.CircuitObject import CircuitObject
        import numpy as np

        assert image_width == image_height, "Image width and height are not equal..."

        step = image_width / subdivisions

        boxes = Boxes()

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                box_class, box_x, box_y, box_width, box_height = grid[i, j, 0:5]
                box_class_new = CircuitObject.get_box_class_from_type(np.argmax(grid[i, j, 5:]))

                if box_class >= confidence:
                    box_x_new = box_x * step + (j * step)
                    box_y_new = box_y * step + (i * step)
                    box_width_new = box_width * (image_width / 2)
                    box_height_new = box_height * (image_height / 2)

                    boxes.add_box(Box(box_class_new, box_x_new, box_y_new, box_width_new, box_height_new))
        return boxes

    @staticmethod
    def to_grid_from_augmented_image(augmented_image, subdivisions=13):
        return GridBoxesUtil.to_grid(augmented_image.get_image(), augmented_image.get_boxes(), subdivisions=subdivisions)

    @staticmethod
    def to_grid_from_files(image_path, box_path, subdivisions=13):
        from src.AugmentedImage import AugmentedImage
        augmented_image = AugmentedImage.from_files(image_path, box_path)
        return GridBoxesUtil.to_grid_from_augmented_image(augmented_image, subdivisions=subdivisions)

    @staticmethod
    def to_file(grid, file_path):
        import numpy as np
        grid_flat = grid.flatten()
        np.savetxt(file_path, grid_flat)

    @staticmethod
    def to_grid_from_file(file_path, subdivisions=13):
        from src.CircuitObject import CircuitObject
        import numpy as np
        grid_flat = np.loadtxt(file_path)
        return np.reshape(grid_flat, (subdivisions, subdivisions, 5 + CircuitObject.NUMBER_OF_CLASSES))

    @staticmethod
    def get_images_and_grids_file_names_from_folder(folder_images, folder_grids, image_exts=(".png", ".jpg"), grid_ext=".txt"):
        """Returns the file names of images and boxes as a list of tuples"""

        import os

        image_files = [x for x in os.listdir(folder_images) if x.endswith(image_exts)]
        grid_files = [x.split(".")[0] + grid_ext for x in image_files]

        for grid_file in grid_files:
            assert os.path.isfile(folder_grids + grid_file), "Grid file does not exists for the given image..."

        images_and_grids_files_paths = []
        for image_file, grid_file in zip(image_files, grid_files):
            images_and_grids_files_paths.append((image_file, grid_file))
        return images_and_grids_files_paths
