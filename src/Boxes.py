class Boxes:
    """Class that collects Box or AugmentedBox that lets you call functions on all its collected elemets"""

    def __init__(self):
        self.__boxes = []

    def add_box(self, box):
        self.__boxes.append(box)

    def get_box_classes(self):
        """Retuns a list of all box classes"""
        box_classes = []
        for box in self.__boxes:
            box_classes.append(box.get_box_class())
        return box_classes

    def get_box_centers(self):
        """Retuns a list of all box centers"""
        centers = []
        for box in self.__boxes:
            centers.append((box.get_x(), box.get_y()))
        return centers

    def get_box_sizes(self):
        """Retuns a list of all box sizes"""
        sizes = []
        for box in self.__boxes:
            sizes.append((box.get_width(), box.get_height()))
        return sizes

    def rotate_boxes_around_a_point(self, x, y, rad):
        from src.AugmentedBox import AugmentedBox

        for box in self.__boxes:
            assert isinstance(box, AugmentedBox), "Rotation on a Box cannot be done...Convert it to AugmentedBox?"
            box.rotate_box_around_a_point(x, y, rad)

    def zoom_boxes_around_a_point(self, x, y, factor):
        from src.AugmentedBox import AugmentedBox

        for box in self.__boxes:
            assert isinstance(box, AugmentedBox), "Zoom on a Box cannot be done...Convert it to AugmentedBox?"
            box.zoom_box_around_a_point(x, y, factor)

    def flip_boxes_around_axis_vertically(self, y):
        from src.AugmentedBox import AugmentedBox

        for box in self.__boxes:
            assert isinstance(box, AugmentedBox), "Flip on a Box cannot be done...Convert it to AugmentedBox?"
            box.flip_box_around_axis_vertically(y)

    def flip_boxes_around_axis_horizontally(self, x):
        from src.AugmentedBox import AugmentedBox

        for box in self.__boxes:
            assert isinstance(box, AugmentedBox), "Flip on a Box cannot be done...Convert it to AugmentedBox?"
            box.flip_box_around_axis_horizontally(x)

    def convert_boxes_to_augmented(self):
        from src.AugmentedBox import AugmentedBox

        for i in range(len(self.__boxes)):
            self.__boxes[i] = AugmentedBox.from_box(self.__boxes[i])

    def get_topleft_boxes_data(self):
        boxes_data = []
        for box in self.__boxes:
            boxes_data.append(box.get_topleft_box_data())
        return boxes_data

    def to_string(self, image_width, image_height):
        """Returns a string containing all the to_string of its boxes"""
        to_string = ""
        for box in self.__boxes:
            to_string += box.to_string(image_width, image_height) + "\n"
        return to_string

    def to_strings(self, image_width, image_height):
        """Returns a collection of to_string from its boxes"""
        to_strings = []
        for box in self.__boxes:
            to_strings.append(box.to_string(image_width, image_height))
        return to_strings

    @staticmethod
    def from_files(image_path, box_path, augmented=True):
        import cv2
        from src.AugmentedBox import AugmentedBox
        from src.Box import Box

        boxes = Boxes()

        image = cv2.imread(image_path)
        image_height, image_width = image.shape[0:2]

        with open(box_path, "r") as f:
            box_strings = f.readlines()

            for box_string in box_strings:
                if augmented:
                    boxes.add_box(AugmentedBox.from_string(box_string, image_width, image_height))
                else:
                    boxes.add_box(Box.from_string(box_string, image_width, image_height))
        return boxes

    def clean_boxes(self, iou_threshold=0.2):
        from src.Box import Box

        to_remove = []

        for i in range(0, len(self.__boxes)):
            for j in range(i+1, len(self.__boxes)-1):

                box_first = self.__boxes[i]
                box_second = self.__boxes[j]

                if box_first.get_box_class() == box_second.get_box_class():

                    if Box.get_iou(box_first, box_second) > iou_threshold:
                        self.add_box(Box.get_encompassing_box(box_first, box_second))

                        if box_first not in to_remove:
                            to_remove.append(box_first)
                        if box_second not in to_remove:
                            to_remove.append(box_second)

        for box in to_remove:
            self.__boxes.remove(box)
        return self
