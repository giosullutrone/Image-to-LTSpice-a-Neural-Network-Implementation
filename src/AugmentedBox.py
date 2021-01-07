from src.Box import Box


class AugmentedBox(Box):
    """Class that extends Box that lets you manipulate the Box"""

    @staticmethod
    def from_box(box):
        return AugmentedBox(box.get_box_class(),
                            box.get_x(),
                            box.get_y(),
                            box.get_width(),
                            box.get_height())

    @staticmethod
    def from_string(box_string, image_width, image_height):
        """Overrides the Box method "from_string" and returns instead an AugmentedBox"""
        return AugmentedBox.from_box(Box.from_string(box_string, image_width, image_height))

    def rotate_box_around_a_point(self, x, y, rad):
        """Rotate the box around a point by rad, changing its box class if needed"""

        box_class_new = AugmentedBox.get_box_class_after_rotation_from_box(self, rad)
        x_new, y_new = AugmentedBox.get_box_center_after_rotation_around_point_from_box(self, x, y, rad)
        width, height = AugmentedBox.get_box_size_after_rotation_from_box(self, rad)
        self.set_box_class(box_class_new)
        self.set_x(x_new)
        self.set_y(y_new)
        self.set_width(width)
        self.set_height(height)

    def zoom_box_around_a_point(self, x, y, factor):
        """Zoom box by a factor, ex. 1.5 => 150%"""
        x_box = self.get_x()
        y_box = self.get_y()

        #Get distance from point
        x_distance = x - x_box
        y_distance = y - y_box

        #Scale the distance
        x_shift = x_distance * factor
        y_shift = y_distance * factor

        self.set_x(x - x_shift)
        self.set_y(y - y_shift)

    def flip_box_around_axis_vertically(self, y):
        """Flip box around an axis vertically, changing its box class if needed"""

        y_box = self.get_y()

        #Get distance from axis
        y_distance = y - y_box

        #Scale the distance
        y_shift = y_distance * -1

        self.set_y(y - y_shift)

        #Change the box class if needed
        self.set_box_class(AugmentedBox.get_box_class_after_flip_vertical_from_box(self))

    def flip_box_around_axis_horizontally(self, x):
        """Flip box around an axis horizontally, changing its box class if needed"""

        x_box = self.get_x()

        #Get distance from axis
        x_distance = x - x_box

        #Scale the distance
        x_shift = x_distance * -1

        self.set_x(x - x_shift)

        #Change the box class if needed
        self.set_box_class(AugmentedBox.get_box_class_after_flip_horizontal_from_box(self))

    @staticmethod
    def get_box_class_after_rotation_from_box(box, rad):
        """Returns the new box class of a box rotated by rad (in radians)"""
        return AugmentedBox.get_box_class_after_rotation_from_box_class(box.get_box_class(), rad)

    @staticmethod
    def get_box_class_after_rotation_from_box_class(box_class, rad):
        """Returns the new box class of a box rotated by rad (in radians)"""

        from src.CircuitObject import CircuitObject

        circuit_object = CircuitObject(box_class)
        circuit_object.rotate(rad)
        return circuit_object.get_box_class()

    @staticmethod
    def get_box_class_after_flip_horizontal_from_box(box):
        """Returns the new box class of a box flipped horizontally"""
        return AugmentedBox.get_box_class_after_flip_horizontal_from_box_class(box.get_box_class())

    @staticmethod
    def get_box_class_after_flip_horizontal_from_box_class(box_class):
        """Returns the new box class of a box flipped horizontally"""

        from src.CircuitObject import CircuitObject

        circuit_object = CircuitObject(box_class)
        circuit_object.flip_horizontally()
        return circuit_object.get_box_class()

    @staticmethod
    def get_box_class_after_flip_vertical_from_box(box):
        """Returns the new box class of a box flipped vertically"""
        return AugmentedBox.get_box_class_after_flip_vertical_from_box_class(box.get_box_class())

    @staticmethod
    def get_box_class_after_flip_vertical_from_box_class(box_class):
        """Returns the new box class of a box flipped vertically"""

        from src.CircuitObject import CircuitObject

        circuit_object = CircuitObject(box_class)
        circuit_object.flip_vertically()
        return circuit_object.get_box_class()

    @staticmethod
    def __get_point_after_rotation_around_point_from_point(x, y, center_x, center_y, rad):
        """Returns the new point rotated around a point by rad (in radians)"""
        import math
        rad *= -1
        x_new = center_x + math.cos(rad) * (x - center_x) - math.sin(rad) * (y - center_y)
        y_new = center_y + math.sin(rad) * (x - center_x) + math.cos(rad) * (y - center_y)
        return (x_new, y_new)

    @staticmethod
    def get_box_center_after_rotation_around_point_from_box(box, x, y, rad):
        """Returns the new point rotated around a point by rad (in radians)"""
        return AugmentedBox.get_box_center_after_rotation_around_point_from_point(box.get_x(), box.get_y(), x, y, rad)

    @staticmethod
    def get_box_center_after_rotation_around_point_from_point(box_x, box_y, x, y, rad):
        """Returns the new point rotated around a point by rad (in radians)"""
        return AugmentedBox.__get_point_after_rotation_around_point_from_point(box_x, box_y, x, y, rad)

    @staticmethod
    def get_box_size_after_rotation_from_box(box, rad):
        """Returns the new bounding box of a box rotated by rad (in radians)"""
        return AugmentedBox.get_box_size_after_rotation_from_size(box.get_width(), box.get_height(), rad)

    @staticmethod
    def get_box_size_after_rotation_from_size(width, height, rad):
        """Returns the new bounding box of a box rotated by rad (in radians)"""
        import math
        width_new = width * abs(math.cos(rad)) + height * abs(math.sin(rad))
        height_new = width * abs(math.sin(rad)) + height * abs(math.cos(rad))
        return (width_new, height_new)
