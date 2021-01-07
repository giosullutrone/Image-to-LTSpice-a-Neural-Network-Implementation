class Box:
    def __init__(self, box_class, x, y, width, height):
        assert isinstance(box_class, int), "Box class is not an int"

        self.__box_class = box_class
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    @staticmethod
    def from_perc(box_class, x_perc, y_perc, width_perc, height_perc, image_width, image_height):
        """Returns an instance of Box from values in [0.0, 1.0]"""
        return Box(box_class,
                   x_perc * image_width,
                   y_perc * image_height,
                   width_perc * image_width,
                   height_perc * image_height)

    @staticmethod
    def from_string(box_string, image_width, image_height):
        """Returns an instance of Box from a YOLO formatted string"""
        box_class, x_prec, y_perc, width_perc, height_perc = box_string.split(" ")
        return Box(int(box_class),
                   float(x_prec) * image_width,
                   float(y_perc) * image_height,
                   float(width_perc) * image_width,
                   float(height_perc) * image_height)

    def to_string(self, image_width, image_height):
        return f"{self.__box_class} {self.__x / image_width} {self.__y / image_height} {self.__width / image_width} {self.__height / image_height}"

    def get_box_class(self):
        return self.__box_class

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_width(self):
        return self.__width

    def get_height(self):
        return self.__height

    def set_box_class(self, box_class):
        self.__box_class = box_class

    def set_x(self, x):
        self.__x = x

    def set_y(self, y):
        self.__y = y

    def set_width(self, width):
        self.__width = width

    def set_height(self, height):
        self.__height = height

    def get_box_data(self):
        """Returns tuple of (box_class, x, y, width, height)"""
        return (self.__box_class, self.__x, self.__y, self.__width, self.__height)

    def get_box_area(self):
        return self.__width * self.__height

    def get_topleft_box_data(self):
        """Returns tuple of (box_class, x, y, width, height) where x and y are topleft centered"""
        box_class, x, y, width, height = self.get_box_data()
        return (box_class, x - width / 2, y - height / 2, width, height)

    @staticmethod
    def get_box_data_from_perc(box_class, x_perc, y_perc, width_perc, height_perc, image_width, image_height):
        """Returns tuple of (box_class, x, y, width, height)"""
        return (Box.from_perc(box_class, x_perc, y_perc, width_perc, height_perc, image_width, image_height).get_box_data())

    @staticmethod
    def get_box_data_from_string(box_string, image_width, image_height):
        """Returns tuple of (box_class, x, y, width, height)"""
        return (Box.from_string(box_string, image_width, image_height).get_box_data())

    @staticmethod
    def get_iou(box_first, box_second):
        __, box_first_x, box_first_y, box_first_width, box_first_height = box_first.get_topleft_box_data()
        __, box_second_x, box_second_y, box_second_width, box_second_height = box_second.get_topleft_box_data()

        box_intersection_x = max(box_first_x, box_second_x)
        box_intersection_y = max(box_first_y, box_second_y)

        box_intersection_width = min(box_first_x + box_first_width, box_second_x + box_second_width) - box_intersection_x
        box_intersection_height = min(box_first_y + box_first_height, box_second_y + box_second_height) - box_intersection_y

        if box_intersection_width < 0 or box_intersection_height < 0:
            return 0.0

        box_intersection = Box(-1, box_intersection_x, box_intersection_y, box_intersection_width, box_intersection_height)

        return box_intersection.get_box_area() / (box_first.get_box_area() + box_second.get_box_area() - box_intersection.get_box_area())

    @staticmethod
    def get_encompassing_box(box_first, box_second):
        box_first_class, box_first_x, box_first_y, box_first_width, box_first_height = box_first.get_topleft_box_data()
        __, box_second_x, box_second_y, box_second_width, box_second_height = box_second.get_topleft_box_data()

        box_encompassing_x = min(box_first_x, box_second_x)
        box_encompassing_y = min(box_first_y, box_second_y)
        box_encompassing_width = max(box_first_x + box_first_width, box_second_x + box_second_width) - box_encompassing_x
        box_encompassing_height = max(box_first_y + box_first_height, box_second_y + box_second_height) - box_encompassing_y

        return Box(box_first_class, box_encompassing_x + box_encompassing_width/2, box_encompassing_y + box_encompassing_height/2, box_encompassing_width, box_encompassing_height)
