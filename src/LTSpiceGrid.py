class LTSpiceGrid:
    def __init__(self, components):
        self.__components = components

    @staticmethod
    def __get_line_as_ltspice_string(first_component, second_component):
        first_component_x = first_component.get_component_x()
        first_component_y = first_component.get_component_y()
        second_component_x = second_component.get_component_x()
        second_component_y = second_component.get_component_y()

        x_min = min(first_component_x, second_component_x)
        x_max = max(first_component_x, second_component_x)
        y_min = min(first_component_y, second_component_y)
        y_max = max(first_component_y, second_component_y)

        return f"WIRE {x_max * 80 + 48} {y_max * 80 + 48} {x_min * 80 + 48} {y_min * 80 + 48}"

    def __get_lines_as_ltspice_strings(self):
        import math

        lines = []

        corners = [x for x in self.__components if x.is_corner()]

        for corner in corners:
            if corner.can_top():
                min_distance = math.inf
                min_distance_corner = None

                for other in [x for x in corners if (x.can_bottom() and x.get_component_x() == corner.get_component_x() and corner != x)]:
                    distance = other.get_component_y() - corner.get_component_y()

                    if distance < min_distance and distance < 0:
                        min_distance = distance
                        min_distance_corner = other

                if min_distance_corner is not None:
                    corner.set_top(min_distance_corner)
                    min_distance_corner.set_bottom(corner)
                    lines.append(LTSpiceGrid.__get_line_as_ltspice_string(corner, min_distance_corner))
            ############################################################################################################
            if corner.can_bottom():
                min_distance = math.inf
                min_distance_corner = None

                for other in [x for x in corners if (x.can_top() and x.get_component_x() == corner.get_component_x() and corner != x)]:
                    distance = other.get_component_y() - corner.get_component_y()

                    if distance < min_distance and distance > 0:
                        min_distance = distance
                        min_distance_corner = other

                if min_distance_corner is not None:
                    corner.set_bottom(min_distance_corner)
                    min_distance_corner.set_top(corner)
                    lines.append(LTSpiceGrid.__get_line_as_ltspice_string(corner, min_distance_corner))
            ############################################################################################################
            if corner.can_left():
                min_distance = math.inf
                min_distance_corner = None

                for other in [x for x in corners if (x.can_right() and x.get_component_y() == corner.get_component_y() and corner != x)]:
                    distance = other.get_component_x() - corner.get_component_x()

                    if distance < min_distance and distance < 0:
                        min_distance = distance
                        min_distance_corner = other

                if min_distance_corner is not None:
                    corner.set_left(min_distance_corner)
                    min_distance_corner.set_right(corner)
                    lines.append(LTSpiceGrid.__get_line_as_ltspice_string(corner, min_distance_corner))
            ############################################################################################################
            if corner.can_right():
                min_distance = math.inf
                min_distance_corner = None

                for other in [x for x in corners if (x.can_left() and x.get_component_y() == corner.get_component_y() and corner != x)]:
                    distance = other.get_component_x() - corner.get_component_x()

                    if distance < min_distance and distance > 0:
                        min_distance = distance
                        min_distance_corner = other

                if min_distance_corner is not None:
                    corner.set_right(min_distance_corner)
                    min_distance_corner.set_left(corner)
                    lines.append(LTSpiceGrid.__get_line_as_ltspice_string(corner, min_distance_corner))
            ############################################################################################################
        return lines

    def __get_components_as_ltspice_strings(self):
        strings = []
        components = [x for x in self.__components if not x.is_corner()]

        index = 0

        for component in components:
            s = component.get_ltspice_component(name=index)
            if s is not None:
                strings.append(s)
                index += 1
        return strings

    def to_boxes(self, image_width, image_height, subdivisions=13):
        from src.Boxes import Boxes
        from src.Box import Box

        boxes = Boxes()

        #Get the size of each grid cell
        step = image_width / subdivisions

        for component in self.__components:
            box_class = component.get_component_class()
            box_x = component.get_component_x() * step
            box_y = component.get_component_y() * step
            box_width = step
            box_height = step

            boxes.add_box(Box(box_class=box_class, x=box_x, y=box_y, width=box_width, height=box_height))
        return boxes

    @staticmethod
    def from_boxes(boxes, image_width, image_height, subdivisions=13):
        from src.LTSpiceComponent import LTSpiceComponent

        assert image_width == image_height, "Image width and height are not equal..."

        components = []

        #Get the size of each grid cell
        step = image_width / subdivisions

        box_classes = boxes.get_box_classes()
        box_centers = boxes.get_box_centers()

        for box_class, box_center in zip(box_classes, box_centers):
            box_x, box_y = box_center

            #Find the cell that contains the center of the box
            cell_x = int(box_x // step)
            cell_y = int(box_y // step)

            components.append(LTSpiceComponent(component_class=box_class, component_x=cell_x, component_y=cell_y))
        return LTSpiceGrid(components)

    @staticmethod
    def from_file_and_boxes(image_path, boxes, subdivisions=13):
        from src.AugmentedImage import AugmentedImage

        image = AugmentedImage.image_from_file(image_path, grayscale=True)
        image_width = AugmentedImage.get_image_width(image)
        image_height = AugmentedImage.get_image_height(image)

        return LTSpiceGrid.from_boxes(boxes, image_width, image_height, subdivisions)

    @staticmethod
    def from_augmented_image(augmented_image, subdivisions=13):
        return LTSpiceGrid.from_boxes(boxes=augmented_image.get_boxes(),
                                      image_width=augmented_image.get_image_width(augmented_image.get_image()),
                                      image_height=augmented_image.get_image_height(augmented_image.get_image()),
                                      subdivisions=subdivisions)

    def to_file(self, file_path, subdivisions=13):
        with open(file_path, "w") as f:
            f.write("Version 4\nSHEET 1 {} {}\n".format(subdivisions * 80, subdivisions * 80))

            strings = self.__get_lines_as_ltspice_strings() + self.__get_components_as_ltspice_strings()

            for s in strings:
                f.write(s + "\n")
