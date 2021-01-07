class AugmentedImage:
    """Class that lets you manipualte an image with its boxes"""

    def __init__(self, image, boxes):
        """Generate augmented image from cv2 image and Boxes (with AugmentedBox inside)"""
        self.__base_image = image
        self.__image = image

        self.__boxes = boxes

        #Keep track of the image rotation in order to correctly change the box class if needed
        self.__is_rotated = False

    def get_image(self):
        return self.__image

    def get_boxes(self):
        return self.__boxes

    @staticmethod
    def get_image_width(image):
        return image.shape[1]

    @staticmethod
    def get_image_height(image):
        return image.shape[0]

    @staticmethod
    def image_from_file(image_path, grayscale=True):
        import cv2

        if grayscale:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
        return image

    @staticmethod
    def from_files(image_path, box_path, grayscale=True):
        """Generate an augmented image and boxes from file"""
        from src.Boxes import Boxes
        image = AugmentedImage.image_from_file(image_path, grayscale)
        boxes = Boxes.from_files(image_path, box_path, augmented=True)
        return AugmentedImage(image, boxes)

    def zoom_image(self, factor):
        """Zoom image and boxes by a factor, ex. factor=1.5 => 150%"""
        import cv2

        assert factor > 0.0, "Zoom factor should not be negative or zero..."

        #Just return if factor is 1.0
        if factor == 1.0:
            return

        image_width = AugmentedImage.get_image_width(self.__image)
        image_height = AugmentedImage.get_image_height(self.__image)

        #Zoom boxes
        self.__boxes.zoom_boxes_around_a_point(image_width/2, image_height/2, factor)

        #Resize image by factor
        image_resized = cv2.resize(self.__image, (int(image_width * factor), int(image_height * factor)))
        resized_height, resized_width = image_resized.shape[0:2]

        if factor < 1.0:
            ########################
            # If factor < 1.0 then add border all around the image to bring it back
            # to the original size
            ########################

            #Get borders to add to image
            border_top = (image_height - resized_height) // 2
            border_bottom = image_height - resized_height - border_top

            border_left = (image_width - resized_width) // 2
            border_right = image_width - resized_width - border_left

            #Add black border to the resized image
            image_resized = cv2.copyMakeBorder(image_resized,
                                               border_top, border_bottom, border_left, border_right,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

        else:
            ########################
            # If factor > 1.0 then cut the image to bring it back to the
            #  original size
            ########################

            border_top = (resized_height - image_height) // 2
            border_bottom = resized_height - image_height - border_top

            border_left = (resized_width - image_width) // 2
            border_right = resized_width - image_width - border_left

            border_top = -resized_height if border_top==0 else border_top
            border_right = -resized_width if border_right==0 else border_right

            #Cut the image to the original size
            image_resized = image_resized[border_bottom:-border_top, border_left:-border_right, ...]

        self.__image = image_resized

    def rotate_image(self, rad):
        """Rotate image and boxes by rad non-destructvely"""

        import imutils
        from math import degrees
        from src.AugmentedBox import AugmentedBox

        #The CircuitObject used to change the box_class after a rotation, cannot take into account previous rotations,
        #leading to a box_class that may not be changed correctly, therefore only one rotation will be allowed
        assert not self.__is_rotated, "The image has already been rotated once, a second rotation may lead to unexpected behaviors"

        #Just return if rad is 0.0
        if rad == 0.0:
            return

        ########################
        # The pure rotation of an image will cut sections of the image itself,
        # therefore, we calculate the size after the rotation and scale the
        # image to that size so that after the rotation no information
        # will be lost
        ########################

        #Get the max increase in size needed (for the image) in order to not lose data after the rotation
        image_width = AugmentedImage.get_image_width(self.__image)
        image_height = AugmentedImage.get_image_height(self.__image)
        rotated_width, rotated_height = AugmentedBox.get_box_size_after_rotation_from_size(image_width, image_height, rad)
        zoom_factor = max(image_width / rotated_width, image_height / rotated_height)
        #Resize the image by the zoom factor before the rotation
        self.zoom_image(zoom_factor)

        #Rotate boxes by rad around the center of the zoomed image
        image_center_x = AugmentedImage.get_image_width(self.__image) / 2
        image_center_y = AugmentedImage.get_image_height(self.__image) / 2

        self.__boxes.rotate_boxes_around_a_point(image_center_x, image_center_y, rad)

        #Rotate the image by rad
        self.__image = imutils.rotate(self.__image, angle=degrees(rad))

    def flip_image_horizontally(self):
        """Flip image and boxes horizontally"""

        import cv2

        self.__image = cv2.flip(self.__image, 1)
        self.__boxes.flip_boxes_around_axis_horizontally(AugmentedImage.get_image_width(self.__image) / 2)

    def flip_image_vertically(self):
        """Flip image and boxes vertically"""

        import cv2

        self.__image = cv2.flip(self.__image, 0)
        self.__boxes.flip_boxes_around_axis_vertically(AugmentedImage.get_image_height(self.__image) / 2)

    def to_files(self, image_path, box_path, image_size=None, grayscale=False, invert=False):
        """Save the image and the boxes to the specified paths"""
        AugmentedImage.image_to_file(self.__image, image_path, image_size=image_size, grayscale=grayscale, invert=invert)
        AugmentedImage.boxes_to_file(self.__boxes, box_path, AugmentedImage.get_image_width(self.__image), AugmentedImage.get_image_height(self.__image))

    @staticmethod
    def image_to_file(image, image_path, image_size=None, grayscale=False, invert=False):
        import cv2

        if image_size is not None:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        try:
            if grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            pass
        if invert:
            image = cv2.bitwise_not(image)
        cv2.imwrite(image_path, image)

    @staticmethod
    def boxes_to_file(boxes, box_path, image_width, image_height):
        boxes = boxes.to_string(image_width, image_height)

        with open(box_path, "w") as f:
            f.write(boxes)
