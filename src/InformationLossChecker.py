class InformationLossChecker:
    @staticmethod
    def box_centers_out_of_bounds(augmented_image):
        """Returns True if any box is out of bounds else returns False"""

        from src.AugmentedImage import AugmentedImage

        image_width = AugmentedImage.get_image_width(augmented_image.get_image())
        image_height = AugmentedImage.get_image_height(augmented_image.get_image())

        boxes = augmented_image.get_boxes()

        centers = boxes.get_box_centers()

        for center in centers:
            box_x, box_y = center

            if ((box_x < 0 or box_x > image_width) or
               (box_y < 0 or box_y > image_height)):
                return True
        return False
