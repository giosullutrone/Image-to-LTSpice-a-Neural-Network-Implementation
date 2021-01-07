class Debugger:
    @staticmethod
    def show_boxes_from_image_and_boxes(image, boxes, color=(0, 0, 255), subdivisions=13):
        """Show and return an image with its boxes from the image (cv2) and its boxes (Boxes)"""

        import cv2

        #Get size of image from shape == (height, width, channels if there are)
        image_height, image_width = image.shape[0:2]
        assert image_height == image_width, "Image has different width and height..."

        step = image_height / subdivisions

        #Get the rectangles to draw on top of the image
        rectangles = boxes.get_topleft_boxes_data()

        for i in range(subdivisions):
            image = cv2.line(image, (int(i * step), 0), (int(i * step), image_height), color=color, thickness=1)
            image = cv2.line(image, (0, int(i * step)), (image_width, int(i * step)), color=color, thickness=1)

        #Draw rectangles on top of the image
        for rect in rectangles:
            box_class, x, y, width, height = rect
            image = cv2.rectangle(image, (int(x), int(y)), (int(x+width), int(y+height)), color)
            image = cv2.putText(image, str(box_class), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=color[::-1], thickness=2)
            image = cv2.circle(image, (int(x+width/2), int(y+height/2)), radius=3, color=color, thickness=-1)

        #Show image
        cv2.imshow("Debug", image)
        cv2.waitKey(0)

        return image

    @staticmethod
    def show_boxes_from_files(image_path, box_path, color=(0, 0, 255)):
        """Show and return an image with its boxes from the image file and from its boxes file"""

        import os
        import cv2
        from src.Boxes import Boxes

        #Making sure that the files exist
        assert (os.path.isfile(image_path) and os.path.isfile(box_path)), "File not found..."

        #Get image and boxes from files
        image = cv2.imread(image_path)
        boxes = Boxes.from_files(image_path, box_path, augmented=False)

        return Debugger.show_boxes_from_image_and_boxes(image, boxes, color)

    @staticmethod
    def show_boxes_from_files_grid(image_path, box_path, color=(0, 0, 255), subdivisions=13):
        """Show and return an image with its boxes from the image file and from its boxes file"""

        import os
        import cv2
        from src.Boxes import Boxes
        from src.GridBoxesUtil import GridBoxesUtil
        from src.AugmentedImage import AugmentedImage

        #Making sure that the files exist
        assert (os.path.isfile(image_path) and os.path.isfile(box_path)), "File not found..."

        #Get image and boxes from files
        image = cv2.imread(image_path)
        grid = GridBoxesUtil.to_grid_from_file(file_path=box_path)
        boxes = GridBoxesUtil.to_boxes(grid, image_width=AugmentedImage.get_image_width(image), image_height=AugmentedImage.get_image_height(image))

        return Debugger.show_boxes_from_image_and_boxes(image, boxes, color)

    @staticmethod
    def show_boxes_from_augmented_image(augmented_image, color=(0, 0, 255), subdivisions=13):
        """Show and return an image with its boxes from its augmented image"""
        return Debugger.show_boxes_from_image_and_boxes(augmented_image.get_image(), augmented_image.get_boxes(), color=color)

    @staticmethod
    def show_boxes_from_image_and_grid_boxes(image, grid_boxes, color=(0, 0, 255), subdivisions=13):
        """Show and return an image with its boxes from the image (cv2) and its grid_boxes (numpy array)"""
        from src.GridBoxesUtil import GridBoxesUtil

        boxes = GridBoxesUtil.to_boxes(grid_boxes, image.shape[1], image.shape[0])
        return Debugger.show_boxes_from_image_and_boxes(image, boxes, color=color)
