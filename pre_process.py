if __name__ == "__main__":
    import argparse
    from src.AugmentedImagesUtil import AugmentedImagesUtil
    from src.Util import get_fixed_path

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="From where to load the images", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Where to save the images", type=str, required=True)

    parser.add_argument("-inv", "--invert", help="Whether to invert the image or not", type=bool, required=False, default=False)

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)

    AugmentedImagesUtil.images_and_boxes_to_folder(folder_images_input=input_folder,
                                                   folder_images_output=output_folder,
                                                   grayscale=True,
                                                   image_size=(416, 416),
                                                   invert=args.invert)
