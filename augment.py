if __name__ == "__main__":
    import argparse
    from src.GeneratorAugmentedImages import GeneratorAugmentedImages
    from src.Util import get_fixed_path
    from src.InformationLossChecker import InformationLossChecker

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="From where to load the images and boxes", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Where to save the images", type=str, required=True)

    parser.add_argument("-n", "--number_of_images", help="How many images to generate for each input image", type=int, required=True)

    parser.add_argument("-rp", "--rotation_probability", help="Probability of a rotation", type=float, required=False, default=1.0)
    parser.add_argument("-zp", "--zoom_probability", help="Probability of a zoom", type=float, required=False, default=1.0)
    parser.add_argument("-fvp", "--flip_vertical_probability", help="Probability of a vertical flip", type=float, required=False, default=0.25)
    parser.add_argument("-fhp", "--flip_horizontal_probability", help="Probability of a horizontal flip", type=float, required=False, default=0.25)

    parser.add_argument("-rv", "--rotation_values", help="Tuple of rotation radians ex.(-0.26, 0.26)", type=tuple, required=False, default=(-0.26, 0.26))
    parser.add_argument("-zv", "--zoom_values", help="Tuple of zoom values ex.(-0.01, 0.01) => -1% / 1%", type=tuple, required=False, default=(-0.01, 0.01))

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)

    gen = GeneratorAugmentedImages(input_folder,
                                   input_folder,
                                   rotation_probability=args.rotation_probability,
                                   zoom_probability=args.zoom_probability,
                                   flip_vertical_probability=args.flip_horizontal_probability,
                                   flip_horizontal_probability=args.flip_horizontal_probability,
                                   rotation_values=args.rotation_values,
                                   zoom_values=(1.0 + args.zoom_values[0], 1.0 + args.zoom_values[1]),
                                   seed=42)

    gen.generate_augmented_images_to_folder(output_folder,
                                            number_of_images=args.number_of_images,
                                            image_size=(416, 416),
                                            information_loss_checker_method=InformationLossChecker.box_centers_out_of_bounds)
