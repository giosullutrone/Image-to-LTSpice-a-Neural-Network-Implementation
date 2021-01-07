if __name__ == "__main__":
    import argparse
    from src.DatasetGenerator import DatasetGenerator
    from src.Util import get_fixed_path

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="From where to load the images and boxes", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Where to save the images", type=str, required=True)

    parser.add_argument("-n", "--number_of_images", help="How many inputs to generate for each dataset", type=int, required=True)

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)

    gen = DatasetGenerator(folder_images=input_folder, folder_boxes=input_folder)

    gen.generate_pre_tracking_dataset(folder_images_output=output_folder + "PreTracking/",
                                      number_of_images=args.number_of_images)
    gen.generate_tracking_dataset(folder_images_output=output_folder + "Tracking/",
                                  folder_grids_output=output_folder + "Tracking/",
                                  number_of_images=args.number_of_images)
    gen.generate_identification_dataset(folder_images_output=output_folder + "Identification/",
                                        number_of_images=args.number_of_images)
