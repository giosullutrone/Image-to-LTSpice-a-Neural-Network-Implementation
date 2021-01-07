if __name__ == "__main__":
    import argparse
    import os
    from src.IdentificationNetwork import IdentificationNetwork
    from src.TrackingNetwork import TrackingNetwork
    from src.AugmentedImagesUtil import AugmentedImagesUtil
    from src.LTSpiceGrid import LTSpiceGrid
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    import src.nets

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="Location of the \"Input\" folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Location of the \"Output\" folder", type=str, required=True)
    parser.add_argument("-m", "--models_folder", help="From where to load the models' weights", type=str, required=True)

    parser.add_argument("-c", "--confidence", help="Confidence threshold for the prediction", type=float, required=False, default=0.75)

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)
    models_folder = get_fixed_path(args.models_folder, replace_backslash=True, add_backslash=True)

    ####################################################################################################################
    # Tracking network
    ####################################################################################################################
    K.clear_session()

    net_tracking = TrackingNetwork(src.nets.generate_tracking_model(input_shape=(416, 416, 1),
                                                                    output_shape=(13, 13, 11),
                                                                    number_of_classes=6))
    net_tracking.load_weights(models_folder + "tracking.h5")

    ####################################################################################################################
    # Identification network
    ####################################################################################################################

    net_identification = IdentificationNetwork(models=(
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=9),
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=4),
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2),
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2),
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=4),
        src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2)
    ))

    net_identification.load_all_weights(models_folder + "identification.h5", number_of_models=6)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################
    image_file_names = AugmentedImagesUtil.get_images_file_names_from_folder(input_folder, image_exts=(".jpg", ".png"))

    for image_file_name in image_file_names:
        boxes_tracking = net_tracking.predict_boxes_from_file(input_folder + image_file_name, confidence=args.confidence)

        boxes_identification = net_identification.predict_boxes_from_file_and_tracking_output(image_path=input_folder + image_file_name,
                                                                                              boxes=boxes_tracking,
                                                                                              number_of_models=6,
                                                                                              image_input_size=(50, 50))
        ltspice = LTSpiceGrid.from_file_and_boxes(image_path=input_folder + image_file_name,
                                                  boxes=boxes_identification,
                                                  subdivisions=13)
        os.makedirs(output_folder, exist_ok=True)
        ltspice.to_file(output_folder + image_file_name.split(".")[0] + ".asc")
