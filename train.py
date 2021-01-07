if __name__ == "__main__":
    import argparse
    from src.IdentificationNetwork import IdentificationNetwork
    from src.TrackingNetwork import TrackingNetwork
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    import src.nets

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--datasets_folder", help="Location of the \"Datasets\" folder", type=str, required=True)
    parser.add_argument("-o", "--save_folder", help="Where to save the models", type=str, required=True)

    parser.add_argument("-l", "--load_folder", help="If and from where to load the models' weights", type=str,
                        required=False, default=None)

    parser.add_argument("-pte", "--pre_tracking_epochs", help="Epochs for the pre tracking network", type=int,
                        required=False, default=20)
    parser.add_argument("-te", "--tracking_epochs", help="Epochs for the tracking network", type=int,
                        required=False, default=50)
    parser.add_argument("-ie", "--identification_epochs", help="Epochs for the identification network", type=int,
                        required=False, default=25)

    parser.add_argument("-spe", "--steps_per_epoch", help="Steps per epoch", type=int, required=False, default=128)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, required=False, default=2)

    parser.add_argument("-dc", "--do_checkpoint", help="Should save checkpoints", type=bool, required=False, default=True)

    args = parser.parse_args()

    datasets_folder = get_fixed_path(args.datasets_folder, replace_backslash=True, add_backslash=True)
    save_folder = get_fixed_path(args.save_folder, replace_backslash=True, add_backslash=True)
    load_folder = get_fixed_path(args.load_folder, replace_backslash=True, add_backslash=True) if args.load_folder is not None else None

    ####################################################################################################################
    # Pre-tracking network
    ####################################################################################################################
    if args.pre_tracking_epochs > 0:
        K.clear_session()

        if load_folder is None:
            net_pre_tracking = IdentificationNetwork(src.nets.generate_pre_tracking_model(input_shape=(416, 416, 1), number_of_classes=7))

            net_pre_tracking.train_model(epochs=args.pre_tracking_epochs,
                                         steps_per_epoch=args.steps_per_epoch,
                                         weights_save_path=save_folder + "pre_tracking.h5",
                                         weights_load_path=None,
                                         do_checkpoint=args.do_checkpoint,
                                         generator=IdentificationNetwork.generator(
                                             folder_images=datasets_folder + "PreTracking/",
                                             batch_size=args.batch_size,
                                             image_ext=(".jpg", ".png")
                                         ))

    ####################################################################################################################
    # Tracking network
    ####################################################################################################################
    if args.tracking_epochs > 0:
        K.clear_session()

        net_tracking = TrackingNetwork(src.nets.generate_tracking_model(input_shape=(416, 416, 1),
                                                                        output_shape=(13, 13, 11),
                                                                        number_of_classes=6))

        load_path_tracking = load_folder + "tracking.h5" if load_folder is not None else save_folder + "pre_tracking.h5"

        net_tracking.get_model().summary()

        net_tracking.train_model(epochs=args.tracking_epochs,
                                 steps_per_epoch=args.steps_per_epoch,
                                 weights_save_path=save_folder + "tracking.h5",
                                 weights_load_path=load_path_tracking,
                                 do_checkpoint=args.do_checkpoint,
                                 generator=TrackingNetwork.generator(
                                     folder_images=datasets_folder + "Tracking/",
                                     folder_grids=datasets_folder + "Tracking/",
                                     batch_size=args.batch_size
                                 ))

    ####################################################################################################################
    # Identification network
    ####################################################################################################################
    if args.identification_epochs > 0:
        K.clear_session()

        net_identification = IdentificationNetwork(models=(
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=9),
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=4),
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2),
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2),
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=4),
            src.nets.generate_identification_model(input_shape=(50, 50, 1), number_of_classes=2)
        ))

        load_path_identification = load_folder + "identification.h5" if load_folder is not None else None

        net_identification.train_models(number_of_models=6,
                                        epochs=args.identification_epochs,
                                        steps_per_epoch=args.steps_per_epoch,
                                        weights_save_path=save_folder + "identification.h5",
                                        weights_load_path=None,
                                        do_checkpoint=args.do_checkpoint,
                                        generators=IdentificationNetwork.generators(
                                            folder_images=datasets_folder + "Identification/",
                                            batch_size=args.batch_size,
                                            image_ext=(".jpg", ".png")
                                        ))
