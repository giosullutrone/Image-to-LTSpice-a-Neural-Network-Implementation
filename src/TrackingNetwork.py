class TrackingNetwork:
    def __init__(self, model=None):
        self.__model = model

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def train_model(self, epochs, steps_per_epoch, generator, weights_save_path, weights_load_path=None, do_checkpoint=True, optimizer=None, loss_function=None):
        from tensorflow.keras.callbacks import ModelCheckpoint

        loss_function = TrackingNetwork.loss_function if loss_function is None else loss_function
        optimizer = TrackingNetwork.optimizer() if optimizer is None else optimizer

        self.__model.compile(optimizer, loss=loss_function, metrics=[TrackingNetwork.loss_function_class_accuracy,
                                                                     TrackingNetwork.loss_function_class,
                                                                     TrackingNetwork.loss_function_position,
                                                                     TrackingNetwork.loss_function_size,
                                                                     TrackingNetwork.loss_function_class_type])

        if weights_load_path is not None:
            self.load_weights(weights_load_path)

        if do_checkpoint:
            checkpoint = [ModelCheckpoint(weights_save_path, monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)]
        else:
            checkpoint = None

        self.__model.fit_generator(generator=generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   callbacks=checkpoint,
                                   verbose=1)

        self.__model.save_weights(filepath=weights_save_path)

    def load_weights(self, weights_load_path):
        try:
            self.__model.load_weights(weights_load_path, by_name=False)
        except Exception as e:
            self.__model.load_weights(weights_load_path, by_name=True)

    def predict_boxes_from_file(self, image_path, confidence=0.5):
        from src.AugmentedImage import AugmentedImage
        from src.GridBoxesUtil import GridBoxesUtil
        import numpy as np

        image = AugmentedImage.image_from_file(image_path, grayscale=True) / 255.0
        batch_x = np.expand_dims(image, axis=0)

        if len(batch_x.shape) == 3:
            batch_x = np.expand_dims(batch_x, axis=-1)

        grid = self.__model.predict(batch_x, batch_size=None)
        subdivisions = grid.shape[1]

        return GridBoxesUtil.to_boxes(grid[0],
                                      AugmentedImage.get_image_width(image),
                                      AugmentedImage.get_image_height(image),
                                      subdivisions,
                                      confidence)

    @staticmethod
    def optimizer(lr=0.001, amsgrad=True):
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=lr, amsgrad=amsgrad, decay=0.1, clipvalue=10.0)

    @staticmethod
    def loss_function(y_true, y_pred):
        return (TrackingNetwork.loss_function_class(y_true, y_pred) +
                TrackingNetwork.loss_function_position(y_true, y_pred) +
                TrackingNetwork.loss_function_size(y_true, y_pred) +
                TrackingNetwork.loss_function_class_type(y_true, y_pred))

    @staticmethod
    def loss_function_class(y_true, y_pred):
        import tensorflow.keras.backend as K

        y_true_class = y_true[..., 0]
        y_pred_class = y_pred[..., 0]

        y_true_class_zeros = K.sum(K.ones_like(y_true_class) - y_true_class)
        y_true_class_ones = K.sum(y_true_class)

        return (K.sum((K.ones_like(y_true_class) - y_true_class) * K.square(y_true_class - y_pred_class)) / y_true_class_zeros +
                K.sum(y_true_class * K.square(y_true_class - y_pred_class)) / y_true_class_ones)

    @staticmethod
    def loss_function_class_type(y_true, y_pred):
        import tensorflow.keras.backend as K

        y_true_class = y_true[..., 0]

        y_true_class_type = y_true[..., 5:]
        y_pred_class_type = y_pred[..., 5:]

        return K.mean(y_true_class * K.sum(K.square(y_true_class_type - y_pred_class_type), axis=-1))

    @staticmethod
    def loss_function_position(y_true, y_pred):
        import tensorflow.keras.backend as K

        y_true_class = y_true[..., 0]

        y_true_position = y_true[..., 1:3]
        y_pred_position = y_pred[..., 1:3]

        return K.max(K.mean(K.abs(y_true_position - y_pred_position), axis=-1) * y_true_class)

    @staticmethod
    def loss_function_size(y_true, y_pred):
        import tensorflow.keras.backend as K

        y_true_class = y_true[..., 0]

        y_true_size = y_true[..., 3:5]
        y_pred_size = y_pred[..., 3:5]

        return K.sum(K.sum(K.square(y_true_size - y_pred_size), axis=-1) * y_true_class) / (K.sum(y_true_class) + K.epsilon())

    @staticmethod
    def loss_function_class_accuracy(y_true, y_pred):
        import tensorflow.keras.backend as K

        y_true_class = K.round(y_true[..., 0])
        y_pred_class = K.round(y_pred[..., 0])

        return K.mean(K.equal(y_true_class, y_pred_class))

    @staticmethod
    def generator(folder_images, folder_grids, batch_size=64, image_exts=(".png", ".jpg"), grid_ext=".txt"):
        from src.GridBoxesUtil import GridBoxesUtil
        import numpy as np
        import random
        import cv2

        images_and_grids_file_names = GridBoxesUtil.get_images_and_grids_file_names_from_folder(folder_images,
                                                                                                folder_grids,
                                                                                                image_exts=image_exts,
                                                                                                grid_ext=grid_ext)

        while True:
            batch_files = random.choices(images_and_grids_file_names, k=batch_size)
            batch_input = []
            batch_output = []

            for batch_file in batch_files:
                image_file, grid_file = batch_file

                inp = np.expand_dims(cv2.imread(folder_images + image_file, 0), axis=-1)
                output = GridBoxesUtil.to_grid_from_file(folder_grids + grid_file)

                if np.isnan(np.sum(inp)) or np.isnan(np.sum(output)):
                    print("Found an NaN in input and/or output, skipping the file...")
                    continue

                inp = (inp / 255.0)
                batch_input += [inp]
                batch_output += [output]

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
