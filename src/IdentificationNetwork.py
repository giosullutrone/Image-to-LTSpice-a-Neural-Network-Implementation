class IdentificationNetwork:
    def __init__(self, model=None, models=None):
        self.__model = model
        self.__models = models

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def get_models(self):
        return self.__models

    def set_models(self, models):
        self.__models = models

    def train_models(self, number_of_models, epochs, steps_per_epoch, generators, weights_save_path,
                     weights_load_path=None, do_checkpoint=True, optimizer=None, loss_function=None):
        assert self.__models is not None, "The list of \"models\" given is None..."
        assert len(self.__models) == number_of_models, "The number of models given is different from the actual models to use"

        old_model = self.__model
        for i in range(number_of_models):
            self.__model = self.__models[i]
            self.train_model(epochs, steps_per_epoch, generators[i], weights_save_path.replace(".", f"_{i}."),
                             weights_load_path.replace(".", f"_{i}.") if weights_load_path is not None else None,
                             do_checkpoint, optimizer, loss_function)
        self.__model = old_model

    def train_model(self, epochs, steps_per_epoch, generator, weights_save_path, weights_load_path=None, do_checkpoint=True, optimizer=None, loss_function=None):
        from tensorflow.keras.callbacks import ModelCheckpoint

        loss_function = IdentificationNetwork.loss_function if loss_function is None else loss_function
        optimizer = IdentificationNetwork.optimizer() if optimizer is None else optimizer

        self.__model.compile(optimizer, loss=loss_function, metrics=["acc"])

        if weights_load_path is not None:
            try:
                self.load_weights(weights_load_path)
            except Exception as e:
                print("Could not load the weights: " + weights_load_path + " not found, skipping...")

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
        self.__model.load_weights(weights_load_path, by_name=True)

    def load_all_weights(self, weights_load_path, number_of_models):
        for i in range(number_of_models):
            self.__models[i].load_weights(weights_load_path.replace(".", f"_{i}."), by_name=False)

    def predict_boxes_from_file_and_tracking_output(self, image_path, boxes, number_of_models, image_input_size=(50, 50)):
        import numpy as np
        import cv2
        from src.CircuitObject import CircuitObject
        from src.AugmentedImage import AugmentedImage
        from src.Boxes import Boxes
        from src.Box import Box

        image = AugmentedImage.image_from_file(image_path, grayscale=True) / 255.0

        boxes_new = Boxes()

        rectangles = boxes.get_topleft_boxes_data()
        for i in range(number_of_models):
            batch_x = []
            boxes_data = []

            for rect in rectangles:
                box_class, x, y, width, height = rect

                if CircuitObject.get_type_from_box_class(box_class) != i:
                    continue

                x = max(0, x)
                y = max(0, y)

                image_section = image[int(y): int(y + height), int(x): int(x + width)]

                # Resize to match input size
                image_section = np.expand_dims(cv2.resize(image_section, image_input_size), axis=-1)

                batch_x += [image_section / 255.0]
                boxes_data.append((x+width/2, y+height/2, width, height))

            if len(batch_x) == 0:
                continue

            batch_x = np.array(batch_x)
            batch_y = self.__models[i].predict(batch_x, batch_size=None)

            for j in range(len(boxes_data)):
                box_class_new = np.argmax(batch_y[j]) + CircuitObject.get_box_class_from_type(i)

                boxes_new.add_box(Box(int(box_class_new), boxes_data[j][0], boxes_data[j][1], boxes_data[j][2], boxes_data[j][3]))
        return boxes_new

    @staticmethod
    def optimizer(lr=0.001, amsgrad=True):
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=lr, amsgrad=amsgrad, decay=0.1, clipvalue=10.0)

    @staticmethod
    def loss_function(y_true, y_pred):
        from tensorflow.keras.losses import categorical_crossentropy
        return categorical_crossentropy(y_true, y_pred)

    @staticmethod
    def generators(folder_images, batch_size=64, image_ext=(".jpg", ".png")):
        """Get a list of generators, one for each type of object"""
        import os
        generators = []
        for i in range(len(os.listdir(folder_images))):
            generators.append(IdentificationNetwork.generator(folder_images + str(i) + "/", batch_size, image_ext))
        return generators

    @staticmethod
    def generator(folder_images, batch_size=64, image_ext=(".jpg", ".png")):
        import numpy as np
        import random
        import cv2
        import os

        image_files_and_type = []
        for folder in os.listdir(folder_images):
            image_files_of_folder = [x for x in os.listdir(folder_images + folder) if x.endswith(image_ext)]
            for image_file_of_folder in image_files_of_folder:
                image_files_and_type += [(folder + "/" + image_file_of_folder, int(folder))]

        while True:
            batch_files = random.choices(image_files_and_type, k=batch_size)
            batch_input = []
            batch_output = []

            for batch_file in batch_files:
                image_file, box_type = batch_file

                inp = np.expand_dims(cv2.imread(folder_images + image_file, 0), axis=-1)
                output = np.zeros(shape=(len(os.listdir(folder_images))))
                output[box_type] = 1.0

                if np.isnan(np.sum(inp)) or np.isnan(np.sum(output)):
                    print("Found an NaN in input and/or output, skipping the file...")
                    continue

                inp = (inp / 255.0)
                batch_input += [inp]
                batch_output += [output]

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
