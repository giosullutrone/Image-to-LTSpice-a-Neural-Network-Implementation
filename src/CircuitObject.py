class CircuitObject:
    #TODO: Enum instead of simple integers

    NOTHING = 23
    SOMETHING = 24
    NUMBER_OF_CLASSES = 6

    def __init__(self, box_class):
        self.__box_class = box_class

    def get_box_class(self):
        return self.__box_class

    @staticmethod
    def is_an_object(box_class):
        return box_class != CircuitObject.NOTHING

    @staticmethod
    def is_an_unknown_object(box_class):
        return box_class == CircuitObject.SOMETHING

    def get_type(self):
        """Returns the type of object associated with the box class"""
        if self.__box_class in range(0, 9):
            return 0
        elif self.__box_class in range(9, 13):
            return 1
        elif self.__box_class in range(13, 15):
            return 2
        elif self.__box_class in range(15, 17):
            return 3
        elif self.__box_class in range(17, 21):
            return 4
        elif self.__box_class in range(21, 23):
            return 5
        elif self.__box_class == CircuitObject.SOMETHING:
            return 7
        else:
            return 6

    @staticmethod
    def from_type(box_type):
        """Returns the type of object associated with the box class"""
        if box_type == 0:
            return CircuitObject(0)
        elif box_type == 1:
            return CircuitObject(9)
        elif box_type == 2:
            return CircuitObject(13)
        elif box_type == 3:
            return CircuitObject(15)
        elif box_type == 4:
            return CircuitObject(17)
        elif box_type == 5:
            return CircuitObject(21)
        elif box_type == 7:
            return CircuitObject(CircuitObject.SOMETHING)
        else:
            return CircuitObject(CircuitObject.NOTHING)

    @staticmethod
    def get_type_from_box_class(box_class):
        """Returns the type of the object assoiciated with the box class number"""
        return CircuitObject(box_class).get_type()

    @staticmethod
    def get_box_class_from_type(box_type):
        """Returns the type of the object assoiciated with the box class number"""
        return CircuitObject.from_type(box_type).get_box_class()

    def get_name(self):
        """Returns the name of the object associated with the box class number"""
        if self.__box_class in range(0, 9):
            return "Corner"
        elif self.__box_class in range(9, 13):
            return "Voltage generator"
        elif self.__box_class in range(13, 15):
            return "Resistance"
        elif self.__box_class in range(15, 17):
            return "Inductor"
        elif self.__box_class in range(17, 21):
            return "Current generator"
        elif self.__box_class in range(21, 23):
            return "Capacitor"
        elif self.__box_class == CircuitObject.SOMETHING:
            return "Something"
        else:
            return "Nothing"

    @staticmethod
    def get_name_from_box_class(box_class):
        """Returns the name of the object assoiciated with the box class number"""
        return CircuitObject(box_class).get_name()

    @staticmethod
    def __opposite(value, couple):
        """Returns the other value from a couple"""
        if value == couple[0]:
            return couple[1]
        return couple[0]

    @staticmethod
    def __next(value, couple, step):
        """Returns the next value (by a step) from a couple"""

        import numpy as np

        #Create a np array from couple
        couple_np = np.array(couple)

        #Get the index of the value
        value_index = np.where(couple_np==value)[0]

        return couple[int((value_index + step) % len(couple))]

    def rotate(self, rad):
        """Rotates the object by rad, changing, if needed, its box class"""

        import math

        if rad < 0:
            rad = (2*math.pi) + rad

        #Get the step to use to get the correct box class after the rotation
        step = 0
        if (rad > math.pi * (1/4) and rad < math.pi * (3/4)):
            step = 1
        elif (rad > math.pi * (3/4) and rad < math.pi * (5/4)):
            step = 2
        elif (rad > math.pi * (5/4) and rad < math.pi * (7 / 4)):
            step = 3

        couples = ((0, 1, 2, 3),
                   (4, 7, 5, 6),
                   (9, 10, 11, 12),
                   (13, 14),
                   (15, 16),
                   (17, 18, 19, 20),
                   (21, 22))
        self.__rotate(couples, step)

    def __rotate(self, couples, step):
        """Sets the box class to the next value of the couple (by a step)"""

        box_class = self.__box_class
        for couple in couples:
            if box_class in couple:
                self.__box_class = CircuitObject.__next(box_class, couple, step)
                break

    def flip_horizontally(self):
        """Flips the object horizontally, changing, if needed, its box class"""

        #Object that in case of a flip change their values
        couples = ((0, 1),
                   (2, 3),
                   (4, 5),
                   (10, 12),
                   (18, 20))
        self.__flip(couples)

    def flip_vertically(self):
        """Flips the object vertically, changing, if needed, its box class"""

        #Object that in case of a flip change their values
        couples = ((0, 3),
                   (1, 2),
                   (6, 7),
                   (9, 11),
                   (17, 19))
        self.__flip(couples)

    def __flip(self, couples):
        """Sets the box class to the other value of the couple"""
        box_class = self.__box_class
        for couple in couples:
            if box_class in couple:
                self.__box_class = CircuitObject.__opposite(box_class, couple)
                break
