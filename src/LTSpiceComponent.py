class LTSpiceComponent:
    def __init__(self, component_class, component_x, component_y):
        self.__component_class = component_class
        self.__component_x = component_x
        self.__component_y = component_y

        self.__connections = self.get_connections()

        self.__top = None
        self.__bottom = None
        self.__left = None
        self.__right = None

    def is_corner(self):
        from src.CircuitObject import CircuitObject
        return CircuitObject.get_type_from_box_class(self.__component_class) == 0

    def get_connections(self):
        from src.CircuitObject import CircuitObject

        if CircuitObject.get_type_from_box_class(self.__component_class) != 0:
            return None

        if self.__component_class == 0:
            return "t", "r"
        elif self.__component_class == 1:
            return "t", "l"
        elif self.__component_class == 2:
            return "b", "l"
        elif self.__component_class == 3:
            return "b", "r"
        elif self.__component_class == 4:
            return "t", "b", "r"
        elif self.__component_class == 5:
            return "t", "b", "l"
        elif self.__component_class == 6:
            return "b", "r", "l"
        elif self.__component_class == 7:
            return "t", "r", "l"
        elif self.__component_class == 8:
            return "t", "b", "r", "l"

    def get_ltspice_component(self, name=None):
        from src.CircuitObject import CircuitObject

        assert CircuitObject.get_type_from_box_class(self.__component_class) != 0, "Type of circuit object is 0..."

        def get_string(symbol, rotation, prefix, delx, dely):
            return (f"SYMBOL {symbol} {self.__component_x * 80 + 48 + delx} {self.__component_y * 80 + 48 + dely} {rotation}" +
                    ("" if name is None else f"\nSYMATTR InstName {prefix}{name}"))

        if self.__component_class == 11:
            delx = 0
            dely = -16
            return get_string("bv", "R0", "V", delx, dely)
        elif self.__component_class == 10:
            delx = 16
            dely = 0
            return get_string("bv", "R90", "V", delx, dely)
        elif self.__component_class == 9:
            delx = 0
            dely = 16
            return get_string("bv", "R180", "V", delx, dely)
        elif self.__component_class == 12:
            delx = -16
            dely = 0
            return get_string("bv", "R270", "V", delx, dely)

        if self.__component_class == 13:
            delx = -16
            dely = -16
            return get_string("res", "R0", "R", delx, dely)
        elif self.__component_class == 14:
            delx = -16
            dely = 16
            return get_string("res", "R270", "R", delx, dely)

        if self.__component_class == 15:
            delx = -16
            dely = -16
            return get_string("ind", "R0", "L", delx, dely)
        elif self.__component_class == 16:
            delx = -16
            dely = 16
            return get_string("ind", "R270", "L", delx, dely)

        if self.__component_class == 19:
            delx = 0
            dely = 0
            return get_string("bi2", "R0", "I", delx, dely)
        elif self.__component_class == 18:
            delx = 0
            dely = 0
            return get_string("bi2", "R90", "I", delx, dely)
        elif self.__component_class == 17:
            delx = 0
            dely = 0
            return get_string("bi2", "R180", "I", delx, dely)
        elif self.__component_class == 20:
            delx = 0
            dely = 0
            return get_string("bi2", "R270", "I", delx, dely)

        if self.__component_class == 21:
            delx = -16
            dely = 0
            return get_string("cap", "R0", "C", delx, dely)
        elif self.__component_class == 22:
            delx = 0
            dely = 16
            return get_string("cap", "R270", "C", delx, dely)

    def get_component_class(self):
        return self.__component_class

    def get_component_x(self):
        return self.__component_x

    def get_component_y(self):
        return self.__component_y

    def has_top(self):
        if self.__connections is None:
            return False
        return "t" in self.__connections

    def has_bottom(self):
        if self.__connections is None:
            return False
        return "b" in self.__connections

    def has_left(self):
        if self.__connections is None:
            return False
        return "l" in self.__connections

    def has_right(self):
        if self.__connections is None:
            return False
        return "r" in self.__connections

    def can_top(self):
        return self.has_top() and self.__top is None

    def can_bottom(self):
        return self.has_bottom() and self.__bottom is None

    def can_left(self):
        return self.has_left() and self.__left is None

    def can_right(self):
        return self.has_right() and self.__right is None

    def get_top(self):
        return self.__top

    def get_bottom(self):
        return self.__bottom

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def set_top(self, top):
        assert self.has_top()
        return self.__top

    def set_bottom(self, bottom):
        assert self.has_bottom()
        return self.__bottom

    def set_left(self, left):
        assert self.has_left()
        return self.__left

    def set_right(self, right):
        assert self.has_right()
        return self.__right
