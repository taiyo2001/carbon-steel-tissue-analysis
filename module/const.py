class _const(object):
    """
    Class for creating constants that cannot be rebound.

    Raises:
    - ConstError: If attempting to rebind a constant.

    Usage:
    Constants can be defined as class attributes.
    Example:
    _const.PI = 3.14
    """
    class ConstError(TypeError):
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return f"Cannot rebind constant '{self.name}'"

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError(name)
        self.__dict__[name] = value

import sys

sys.modules[__name__] = _const()
