class _const(object):
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
