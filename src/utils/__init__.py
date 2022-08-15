class AttributeDict(dict):
    """ Nested Attribute Dictionary
    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
    """
    def __init__(self, mapping=None):
        super(AttributeDict, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttributeDict(value)
        elif isinstance(value, list):
            # if the value is list, try to go recursively only on the dicts from that list
            value_list = list()
            for v in value:
                if isinstance(v, dict):
                    value_list.append(AttributeDict(v))
                else:
                    value_list.append(v)
            value = value_list
        super(AttributeDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__
    
    def get(self, __key, __default):
        rez_VT = super().get(__key, __default)
        rez_VT = rez_VT if rez_VT is not None else __default
        return rez_VT

