



class ttclass(object):
    def __init__(self):
        super(ttclass, self).__init__()

    @property
    def module_str(self):
        raise NotImplementedError

class ttclass_son(ttclass):
    def __init__(self):
        super(ttclass_son, self).__init__()

        self.layer = 1

    def module_str(self):
        return self.layer



ins = ttclass_son()

print(ins.module_str)


print('{}'.format(None))