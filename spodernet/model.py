class Model(object):

    def __init__(self, input_module=None):
        self.modules = []
        self.input_module = input_module
        pass


    def add(self, module):
        self.modules.append(module)

    def forward(self):
        output = []
        for module in self.modules:
            output = module.forward(*output)
        return output
