class Model(object):

    def __init__(self, input_module=None):
        self.modules = []
        self.input_module = input_module
        pass


    def add(self, module):
        self.modules.append(module)

    def forward(self, feed_dict=None, *inputs):
        outputs = inputs
        if inputs == None:
            outputs = []
        for module in self.modules:
            outputs = module.forward(feed_dict, *outputs)
        return outputs
