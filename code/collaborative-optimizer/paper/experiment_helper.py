class Parser:
    def __init__(self):
        self.key_val = dict()

    def parse(self, arguments):
        for arg in arguments:
            split = arg.split('=')
            if len(split) != 2:
                raise Exception('Invalid argument: {}'.format(arg))
            self.key_val[split[0]] = split[1]

    def get(self, key):
        return self.key_val[key]
