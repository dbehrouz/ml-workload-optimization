class Parser:
    def __init__(self, args):
        self.key_val = dict()
        self.parse(args)

    def parse(self, arguments):
        for arg in arguments:
            if '=' in arg:
                split = arg.split('=')
                if len(split) != 2:
                    raise Exception('Invalid argument: {}'.format(arg))
                self.key_val[split[0]] = split[1]

    def get(self, key, default):
        return self.key_val.get(key, default)
