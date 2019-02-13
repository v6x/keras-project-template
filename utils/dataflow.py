from tensorpack import DataFlow, ProxyDataFlow


class GeneratorToDataFlow(DataFlow):
    def __init__(self, generator, size):
        super().__init__()
        self.generator = generator
        self.generator_size = size

    def get_data(self):
        for dp in self.generator:
            yield dp

    def size(self):
        return self.generator_size


class InfiniteDataFlow(ProxyDataFlow):
    def get_data(self):
        while True:
            for dp in self.ds.get_data():
                yield dp


class LimitSizedDataFlow(ProxyDataFlow):
    def __init__(self, ds, limited_size):
        super().__init__(ds)
        self.limited_size = limited_size

    def get_data(self):
        i = 0
        for dp in self.ds.get_data():
            i += 1
            if i > self.limited_size:
                break
            yield dp

    def size(self):
        return self.limited_size
