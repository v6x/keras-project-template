from typing import Generator, Optional

from tensorpack import DataFlow


class GeneratorToDataFlow(DataFlow):
    def __init__(self, generator: Generator, size: Optional[int] = None):
        super().__init__()
        self.generator: Generator = generator
        self.generator_size: Optional[int] = size

    def __iter__(self):
        for dp in self.generator:
            yield dp

    def __len__(self):
        if self.generator_size is None:
            raise NotImplementedError
        return self.generator_size
