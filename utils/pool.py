import numpy as np


class FakeImagePool:
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        output = []
        for image in images:
            if len(self.images) < self.pool_size:
                output.append(image)
                self.images.append(image)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    index = np.random.randint(0, len(self.images))
                    _image = self.images[index]
                    self.images[index] = image
                    output.append(_image)
                else:
                    output.append(image)
        return np.array(output)
