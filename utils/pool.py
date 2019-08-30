from typing import List

import numpy as np


class FakeImagePool:
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size: int = pool_size
        self.image_pool: List[np.array] = []

    def query(self, images: np.array) -> np.array:
        if self.pool_size == 0:
            return images

        output = []
        for image in images:
            if len(self.image_pool) < self.pool_size:
                output.append(image)
                self.image_pool.append(image)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    index = np.random.randint(0, len(self.image_pool))
                    _image = self.image_pool[index]
                    self.image_pool[index] = image
                    output.append(_image)
                else:
                    output.append(image)
        return np.array(output)
