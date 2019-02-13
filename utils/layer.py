from keras.layers import Lambda, Dropout


def named(name):
    return Lambda(lambda x: x, name=name)


class DropoutPermanent(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, False)
