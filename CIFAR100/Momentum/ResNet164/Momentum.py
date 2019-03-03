from __future__ import division
from __future__ import print_function
from keras.legacy import interfaces

from keras import backend as K
from six.moves import zip
import keras

class Momentum(keras.optimizers.Optimizer):
    """SGD with momentum optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta: float, 0 < beta < 1. Generally close to 1.
        decay: float >= 0. Learning rate decay over each update.
    """

    def __init__(self, lr=0.001, beta=0.9, decay=0., **kwargs):
        super(Momentum, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta = K.variable(beta, name='beta')
            self.decay = K.variable(decay, name='decay')

        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms

        for p, g, m in zip(params, grads, ms):
            m_t = (self.beta * m) + (1. - self.beta) * g
 
            p_t = p - lr * m_t

            self.updates.append(K.update(m, m_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta': float(K.get_value(self.beta)),
                  'decay': float(K.get_value(self.decay)),
                 }
        base_config = super(Momentum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

