import math

import tensorflow as tf

import core.general.param_init as pini


class PretrainAutoencoder:
    def __init__(self, config, num_drop_out):
        self.n_input = config[0]
        self.n_hidden = 40

        self.rho = 0.01
        self.alpha = 0.0001
        self.beta = 3
        self.activation = tf.nn.sigmoid

        # Encode
        self.W1 = self.init_weights((self.n_input, 500))
        self.b1 = self.init_weights((1, 500))

        self.W2 = self.init_weights((500, self.n_hidden))
        self.b2 = self.init_weights((1, self.n_hidden))

        # Decode
        self.W3 = self.init_weights((self.n_hidden, 500))
        self.b3 = self.init_weights((1, 500))

        self.W4 = self.init_weights((500, self.n_input))
        self.b4 = self.init_weights((1, self.n_input))

    def init_weights(self, shape):
        r = math.sqrt(6) / math.sqrt(self.n_input + self.n_hidden + 1)
        weights = tf.random_normal(shape, stddev=r)
        return tf.Variable(weights)

    def encode(self, X, W, b):
        l = tf.matmul(X, W) + b
        return self.activation(l)

    def decode(self, H, W, b):
        l = tf.matmul(H, W) + b
        return self.activation(l)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def regularization(self, weights):
        return tf.nn.l2_loss(weights)

    def run(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        var_list = []
        reg = []
        l2_reg = 0

        H1 = self.encode(x, self.W1, self.b1)
        H2 = self.encode(H1, self.W2, self.b2)

        rho_hat = tf.reduce_mean(H2, axis=0)
        kl = self.kl_divergence(self.rho, rho_hat)
        x_1 = self.decode(H2, self.W3, self.b3)
        x_2 = self.decode(x_1, self.W4, self.b4)
        diff = x - x_2


        cost1 = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) \
               + 0.5 * self.alpha * tf.nn.l2_loss(self.W1)  \
               + self.beta * tf.reduce_sum(kl)

        cost2 = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) \
               + 0.5 * self.alpha * tf.nn.l2_loss(self.W1) + 0.5 * self.alpha * tf.nn.l2_loss(self.W2)  \
               + self.beta * tf.reduce_sum(kl)

        cost3 = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) \
               + 0.5 * self.alpha * tf.nn.l2_loss(self.W1) +  0.5 * self.alpha * tf.nn.l2_loss(self.W2) \
               + tf.nn.l2_loss(self.W3) + tf.nn.l2_loss(self.W4) \
               + self.beta * tf.reduce_sum(kl)

        # encode
        reg1 = self.regularization(self.W1)
        reg2 = self.regularization(self.W2)
        reg3 = self.regularization(self.W3)
        reg4 = self.regularization(self.W4)

        #decode
        l2_reg1 = reg1 + reg4
        l2_reg2 = reg2 + reg3

        return H2, [cost1, cost2, cost3], [[self.W1, self.b1], [self.W2, self.b2]], l2_reg1 + l2_reg2, [l2_reg1, l2_reg2]

    def test(self, x):
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj

        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 20), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        # relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        # xo = tf.concat([zc, relative_dist, cos_sim], 1)
        # xo = tf.concat([zc, relative_dist], 1)
        return dist

    def dcn_run(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        var_list = []
        reg = []
        l2_reg = 0
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            if i < self.num_dropout_layer:
                zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            ni = len(self.wi) - i - 1
            if i == 0:
                z_r = tf.matmul(zj, self.wi[len(self.wi) - 1]) + self.bi[len(self.bi) - 1]
            else:
                z_r = tf.nn.tanh(tf.matmul(zj, self.wi[ni]) + self.bi[ni])
            error_l = tf.reduce_mean(tf.norm(zi - z_r, ord=2, axis=1, keep_dims=True))
            error.append(error_l * vision_coef)
            zi = zj
            reg.append(tf.nn.l2_loss(self.wi[i]) + tf.nn.l2_loss(self.wi[ni]))
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            var_list.append([self.wi[i], self.bi[i], self.wi[ni], self.bi[ni]])
        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 2), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
                if i >= len(self.wi) - 1 - self.num_dropout_layer:
                    zj = tf.nn.dropout(zj, keep_prob=keep_prob)
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            l2_reg = l2_reg + tf.nn.l2_loss(self.wi[i])
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        loss = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # self.dist_min = tf.reduce_min(dist)
        # self.dist_max = tf.reduce_max(dist)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        xo = zc
        # xo = tf.concat([zc, relative_dist], 1)
        error_all = tf.reduce_mean(loss)
        error.append(error_all)
        # var_list.append([self.w6, self.b6, self.w9, self.b9])
        # error = error_all + error_1 + error_2 + error_3 + error_4 + error_5 + error_6 + error_7
        return xo, error, var_list, l2_reg, reg

