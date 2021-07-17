import numpy as np
import math
import tensorflow as tf
import random


class InverseGamma:
    def __init__(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.a1 = self.beta ** self.alpha
        self.a2 = math.gamma(self.alpha) + 1e-12

    def eval(self, x):
        t1 = tf.pow(x, -self.alpha-1)
        t2 = tf.exp(-self.beta / x)
        res = (self.a1 / self.a2) * t1 * t2
        energy = tf.reduce_sum(- tf.reduce_sum(tf.log(res+1e-12), 1))
        return res, energy

    def eval_np(self, input_x):
        input_x = input_x
        t1 = np.power(input_x, -self.alpha-1)
        t2 = np.exp(-self.beta/x)
        print(t1,t2)
        res = (self.a1 / self.a2) * t1 * t2
        return res


class GaussianMixtureModeling:
    def __init__(self, gmm_config):
        self.num_mixture = gmm_config[0]
        self.num_dim = gmm_config[1]
        self.num_dynamic_dim = gmm_config[2]
        # self.inverse_gamma = InverseGamma(1.0, 2e-3)
        # self.mixture_cov_0 = tf.constant(np.ones([self.num_mixture, 1])*0.1, dtype=tf.float32)

    def eval(self, x, p):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Cluster distribution phi: [num_mixture]
        phi = tf.reduce_mean(p, 0)
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])
        # mixture mean: [num_mixture, num_dim]
        p_t = tf.reshape(p, shape=[-1, self.num_mixture, 1])
        z_p = tf.reduce_sum(p_t, 0)
        mixture_mean = tf.reduce_sum(x_t * p_t, 0) / z_p
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        # mixture_cov_1 = tf.slice(tf.reduce_sum(z_t * p_t, 0), [0, 1], [2, 1])
        # print mixture_cov_1
        # mixture_cov_1 = mixture_cov_1 / z_p
        # print mixture_cov_1, z_p
        # mixture_cov = tf.concat([self.mixture_cov_0, mixture_cov_1], 1)
        # mixture_cov = tf.reduce_sum(z_t * p_t, 0) / z_p
        mixture_cov = np.ones([self.num_mixture, 1]) * 0.001
        mixture_dev = mixture_cov ** 0.5
        # probability density evaluation
        z_norm = tf.reduce_sum(z_t / mixture_cov, 2)
        mixture_dev_det = tf.reduce_prod(mixture_dev, 1)
        t1 = tf.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # Likelihood
        # prior_prob, _ = self.inverse_gamma.eval(mixture_dev)
        tmp = phi * (t1 / t2)
        likelihood = tf.reduce_sum(tmp, 1)
        energy = tf.reduce_mean(-tf.log(likelihood))
        pen_dev = tf.reduce_sum(1.0 / mixture_cov[:, 0:self.num_dynamic_dim])
        # pen_dev = tf.reduce_sum(1.0 / mixture_cov)
        # pen_dev = tf.reduce_sum(tf.reduce_sum(- tf.log(mixture_cov), 1) * phi)
        return energy, pen_dev, likelihood, phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        # probability density evaluation
        z_norm = tf.reduce_sum(z_t / mixture_cov, 2)
        mixture_dev_det = tf.reduce_prod(mixture_dev, 1)
        t1 = tf.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # Likelihood
        tmp = phi * t1 / t2
        likelihood = tf.reduce_sum(tmp, 1)
        return likelihood

    def em_evaluate(self, x_t, phi, mixture_mean, mixture_dev, mixture_cov):
        """
        :param x: input samples [batch_size, num_mixture, num_dim]
        :param phi: [num_mixture]
        :param mixture_mean: [num_mixture, num_dim]
        :param mixture_dev: [num_mixture, num_dim]
        :param mixture_cov: [num_mixture, num_dim]
        :return: [batch_size], [batch_size, num_mixture]
        """
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        # probability density evaluation
        z_norm = np.sum(z_t / mixture_cov, 2)
        mixture_dev_det = np.prod(mixture_dev, 1)
        t1 = np.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # Likelihood
        tmp = phi * t1 / t2
        likelihood = np.sum(tmp, 1)
        return likelihood, tmp

    def em_learning(self, x, tolerance):
        """
        :param x: input samples
        :param tolerance: convergence threshold
        :return: phi, mixture_mean, mixture_dev, mixture_cov
        """
        num_sample = np.size(x, axis=0)
        # Initialization
        prev_phi = np.ones(shape=[self.num_mixture]) * 1.0/self.num_mixture

        prev_mean = np.zeros(shape=[self.num_mixture, self.num_dim])
        idx = range(0, num_sample)
        random.shuffle(idx)
        for i in range(0, self.num_mixture):
            j = idx[i]
            prev_mean[i, :] = x[j, :]

        prev_cov = np.zeros(shape=[self.num_mixture, self.num_dim])
        global_mean = np.mean(x, axis=0)
        global_var = np.mean((x - global_mean) ** 2, axis=0)
        for i in range(0, self.num_mixture):
            prev_cov[i, :] = global_var

        prev_dev = prev_cov ** 0.5

        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = np.reshape(x, [-1, 1, self.num_dim])
        x_t = np.tile(x_t, (1, self.num_mixture, 1))

        diff = 1e6

        phi = prev_phi
        mixture_mean = prev_mean
        mixture_dev = prev_dev
        mixture_cov = prev_cov

        cnt = 0

        while diff >= tolerance:
            # E-step
            likelihood, mixture_likelihood = self.em_evaluate(x_t, prev_phi, prev_mean, prev_dev, prev_cov)
            likelihood = np.reshape(likelihood, [-1, 1])
            gamma = mixture_likelihood / likelihood
            # M-step
            phi = np.mean(gamma, 0)
            # mixture mean: [num_mixture, num_dim]
            p_t = np.reshape(gamma, [-1, self.num_mixture, 1])
            z_p = np.sum(p_t, 0)
            mixture_mean = np.sum(x_t * p_t, 0) / z_p

            # mixture diagonal covariance: [num_mixture, num_dim]
            z_t = (x_t - mixture_mean) ** 2
            mixture_cov = np.sum(z_t * p_t, 0) / z_p
            mixture_dev = mixture_cov ** 0.5

            diff = 0
            diff = max(diff, np.amax(np.abs(prev_phi - phi)))
            diff = max(diff, np.amax(np.abs(prev_mean - mixture_mean)))
            diff = max(diff, np.amax(np.abs(prev_dev - mixture_dev)))

            prev_phi = phi
            prev_mean = mixture_mean
            prev_cov = mixture_cov
            prev_dev = mixture_dev

            cnt = cnt + 1
            if cnt % 10 == 0:
                print('Diff: ' + str(diff))
        return phi, mixture_mean, mixture_dev, mixture_cov

    def vi_learning(self, x, p):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Cluster distribution phi: [num_mixture]
        phi = tf.reduce_mean(p, 0)

        # Augmenting input: [batch_size, num_mixture, num_dim]
        #x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim]) # self.num_dim, self.num_mixture, or XXX
        #x_t = tf.reshape(x, shape=[-1, 1, 40])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])

        #print'x_t'
        #print(x_t)

        # mixture mean: [num_mixture, num_dim]
        p_t = tf.reshape(p, shape=[-1, self.num_mixture, 1])
        z_p = tf.reduce_sum(p_t, 0)
        
        '''
        print'p_t'
        print(p_t)
        print'z_p'
        print(z_p)
        '''
    
        mixture_mean = tf.reduce_sum(x_t * p_t, 0) / z_p
        
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2  #[batch_size, num_mixture, num_dim]
        mixture_cov = tf.reduce_sum(z_t * p_t, 0) / z_p #[num_mixture, num_dim]
        #mixture_cov = tf.ones([self.num_mixture, self.num_dim], tf.float64) * 1
        mixture_dev = mixture_cov ** 0.5
        
        # probability density evaluation
        z_norm = tf.reduce_sum(z_t / mixture_cov, 2)
        mixture_dev_det = tf.reduce_prod(mixture_dev, 1) #[num_mixture]
        t1 = tf.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        
        # Likelihood
        tmp = phi * (t1 / t2)
        likelihood = tf.reduce_sum(tmp, 1)
        pen_dev = tf.reduce_sum(1.0 / mixture_cov)
        posterior_z = tf.reduce_sum(tmp, 1, keep_dims=True)
        posterior = tmp / posterior_z
        kl_divergence = tf.reduce_sum(p * tf.log(p / (posterior + 1e-12)), 1)
        energy = tf.reduce_mean(-tf.log(likelihood) + kl_divergence)
        #energy = tf.reduce_mean(-tf.log(likelihood))
        return energy, posterior, pen_dev, likelihood, phi, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det


class SoftKMeansMixtureModeling:
    def __init__(self, gmm_config):
        self.num_mixture = gmm_config[0]
        self.num_dim = gmm_config[1]
        self.mixture_cov_0 = tf.constant(np.ones([self.num_mixture, 1])*0.1, dtype=tf.float32)

    def eval(self, x, p):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Cluster distribution phi: [num_mixture]
        phi = tf.reduce_mean(p, 0)
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])
        # mixture mean: [num_mixture, num_dim]
        p_t = tf.reshape(p, shape=[-1, self.num_mixture, 1])
        z_p = tf.reduce_sum(p_t, 0)
        mixture_mean = tf.reduce_sum(x_t * p_t, 0) / z_p
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        # mixture_cov = tf.reduce_sum(z_t * p_t, 0) / z_p
        # mixture_cov = tf.maximum(mixture_cov, 1e-2)
        # mixture_dev = mixture_cov ** 0.5
        # probability density evaluation
        # z_norm = tf.reduce_sum(z_t / (mixture_cov + 1e-12), 2)
        # mixture_dev_det = tf.reduce_prod(mixture_dev, 1)
        # t1 = tf.exp(-0.5 * z_norm)
        # t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # # Likelihood
        # energy = tf.reduce_mean(-tf.log(tf.reduce_sum(phi * t1 / t2, 1)))
        distance = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(z_t, 2) * phi, 1))
        # pen_dev = tf.reduce_sum(0.05 / mixture_cov)
        return distance


if __name__ == '__main__':
    inverse_gamma = InverseGamma(2, 0.3)
    # x = np.array([0.0036749, 0.19699372, 0.00374158, 0.20074838])
    x = np.array([1e-6])
    print - np.log(inverse_gamma.eval_np(x))
