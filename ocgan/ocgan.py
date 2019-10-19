import numpy as np
from sklearn.metrics import roc_auc_score
from .models import *
from .utils import task, attrdict, d_of_l, assure_dtype_uint8, merge, gray2rgb, add_border


class OCGAN:
    def __init__(self, sess, steps, lr, BS, use_informative_mining=True):
        self.sess = sess
        latent_shape = [288]

        self.enc = get_encoder()
        self.dec = get_decoder()
        self.disc_v = get_disc_visual()
        self.disc_l = get_disc_latent(latent_shape)
        self.cl = get_classifier()
        self.BS = BS

        self.steps = steps
        self.use_informative_mining = use_informative_mining

        with task('Build Graph'):
            X = tf.placeholder(tf.float32, [BS, 28, 28, 1])
            z = tf.random.normal(tf.shape(X), stddev=1e-5)

            l2 = tf.random.uniform([BS] + latent_shape, minval=-1, maxval=1)
            Xz = X + z
            l1 = self.enc(Xz)
            self.recon = self.dec(self.enc(X))
            dec_l2 = self.dec(l2)
            self.gen = dec_l2

            with task('Loss'):
                loss_op = attrdict()
                with task('1. Classifier loss'):
                    logits_C_l1 = self.cl(self.dec(l1))
                    logits_C_l2 = self.cl(dec_l2)

                    loss_op.C_l1 = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_C_l1), logits=logits_C_l1)
                    loss_op.C_l2 = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_C_l2), logits=logits_C_l2)
                    loss_op.C = (loss_op.C_l1 + loss_op.C_l2) / 2

                with task('2. Discriminator latent loss'):
                    logits_Dl_l1 = self.disc_l(l1)
                    logits_Dl_l2 = self.disc_l(l2)

                    loss_op.Dl_l1 = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_Dl_l1), logits=logits_Dl_l1)
                    loss_op.Dl_l2 = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_Dl_l2), logits=logits_Dl_l2)
                    loss_op.Dl = (loss_op.Dl_l1 + loss_op.Dl_l2) / 2

                with task('3. Discriminator visual loss'):
                    logits_Dv_X = self.disc_v(X)
                    logits_Dv_l2 = self.disc_v(self.dec(l2))

                    loss_op.Dv_X = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_Dv_X), logits=logits_Dv_X)
                    loss_op.Dv_l2 = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_Dv_l2), logits=logits_Dv_l2)
                    loss_op.Dv = (loss_op.Dv_X + loss_op.Dv_l2) / 2

                with task('4. Informative-negative mining'):
                    l2_mine = tf.get_variable('l2_mine', [BS] + latent_shape, tf.float32)
                    logits_C_l2_mine = self.cl(self.dec(l2_mine))
                    loss_C_l2_mine = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_C_l2_mine), logits=logits_C_l2_mine)
                    opt = tf.train.GradientDescentOptimizer(1)

                    def cond(i):
                        return i < self.steps.IM

                    def body(i):
                        descent_op = opt.minimize(loss_C_l2_mine, var_list=[l2_mine])
                        with tf.control_dependencies([descent_op]):
                            return i + 1

                    self.l2_mine_descent = tf.while_loop(cond, body, [tf.constant(0)])

                with task('5. AE loss'):
                    Xh = self.dec(l1)
                    loss_AE_recon = tf.reduce_mean(tf.square(X - Xh), axis=[1, 2, 3])
                    loss_op.AE_recon = tf.reduce_mean(loss_AE_recon)

                    loss_op.AE_l = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_Dl_l1), logits=logits_Dl_l1)

                    logits_Dv_l2_mine = self.disc_v(self.dec(l2_mine))
                    loss_op.AE_v = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_Dv_l2_mine), logits=logits_Dv_l2_mine)

                    self.lamb = tf.placeholder_with_default(10., [])
                    loss_op.AE = loss_op.AE_l + loss_op.AE_v + self.lamb * loss_op.AE_recon

            with task('Optimize'):
                Opt = tf.train.AdamOptimizer
                train_op = attrdict()
                ae_vars = self.enc.trainable_variables + self.dec.trainable_variables
                train_op.C = Opt(lr.C).minimize(loss_op.C, var_list=self.cl.trainable_variables)
                train_op.Dl = Opt(lr.Dl).minimize(loss_op.Dl, var_list=self.disc_l.trainable_variables)
                train_op.Dv = Opt(lr.Dv).minimize(loss_op.Dv, var_list=self.disc_v.trainable_variables)
                train_op.AE = Opt(lr.AE).minimize(loss_op.AE, var_list=ae_vars)
                train_op.recon = Opt(lr.recon).minimize(loss_op.AE_recon, var_list=ae_vars)

            with task('Placeholders'):
                self.loss_op = loss_op
                self.train_op = train_op
                self.X = X
                self.Xz = Xz
                self.l2_mine = l2_mine
                self.l2 = l2
                self.anomaly_score = tf.reduce_mean(tf.square(X - self.recon), axis=[1, 2, 3])

    def pretrain_AE(self, X):
        sess = self.sess
        feed_dict = {self.X: X}
        loss = attrdict()
        loss.AE_recon, _ = sess.run([self.loss_op.AE_recon, self.train_op.recon], feed_dict)
        return loss.as_dict()

    def train_step(self, X):
        sess = self.sess
        feed_dict = {self.X: X}
        loss = attrdict()

        with task('1. Train classifier'):
            for _ in range(self.steps.C):
                loss_C, _ = sess.run([self.loss_op.filt_keys('C'), self.train_op.C], feed_dict)
                loss.update(loss_C)

        with task('2. Train discriminators'):
            for _ in range(self.steps.Dl):
                loss_Dl, _ = sess.run([self.loss_op.filt_keys('Dl'), self.train_op.Dl], feed_dict)
                loss.update(loss_Dl)

            for _ in range(self.steps.Dv):
                loss_Dv, _ = sess.run([self.loss_op.filt_keys('Dv'), self.train_op.Dv], feed_dict)
                loss.update(loss_Dv)

        if self.use_informative_mining:
            self.informative_negative_mining()

        with task('4. Train AutoEncoder'):
            for _ in range(self.steps.AE):
                loss_AE, _ = sess.run([self.loss_op.filt_keys('AE'), self.train_op.AE], feed_dict)
                loss.update(loss_AE)

        return loss.as_dict()

    def informative_negative_mining(self):
        self.sess.run(tf.assign(self.l2_mine, self.l2))
        self.sess.run(self.l2_mine_descent)

    def fit(self, x_train, x_test, y_test, epochs=100, ckpt_path='ocgan'):
        BS = self.BS
        saver = tf.train.Saver()
        N = x_train.shape[0]
        sess = self.sess

        for i_epoch in range(epochs):
            keras.backend.set_learning_phase(True)
            results = d_of_l()
            X = x_train[np.random.permutation(N)]
            BN = N // BS  # residual batches are dropped!
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]

                if i_epoch < 20:
                    result = self.pretrain_AE(x_batch)
                else:
                    result = self.train_step(x_batch)

                results.appends(result)

            else:
                with task('Eval test performance'):
                    N_test = BS * (N // BS)
                    auc = self.evaluate(x_test[:N_test], y_test[:N_test])
                    print('Epoch %d AUROC: %.4f' % (i_epoch, auc))
                saver.save(sess, ckpt_path)

                with task('Save Images'):
                    keras.backend.set_learning_phase(False)
                    gens = self.generate()[:BS]
                    gens = assure_dtype_uint8(gens)

                    is_normal = (y_test[:BS] == 1.)[:BS]

                    origin = x_test[:BS]
                    recons = self.reconstruct(origin)
                    recons = gray2rgb(assure_dtype_uint8(recons))
                    recons[is_normal] = add_border(recons[is_normal])

                    origin = gray2rgb(assure_dtype_uint8(origin))[:BS]
                    origin[is_normal] = add_border(origin[is_normal])

                    d = {
                        'example/generated': merge(gens[:64], (8, 8)),
                        'example/original': merge(origin[:64], (8, 8)),
                        'example/recon': merge(recons[:64], (8, 8))
                    }  # example images are created.

    # feedforward

    def generate(self):
        return self.sess.run(self.gen)

    def reconstruct(self, X):
        recon = self.sess.run(self.recon, feed_dict={self.X: X})
        return recon

    def predict(self, X):
        anomaly_scores = list()
        BS = self.BS
        N = X.shape[0]
        BN = N // BS  # residual batches are dropped!
        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            anomaly_score = self.sess.run(self.anomaly_score, feed_dict={self.X: x_batch})
            anomaly_scores.append(anomaly_score)

        return np.concatenate(anomaly_scores)

    def evaluate(self, X, y):
        anomaly_score = self.predict(X)
        auc = roc_auc_score(y, -anomaly_score)
        return auc
