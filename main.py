import tensorflow as tf
from src.utils import attrdict
from src.ocgan import OCGAN
from src.dataset import get_mnist


def train(use_informative_mining=True, cls=1):
    x_train, x_test, y_test = get_mnist(cls)
    BS = 128

    steps = attrdict({
        'C': 5,
        'AE': 5,
        'Dl': 3,
        'Dv': 2,
        'IM': 5
    })
    lr = attrdict({
        'C': 1e-4,
        'recon': 3e-4,
        'Dl': 1e-5,
        'Dv': 1e-5,
        'AE': 3e-5,
    })

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ocgan = OCGAN(sess, steps, lr, BS, use_informative_mining=use_informative_mining)
        if use_informative_mining:
            tb_name = 'OCGAN(%s)' % cls
        else:
            tb_name = 'OCGAN_NOIM(%s)' % cls

        sess.run(tf.global_variables_initializer())
        ckpt_path = './ckpts/%s/%s' % (tb_name, tb_name)
        ocgan.fit(x_train, x_test, y_test, epochs=50, ckpt_path=ckpt_path)


def main():
    for cls in range(10):
        tf.reset_default_graph()
        train(True, cls)


if __name__ == '__main__':
    main()
