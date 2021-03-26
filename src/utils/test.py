import start_tf
from start_tf import *
from start_tf import _TAGS


def test():
    img = Image.open('../../data/test/img.png')
    img = np.reshape(np.array(img), [1, 256, 256, 3])

    # Encoding speed
    print('encoding')
    eps = encode(img)
    t = time.time()
    # for _ in tqdm(range(10)):
    #     eps = encode(img)
    # print("Encoding latency {} sec/img".format((time.time() - t) / (1 * 10)))

    # Decoding speed
    print('decoding')
    dec = decode(eps)
    # t = time.time()
    # for _ in tqdm(range(10)):
    #     dec = decode(eps)
    # print("Decoding latency {} sec/img".format((time.time() - t) / (1 * 10)))
    img = Image.fromarray(dec[0])
    img.save('../../data/test/dec.png')

    # Manipulation
    print('manipulation')
    dec, _ = manipulate(eps, _TAGS.index('Smiling'), 0.66)
    img = Image.fromarray(dec[0])
    img.save('../../data/test/smile.png')


if __name__ == '__main__':
    print('test')
    test()
    print('done')
