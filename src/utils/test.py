import utils.start_tf
from utils.start_tf import *
from utils.start_tf import _TAGS
from utils.test_dir import save_make_dir

source_dir = '../data/test/'
save_make_dir(source_dir) 

def test():
    img = Image.open(source_dir+'img.png')
    img = np.reshape(np.array(img), [1, 256, 256, 3])

    # Encoding speed
    print('encoding')
    t = time.time()
    eps = encode(img)
    # for _ in tqdm(range(10)):
    #     eps = encode(img)
    print("Encoding latency {} sec/img".format((time.time() - t) / (1)))

    # Decoding speed
    print('decoding')
    t = time.time()
    dec = decode(eps)
    # for _ in tqdm(range(10)):
    #     dec = decode(eps)
    print("Decoding latency {} sec/img".format((time.time() - t) / (1)))
    img = Image.fromarray(dec[0])
    img.save(source_dir+'dec.png')

    # Manipulation
    print('manipulation')
    dec, _ = manipulate(eps, _TAGS.index('Smiling'), 0.66)
    img = Image.fromarray(dec[0])
    img.save(source_dir+'smile.png')


if __name__ == '__main__':
    print('test')
    test()
    print('done')
