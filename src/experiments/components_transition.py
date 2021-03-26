import utils.start_tf
from utils.start_tf import *
from utils.test_dir import save_make_dir

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/components_transition/'

save_make_dir(source_dir)
save_make_dir(result_dir)


def test_drop_components_1_1_transition():
    from PIL import Image
    from numpy import linalg as LA

    # Modifications
    # 195309
    # 194582
    # 192426
    # 120327

    # People
    # 082099
    # 032080

    peoples = ['082099', '032080']
    components = [195309, 194582, 192426, 120327]

    for p in peoples:
        print(p)
        img_eg = Image.open(source_dir+p+'.jpg')
        new_size = (256, 256)
        new_img_eg = img_eg.resize(new_size)
        img_eg = np.reshape(np.array(new_img_eg), [1, 256, 256, 3])
        eps_eg = encode(img_eg)

        for c in components:
            print(c)
            eps_eg_100 = eps_eg.copy()

            for t in np.arange(-100, 101, 2):
                print(t)
                eps_eg_100[0, c] = t
                dec = decode(eps_eg_100)
                img = Image.fromarray(dec[0])
                img.save(result_dir+'components_p_'+str(p)+'_c_'+str(c)+'_t_'+str(t)+'.png')

    print('hola')

if __name__ == '__main__':
    print('test_2')
    #test2()
    test_drop_components_1_1_transition()
    print('done')
    #test3()
