import os


def save_make_dir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
