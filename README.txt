# The paper
    https://arxiv.org/pdf/1807.03039.pdf

# The code
    https://github.com/openai/glow
    https://github.com/EGimenez/TFM

# Models Glow & Face Detector
    # To get model weights and manipulation vectors execute
    execute src/models/download_models.sh
    # Face Detector
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    Store it in src/preprocessing/shape_predictor_68_face_landmarks.dat

# Data
    Copy the data directory contained in the TFM into your desired location

# Get CelebA data set [This one IS correct]
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28
    Download the img_celeba.7z.xxx
    Glue them together -> 7z e img_celeba.7z.001 -tsplit
    Un7z them -> 7za e img_celeba.7z

    These are raw images -> We will have to process them and extract 256x256 images of the faces
    Use preprocessing/ImageCropping.py

    From list_attr_celeba.txt we create list_attr_celeba_clean.txt including a colum_name for image [img_id] turnning -1 -> 0 and managing space_limits properlly.
    It is contained int data/celeba_wild/index



# config.py
    Define que src, models, data, and results files are
