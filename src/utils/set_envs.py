import os

try:
    if os.environ['COMPUTERNAME'] == 'LAPTOP-PFTI92GE':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
except:
    pass
