import os

if os.environ['COMPUTERNAME'] == 'LAPTOP-PFTI92GE':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
