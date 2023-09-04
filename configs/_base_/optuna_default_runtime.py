# -*- coding: utf-8 -*-
# @Time : 2022/11/28 下午2:40
# @Author : YANG.C
# @File : default_runtime.py


# runtime setting
max_epochs = 30
log_periods = 10
log_dir = r'./logs'
tensorboard_dir = r'tensorboard'
save_dir = r'checkpoints'
log_file = f'result.txt'

#   |-log
#       |-{model_name_tag}
#           |-checkpoints
#       |-tensorboard
#       |-log_file

dist_params = dict(backend='nccl')
