# -*- coding: utf-8 -*-
# @Time : 2022/11/29 上午11:19
# @Author : YANG.C
# @File : log.py

import logging

format_str = '%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
format = logging.Formatter(format_str)
logging.basicConfig(format=format_str,
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()
