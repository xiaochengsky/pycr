# -*- coding: utf-8 -*-
# @Time : 2022/12/1 下午2:49
# @Author : XXX
# @File : test.py

from abc import abstractmethod


class A:
    def __init__(self):
        self.a = None
        self.b = None

    @abstractmethod
    def aa(self):
        print(self, 'AA')

    def forward(self):
        print(self, self.a + self.b)
        self.aa()


class B(A):
    def __init__(self):
        super(B, self).__init__()
        self.a = 1
        self.b = 2

    def aa(self):
        print(self, 'BB')

    # def forward(self):
    #     print(self, 'BBB-forward')


b = B()
b.forward()
