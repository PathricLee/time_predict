#_*_coding:utf-8_*_
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
def add(x, y, z):
    try:
        t = x / y
    except:
        t = 1
    print(t)
    print(int(0.4449))
    print(int(1.5111))

def test_str():
    input_ = '我是百度国际科技'
    print(type(input_))
    lst = list(input_)
    print(lst)
    

if __name__ == '__main__':
    #add(1, 2, 3)
    test_str()
