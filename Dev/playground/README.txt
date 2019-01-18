运行
python setup.py build
python testima.py

A. 目前的状态: 正常打印100.0
B. 若将 test.cpp 第37, 38行去除注释: build能通过, 运行脚本就报
Traceback (most recent call last):
  File "testima.py", line 4, in <module>
    from testa import test
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x81 in position 11: invalid start byte
C. 若再将 test.cpp 第20, 29行注释: build能通过, 运行脚本就报
Traceback (most recent call last):
  File "testima.py", line 4, in <module>
    from testa import test
ValueError: module functions cannot set METH_CLASS or METH_STATIC
