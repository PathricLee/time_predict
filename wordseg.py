#_*_coding:utf-8_*_
import sys

for line in sys.stdin:
    line = line.strip()
    line = line.decode('utf-8')
    print ' '.join(line).encode('utf-8')
