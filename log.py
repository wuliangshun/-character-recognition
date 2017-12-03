# -*- coding: utf-8 -*-
"""
Log
"""

import time
    
def log(content,filename='./log/log.txt'):
    with open(filename,'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ':\t' + content)
        f.write('\n')
        
def log_and_print(content,filename='./log/log.txt',encoding='gbk'):
    with open(filename,'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ':\t' + content)
        f.write('\n')
    print(content)
    

    
