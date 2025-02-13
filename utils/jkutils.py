# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:06:02 2025

@author: Jin Ken
"""
import os
import re
import shutil

def sparse2ForkData(polar_path, dst_dir):
    # 将SparsePDM数据转化为ForkNet格式
    os.makedirs(dst_dir)

    imagenum = 0
    imageforknum = 0
    for dirpath, dirnames, filenames in os.walk(polar_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            _, ext = os.path.splitext(filename)
            assert ext in ['.png', '.bmp', '.jpg'], filepath
            imagenum += 1
            if '135' not in filename:
                continue
            imageforknum += 1
            for j in range(4):
                # j file
                dst_path = os.path.join(dst_dir, f'image_{imageforknum}_{j*45}{ext}')
                assert not os.path.exists(dst_path), f"{dst_path} Exist!"
                
                src_name = re.sub('135', f'{j*45}', filename)
                src_path = os.path.join(dirpath, src_name)
                assert os.path.exists(src_path), src_path
                
                shutil.copy(src_path, dst_path)
            
            
            


if __name__ == '__main__':
    polar_path = './data/Polar-dataset'
    dst_dir = './data/test_set2'
    sparse2ForkData(polar_path, dst_dir)