# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> py2pyd
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/17 09:36
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/17 09:36:
==================================================  
"""
__author__ = 'zxx'
import os
import shutil
import socket
import subprocess
import threading
import time
import zipfile
import requests
from setuptools import Extension, setup

from Cython.Build import cythonize
from tqdm import tqdm

def py2pyd(source_path:str, clear_by: bool=False):
    """
    将目录下所有py文件编译成pyd文件
    Args:
        source_path: 待编译的目录路径
        clear_by: 是否编译后清除py文件，注意备份
    """
    tmp_path = os.path.join(source_path, 'tmp')
    if not os.path.exists(tmp_path):
        print(f"创建临时目录{tmp_path}")
        os.makedirs(tmp_path)

    # 遍历目录下所有的文件
    for root, dirs, files in os.walk(source_path):
        if dirs != "tmp":
            for file in files:
                # 判断文件名是否以.py结尾
                if file.endswith('.py'):
                    # 编译文件
                    if file == '__init__.py':
                        continue
                    else:
                        # 构建文件的完整路径
                        file_path = os.path.join(root, file)
                        # 构建扩展模块名称
                        module_name = os.path.splitext(file)[0]
                        # 构建扩展对象
                        extension = Extension(module_name, [file_path])
                        print("build:", extension)
                        setup(
                            ext_modules=cythonize(extension, compiler_directives={'language_level': 3}, force=True),
                            script_args=['build_ext', #'--inplace'
                            "--build-lib", f"{tmp_path}", "--build-temp", f"{tmp_path}" ,]
                        )

                        # 移动pyd
                        for f_pyd in os.listdir(tmp_path):
                            if f_pyd.endswith('.pyd'):
                                if f_pyd.split('.')[0] == module_name:
                                    # 保证一次只处理一个文件
                                    pyd_name = f_pyd.split('.')[0] + '.pyd'
                                    old_path = os.path.join(tmp_path, f_pyd)
                                    new_path = os.path.join(root, pyd_name)
                                    try:
                                        print(f"move {old_path} to {new_path}")
                                        os.rename(old_path, new_path)
                                        if clear_by:
                                            print(f"remove {file_path}")
                                            os.remove(file_path)

                                    except Exception as e:
                                        print("Exception:", e)

                        # 删除.c文件
                        c_file = file_path.replace('.py', '.c')
                        print("del:", c_file)
                        os.remove(c_file)
    # 删除临时目录
    if os.path.exists(tmp_path):
        print("del tmp dir:", tmp_path)
        shutil.rmtree(tmp_path)

# 调用示例
py2pyd(r"C:\Users\dell\Desktop\AI_Services\AI_V3_copy")