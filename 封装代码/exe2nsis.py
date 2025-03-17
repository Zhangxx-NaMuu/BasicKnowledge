# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> exe2nsis
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/17 10:22
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/17 10:22:
==================================================  
"""
__author__ = 'zxx'

import os


def exe2nsis(work_dir:str,
             files_to_compress:list,
             exe_name:str,
             appname:str="AI",
             version:str="1.0.0.0",
             author:str="zhangxx",
             license:str="",
             icon_old:str=""):
    """
    将exe文件转换为nsis安装程序
    Note:
        files_to_compress =[f"{self.work_dir}/{i} for i in ["app", "py", third", "app.exe","requirements.txt"]] # 需要压缩的文件列表，可以是目录或文件，如果是目录，则会递归压缩目录下所有文件
    Args:
        work_dir: 工作目录
        files_to_compress: 需要转换的文件夹/文件列表
        exe_name: 指定著运行程序，快捷方式也是用此程序生成
        appname: 产品名
        version: 版本号--必须为X.X.X.X格式
        author: 作者名
        license: 许可证
        icon_old: 图标路径
    """
    # 获取当前脚本的绝对路径
    exe_7z_path = os.path.abspath("./bin/7z/7z.exe")
    exe_nsis_path = os.path.abspath("./bin/NSIS/makensis.exe")
    config_path = os.path.abspath("./bin/config")
    print(exe_7z_path)

    # 压缩app目录
    app_7z_path = f"{work_dir}/app.7z"
    if os.path.exists(app_7z_path):
        print(f"已存在{app_7z_path}, 跳过压缩步骤")
    else:
        print(f"不存在{app_7z_path}， 开始压缩步骤")
    # 替换文件
    nsis_code = f"""
        # ====================== 自定义宏 产品信息==============================
        !define PRODUCT_NAME           		"{appname}"
        !define PRODUCT_PATHNAME           	"{appname}"     #安装卸载项用到的KEY
        !define INSTALL_APPEND_PATH         "{appname}"     #安装路径追加的名称 
        !define INSTALL_DEFALT_SETUPPATH    ""       #默认生成的安装路径 
        !define EXE_NAME               		"{exe_name}" # 指定主运行程序，快捷方式也是用此程序生成
        !define PRODUCT_VERSION        		"{version}"
        !define PRODUCT_PUBLISHER      		"{author}"
        !define PRODUCT_LEGAL          		"${{PRODUCT_PUBLISHER}} Copyright（c）2023"
        !define INSTALL_OUTPUT_NAME    		"{appname}_V{version}.exe"
        
        # ====================== 自定义宏 安装信息==============================
        !define INSTALL_7Z_PATH 	   		"{work_dir}\\app.7z"
        !define INSTALL_7Z_NAME 	   		"app.7z"
        !define INSTALL_RES_PATH       		"skin.zip"
        !define INSTALL_LICENCE_FILENAME    "{os.path.join(config_path, "license.txt") if license == "" else license}"
        !define INSTALL_ICO 				"{os.path.join(config_path, "logo.ico") if icon_old == "" else icon_old}"
        
        
        !include "{os.path.join(config_path, "ui.nsh")}"
        
        # ==================== NSIS属性 ================================
        
        # 针对Vista和win7 的UAC进行权限请求.
        # RequestExecutionLevel none|user|highest|admin
        RequestExecutionLevel admin
        
        #SetCompressor zlib
        
        ; 安装包名字.
        Name "${{PRODUCT_NAME}}"
        
        # 安装程序文件名.
        
        OutFile "{work_dir}\\{appname}_V{version}.exe"
        
        InstallDir "1"
        
        # 安装和卸载程序图标
        Icon              "${{INSTALL_ICO}}"
        UninstallIcon     "uninst.ico"
        """
    # 执行封装命令
    nsis_path = os.path.join(config_path, "output.nsi")
    with open(nsis_path, "w", encoding="gb2312") as file:
        file.write(nsis_code)
    print([f"{exe_nsis_path}", nsis_path])
    try:
        subprocess.run([f"{exe_nsis_path}", nsis_path])
    except Exception as e:
        print(e)
        print([f"{exe_nsis_path}", nsis_path])

    # 清理文件
    os.remove(nsis_path)
    if os.path.exists(os.path.join(work_dir, f"{appname}_V{version}.exe")):
        os.remove(f"{work_dir}/app.7z")
        return True
    else:
        return False



