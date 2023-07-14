# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> args_kwargs
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/14 16:58
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/14 16:58:
==================================================  
"""
__author__ = 'zxx'

"""
args是arguments的缩写，表示位置参数；kwargs是keyword arguments 的缩写，表示关键字参数

· 不一定要写成 *args和**arguments，只有前面的*是必须的
·向python传递参数的两种方式
    1. 位置参数
    2.关键字参数

*args 表示任何多个无名参数，本质上是一个tuple
**arguments 表示关键字参数，本质上是一个dict

同时使用时要求*args参数列必须要在**kwargs前面，因为位置参数在关键字参数的前面
"""

# Splat 运算符
# * 常常与乘法运算，C语言中的指针 有关，但在python中，则是 splat 运算符的 两倍。且 splat运算符需要更强大的类比

a = [1, 2, 3]
b = [*a, 4, 5, 6]
print(b)
# ----------------- 输出结果 -----------------
# [1, 2, 3, 4, 5, 6]
# ----------------- 总结 -----------------
# 将a的内容移入（解包）到新列表b中。

"""
*args 的用法
* args 和 ** args 主要用于函数定义，你可以将不定数量的参数传递给一个函数。

这里不定的意思是： 预先并不知道，函数使用者会传递多少个参数给你，所在在这个场景下使用这两个关键字。 * args 是用来发送一个 非键值 的可变数量的参数列表给一个函数。
"""


def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)


test_var_args('yasoob', 'python', 'eggs', 'test')


# 定义一个打印的函数，传入任意参数即可

def print_func(*args):
    print(type(args))
    print(args)


print_func(1, 2, 'python希望社', [])


# 在打印函数的参数处，新增 x 和 y 变量

def print_func1(x, y, *args):
    print(type(x))
    print(x)
    print(y)
    print(type(args))
    print(args)


print_func1(1, 2, 'python希望社', [])


# 将 *args 放在参数最前面

def print_func2(*args, x, y):
    print(type(x))
    print(x)
    print(y)
    print(type(args))
    print(args)


print_func2(1, 2, 'python希望社', [], x='x', y='y')

"""
** kwargs 的用法
**kwargs允许你将不定长度的 【键值对 key-value 】，作为参数传递给一个函数。如果你想要在一个函数里处理带名字的参数，你应该使用**kwargs
"""


def print_func3(**kwargs):
    print(type(kwargs))
    print(kwargs)


print_func3(a=1, b=2, c='呵呵哒', d=[])


def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))


greet_me(name="yasoob")

"""
arg,*args,**kwargs ,三者是可以组合使用的，但是组合使用需要遵循一定的语法规则，即顺序为王。

需要按照：

arg,*args,**kwargs 作为函数参数的顺序。
"""


def print_func(x, *args, **kwargs):
    print(x)
    print(args)
    print(kwargs)


print_func(1, 2, 3, 4, y=1, a=2, b=3, c=4)


# 那现在我们将看到怎样使用*args 和 **kwargs 来调用一个函数
def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)


# 那现在我们将看到怎样使用*args 和 **kwargs 来调用一个函数
# 首先你可以使用 *args
args = ("two", 3, 5)
test_args_kwargs(*args)

# -------- 得到输出结果如下所示：----------------------
# arg1: two
# arg2: 3
# arg3: 5
# ---------------------------------------------------

#  现在使用 **kwargs
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)

# -------- 得到输出结果如下所示：----------------------
# arg1: 5
# arg2: two
# arg3: 3
# ---------------------------------------------------
