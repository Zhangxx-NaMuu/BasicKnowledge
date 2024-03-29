`1.filter() `

函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
"""

```python
x = ['t', 'None', 's']
list(filter(lambda i: i != 'None', x))  # ['t', 's']
```

`2.next() `

返回迭代器的下一个项目。

`next()` 函数要和生成迭代器的 `iter()` 函数一起使用。

```python
next(iter([1, 2, 3, 4]))   # 1
```

`3.yield()`

该函数有返回某种结果的职责

与`return`的区别是：`return`直接返回所有的结果，程序终止不在运行；`yield`返回的是可迭代的生成器对象，可以使用`for`循环
或者`next`遍历生成器对象来提取结果。

`4.torch.pow()`
求幂指数操作
torch.pow(3, 2) 即求3的2次幂