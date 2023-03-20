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
