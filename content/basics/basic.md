---
title: Basic
date: 2025-07-01
author: Your Name
cell_count: 20
score: 20
---

```python

a = 5
b = 3

```


```python
print(a + b)


```

    8
    


```python
print(a - b)


```

    2
    


```python
print(a * b)


```

    15
    


```python
print(a / b)


```

    1.6666666666666667
    


```python
num = 7
print("Even" if num % 2 == 0 else "Odd")


```

    Odd
    


```python
num = -10
if num > 0:
    print("Positive")
elif num < 0:
    print("Negative")
else:
    print("Zero")


```

    Negative
    


```python
a, b = 10, 20
a, b = b, a
print(a, b)


```

    20 10
    


```python
import math
print(math.sqrt(16))


```

    4.0
    


```python
r = 5
area = math.pi * r ** 2
print(area)


```

    78.53981633974483
    


```python
c = 37
f = (c * 9/5) + 32
print(f)


```

    98.6
    


```python
km = 5
miles = km * 0.621371
print(miles)


```

    3.106855
    


```python
p, r, t = 1000, 5, 2
si = (p * r * t) / 100
print(si)


```

    100.0
    


```python
ci = p * (pow((1 + r/100), t)) - p
print(ci)


```

    102.5
    


```python
n = 5
fact = 1
for i in range(1, n + 1):
    fact *= i
print(fact)


```

    120
    


```python
n = 7
for i in range(1, 11):
    print(f"{n} x {i} = {n*i}")


```

    7 x 1 = 7
    7 x 2 = 14
    7 x 3 = 21
    7 x 4 = 28
    7 x 5 = 35
    7 x 6 = 42
    7 x 7 = 49
    7 x 8 = 56
    7 x 9 = 63
    7 x 10 = 70
    


```python
n = 10
a, b = 0, 1
for _ in range(n):
    print(a, end=' ')
    a, b = b, a + b
print()


```

    0 1 1 2 3 5 8 13 21 34 
    


```python
x, y, z = 10, 25, 5
print(max(x, y, z))

print(min(x, y, z))


```

    25
    5
    


```python
year = 2024
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print("Leap Year")
else:
    print("Not a Leap Year")

```

    Leap Year
    


```python

```


---
**Score: 20**