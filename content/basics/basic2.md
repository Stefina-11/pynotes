---
title: Basic2
date: 2025-07-01
author: Your Name
cell_count: 21
score: 20
---

```python
n = int(input("Enter a number: "))
sum = 0
for i in range(1, n+1):
    sum += i
print("Sum:", sum)

```

    Enter a number:  2
    

    Sum: 3
    


```python
n = int(input("Enter a number: "))
count = 0
while n > 0:
    count += 1
    n //= 10
print("Digit count:", count)

```

    Enter a number:  4
    

    Digit count: 1
    


```python
n = int(input("Enter a number: "))
rev = 0
while n > 0:
    rev = rev * 10 + n % 10
    n //= 10
print("Reversed number:", rev)

```

    Enter a number:  20
    

    Reversed number: 2
    


```python
n = int(input("Enter a number: "))
temp = n
rev = 0
while n > 0:
    rev = rev * 10 + n % 10
    n //= 10
print("Palindrome" if temp == rev else "Not Palindrome")

```

    Enter a number:  2
    

    Palindrome
    


```python
start = int(input("Start: "))
end = int(input("End: "))
for num in range(start, end+1):
    if num > 1:
        for i in range(2, int(num**0.5)+1):
            if num % i == 0:
                break
        else:
            print(num, end=' ')

```

    Start:  2
    End:  5
    

    2 3 5 


```python
n = int(input("Enter number: "))
temp = n
digits = len(str(n))
sum = 0
while n > 0:
    digit = n % 10
    sum += digit ** digits
    n //= 10
print("Armstrong" if sum == temp else "Not Armstrong")

```

    Enter number:  8
    

    Armstrong
    


```python
rows = int(input("Enter rows: "))
for i in range(1, rows + 1):
    print('*' * i)

```

    Enter rows:  5
    

    *
    **
    ***
    ****
    *****
    


```python
base = int(input("Base: "))
exp = int(input("Exponent: "))
result = 1
for _ in range(exp):
    result *= base
print("Power:", result)

```

    Base:  2
    Exponent:  2
    

    Power: 4
    


```python
n = int(input("Enter number: "))
total = 0
while n > 0:
    total += n % 10
    n //= 10
print("Sum of digits:", total)

```

    Enter number:  10
    

    Sum of digits: 1
    


```python
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
while b:
    a, b = b, a % b
print("GCD:", a)

```

    Enter first number:  40
    Enter second number:  20
    

    GCD: 20
    


```python
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
lcm = a * b // gcd(a, b)
print("LCM:", lcm)

```

    Enter first number:  15
    Enter second number:  12
    

    LCM: 60
    


```python
import math
n = int(input("Enter number: "))
temp = n
sum = 0
while n > 0:
    sum += math.factorial(n % 10)
    n //= 10
print("Strong number" if sum == temp else "Not strong number")

```

    Enter number:  145
    

    Strong number
    


```python
n = int(input("Enter number: "))
sum = 0
for i in range(1, n):
    if n % i == 0:
        sum += i
print("Perfect number" if sum == n else "Not perfect number")

```

    Enter number:  6
    

    Perfect number
    


```python
char = input("Enter a character: ")
print("ASCII value:", ord(char))

```

    Enter a character:  A
    

    ASCII value: 65
    


```python
del sum  
text = input("Enter text: ").lower()
vowels = "aeiou"
count = sum(1 for ch in text if ch in vowels)
print("Vowel count:", count)

```

    Enter text:  hello world
    

    Vowel count: 3
    


```python
text = input("Enter text: ").lower()
vowels = "aeiou"
count = sum(1 for ch in text if ch.isalpha() and ch not in vowels)
print("Consonant count:", count)

```

    Enter text:  hello world
    

    Consonant count: 7
    


```python
n = int(input("Enter range limit: "))
for i in range(1, n+1):
    if i % 3 == 0 and i % 5 == 0:
        print(i, end=' ')

```

    Enter range limit:  100
    

    15 30 45 60 75 90 


```python
n = int(input("Enter limit: "))
sum = 0
for i in range(2, n+1, 2):
    sum += i
print("Sum of even numbers:", sum)

```

    Enter limit:  50
    

    Sum of even numbers: 650
    


```python
n = int(input("Enter limit: "))
sum = 0
for i in range(1, n+1, 2):
    sum += i
print("Sum of odd numbers:", sum)

```

    Enter limit:  25
    

    Sum of odd numbers: 169
    


```python
n = int(input("Enter n: "))
sum = 0
for i in range(1, n+1):
    sum += 1/i
print("Sum of series:", round(sum, 4))

```

    Enter n:  4
    

    Sum of series: 2.0833
    


```python

```


---
**Score: 20**