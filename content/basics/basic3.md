---
title: Basic3
date: 2025-07-01
author: Your Name
cell_count: 20
score: 20
---

```python
def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

print(factorial(int(input("Enter number: "))))

```

    Enter number:  62
    

    31469973260387937525653122354950764088012280797258232192163168247821107200000000000000
    


```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

n = int(input("Enter n: "))
for i in range(n):
    print(fib(i), end=' ')

```

    Enter n:  5
    

    0 1 1 2 3 


```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

print("Prime" if is_prime(int(input("Enter number: "))) else "Not Prime")

```

    Enter number:  7
    

    Prime
    


```python
def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

print("GCD:", gcd(int(input("Enter a: ")), int(input("Enter b: "))))

```

    Enter a:  2
    Enter b:  2
    

    GCD: 2
    


```python
def sum_digits(n):
    if n == 0:
        return 0
    return n % 10 + sum_digits(n // 10)

print("Sum:", sum_digits(int(input("Enter number: "))))

```

    Enter number:  32
    

    Sum: 5
    


```python
def reverse(s):
    if len(s) == 0:
        return ""
    return reverse(s[1:]) + s[0]

print("Reversed:", reverse(input("Enter string: ")))

```

    Enter string:  stefina
    

    Reversed: anifets
    


```python
def count_words(sentence):
    return len(sentence.split())

print("Word count:", count_words(input("Enter sentence: ")))

```

    Enter sentence:  nothing is permanent 
    

    Word count: 3
    


```python
def is_palindrome(s):
    return s == s[::-1]

print("Palindrome" if is_palindrome(input("Enter string: ")) else "Not Palindrome")

```

    Enter string:  madam
    

    Palindrome
    


```python
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp - 1)

print("Power:", power(int(input("Base: ")), int(input("Exponent: "))))

```

    Base:  2
    Exponent:  3
    

    Power: 8
    


```python
def count_case(s):
    upper = sum(1 for c in s if c.isupper())
    lower = sum(1 for c in s if c.islower())
    return upper, lower

s = input("Enter string: ")
u, l = count_case(s)
print("Uppercase:", u)
print("Lowercase:", l)

```

    Enter string:  Stefina E
    

    Uppercase: 2
    Lowercase: 6
    


```python
def is_armstrong(n):
    digits = len(str(n))
    return n == sum(int(d)**digits for d in str(n))

print("Armstrong" if is_armstrong(int(input("Enter number: "))) else "Not Armstrong")

```

    Enter number:  7
    

    Armstrong
    


```python
def maximum(a, b, c):
    return max(a, b, c)

print("Max:", maximum(3, 8, 5))

```

    Max: 8
    


```python
def sum_list(lst):
    return sum(lst)

lst = list(map(int, input("Enter numbers: ").split()))
print("Sum:", sum_list(lst))

```

    Enter numbers:  20
    

    Sum: 20
    


```python
def dec_to_bin(n):
    return bin(n)[2:]

print("Binary:", dec_to_bin(int(input("Enter decimal: "))))

```

    Enter decimal:  3
    

    Binary: 11
    


```python
def bin_to_dec(b):
    return int(b, 2)

print("Decimal:", bin_to_dec(input("Enter binary: ")))

```

    Enter binary:  1010
    

    Decimal: 10
    


```python
def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

print("Leap year" if is_leap(int(input("Enter year: "))) else "Not a leap year")

```

    Enter year:  2024
    

    Leap year
    


```python
def square(n):
    return n * n

print("Square:", square(int(input("Enter number: "))))

```

    Enter number:  60
    

    Square: 3600
    


```python
import random
import string

def generate_password(length):
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

print("Password:", generate_password(int(input("Enter length: "))))

```

    Enter length:  4
    

    Password: z']R
    


```python
def to_seconds(h, m, s):
    return h * 3600 + m * 60 + s

h, m, s = map(int, input("Enter time (hh mm ss): ").split())
print("Total seconds:", to_seconds(h, m, s))

```

    Enter time (hh mm ss):  2 30 2
    

    Total seconds: 9002
    


```python

```


---
**Score: 20**