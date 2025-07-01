---
title: Basic4
date: 2025-07-01
author: Your Name
cell_count: 20
score: 20
---

```python
def is_anagram(str1, str2):
    return sorted(str1.lower()) == sorted(str2.lower())

print("Anagram" if is_anagram(input("First string: "), input("Second string: ")) else "Not Anagram")

```

    First string:  listen
    Second string:  silent
    

    Anagram
    


```python
import string

def is_pangram(s):
    s = s.lower()
    return all(ch in s for ch in string.ascii_lowercase)

print("Pangram" if is_pangram(input("Enter sentence: ")) else "Not Pangram")

```

    Enter sentence:  The quick brown fox jumps over the lazy dog
    

    Pangram
    


```python
s = input("Enter string: ")
vowels = "aeiouAEIOU"
result = ''.join('*' if ch in vowels else ch for ch in s)
print("Modified string:", result)

```

    Enter string:  stefina
    

    Modified string: st*f*n*
    


```python
s = input("Enter string: ")
freq = {}
for ch in s:
    freq[ch] = freq.get(ch, 0) + 1
print(freq)

```

    Enter string:  hey there
    

    {'h': 2, 'e': 3, 'y': 1, ' ': 1, 't': 1, 'r': 1}
    


```python
s = input("Enter string: ")
seen = set()
duplicates = set()
for ch in s:
    if ch in seen:
        duplicates.add(ch)
    else:
        seen.add(ch)
print("Duplicates:", duplicates)

```

    Enter string:  happy
    

    Duplicates: {'p'}
    


```python
s = input("Enter sentence: ")
print("Capitalized:", s.title())

```

    Enter sentence:  apple
    

    Capitalized: Apple
    


```python
words = input("Enter sentence: ").split()
words.sort()
print("Sorted words:", ' '.join(words))

```

    Enter sentence:  how are you
    

    Sorted words: are how you
    


```python
import string

s = input("Enter text: ")
clean = ''.join(ch for ch in s if ch not in string.punctuation)
print("Cleaned:", clean)

```

    Enter text:  what's up!
    

    Cleaned: whats up
    


```python
s = input("Enter string: ")
print("Without spaces:", s.replace(" ", ""))

```

    Enter string:  h e l l o
    

    Without spaces: hello
    


```python
list1 = [1, 2, 3]
list2 = [4, 5]
merged = list1 + list2
print("Merged list:", merged)

```

    Merged list: [1, 2, 3, 4, 5]
    


```python
lst = list(map(int, input("Enter numbers: ").split()))
print("Sum:", sum(lst))
print("Min:", min(lst))
print("Max:", max(lst))

```

    Enter numbers:  1 2 3
    

    Sum: 6
    Min: 1
    Max: 3
    


```python
lst = list(map(int, input("Enter numbers: ").split()))
unique = list(set(lst))
print("Without duplicates:", unique)

```

    Enter numbers:  1 2 3 2 4 5 5 6 
    

    Without duplicates: [1, 2, 3, 4, 5, 6]
    


```python
lst = list(map(int, input("Enter list: ").split()))
print("Reversed:", lst[::-1])

```

    Enter list:  1 2 3
    

    Reversed: [3, 2, 1]
    


```python
lst = list(map(int, input("Enter numbers: ").split()))
unique = list(set(lst))
unique.sort()
if len(unique) >= 2:
    print("Second largest:", unique[-2])
else:
    print("Not enough unique elements")

```

    Enter numbers:  1 2 3 4
    

    Second largest: 3
    


```python
lst = list(map(int, input("Enter list: ").split()))
x = int(input("Enter number to count: "))
print(f"{x} occurs {lst.count(x)} times")

```

    Enter list:  2 4 6
    Enter number to count:  4
    

    4 occurs 1 times
    


```python
s = input("Enter comma-separated values: ")
items = s.split(',')
print("Split list:", items)

```

    Enter comma-separated values:  c,l,a,s,s
    

    Split list: ['c', 'l', 'a', 's', 's']
    


```python
lst = [(2, 'b'), (1, 'a'), (3, 'c')]
lst.sort()
print("Sorted tuples:", lst)

```

    Sorted tuples: [(1, 'a'), (2, 'b'), (3, 'c')]
    


```python
a = list(map(int, input("List A: ").split()))
b = list(map(int, input("List B: ").split()))
common = list(set(a) & set(b))
print("Common elements:", common)

```

    List A:  1 2 3 4 5
    List B:  6 7 8 3 9
    

    Common elements: [3]
    


```python
nested = [[1, 2], [3, 4], [5]]
flat = [item for sublist in nested for item in sublist]
print("Flattened:", flat)

```

    Flattened: [1, 2, 3, 4, 5]
    


```python

```


---
**Score: 20**