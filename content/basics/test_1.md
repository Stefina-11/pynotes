---
title: Test 1
date: 2025-06-27
author: Your Name
cell_count: 4
score: 0
---

```python
print("Hello World")
```

    Hello World
    


```python
def solve():
    try:
        n_input = input("Enter number of elements: ").strip()
        if not n_input:
            print("No input provided for number of elements.")
            return  
        n = int(n_input)  
        arr_input = input("Enter the array elements separated by space: ").strip()
        if not arr_input:
            print("No array elements provided.")
            return
        arr = list(map(int, arr_input.split()))
        if len(arr) != n:
            print(f"You entered {len(arr)} elements, but expected {n}.")
            return
        max_ending_here = max_so_far = arr[0]
        for x in arr[1:]:
            max_ending_here = max(x, max_ending_here + x)
            max_so_far = max(max_so_far, max_ending_here)
        print("Maximum Subarray Sum:", max_so_far)
    except ValueError as ve:
        print("Invalid input! Please enter integers only.")
        print("Error details:", ve)
solve()

```

    Enter number of elements:  2
    Enter the array elements separated by space:  10 10
    

    Maximum Subarray Sum: 20
    


```python
import bisect
def solve():
    try:
        arr_input = input().strip()
        if not arr_input:
            return
        arr = list(map(int, arr_input.split()))
        if not arr:
            return
        sub = []
        for num in arr:
            idx = bisect.bisect_left(sub, num)
            if idx == len(sub):
                sub.append(num)
            else:
                sub[idx] = num
        print(len(sub))
    except ValueError:
        print("Invalid input")
solve()

```

     10 9 2 5 3 7 101 18
    

    4
    


```python

```


---
**Score: 0**