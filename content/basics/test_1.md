---
title: Test 1
date: 2025-06-29
author: Your Name
cell_count: 24
score: 20
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
def solve():
    try:
        s1 = input().strip()
        s2 = input().strip()
        n, m = len(s1), len(s2)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        print(dp[n][m])
    except:
        print("Invalid input")
solve()

```

     Stefina
     Ai Engineer
    

    2
    


```python
def solve():
    try:
        arr = list(map(int, input().strip().split()))
        target = int(input().strip())
        n = len(arr)
        dp = [[False]*(target+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = True
        for i in range(1, n+1):
            for j in range(1, target+1):
                if arr[i-1] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-arr[i-1]]
        print("Yes" if dp[n][target] else "No")
    except:
        print("Invalid input")
        
solve()

```

     1  2 3 4 5
     7
    

    Yes
    


```python
def solve():
    arr = list(map(int, input().strip().split()))
    count = 0
    candidate = None
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    print(candidate)
solve()

```

     2 2 3 4 2 5 2 6
    

    2
    


```python
def solve():
    arr = list(map(int, input().strip().split()))
    target = int(input().strip())
    seen = {}
    for i, num in enumerate(arr):
        diff = target - num
        if diff in seen:
            print(seen[diff], i)
            return
        seen[num] = i
    print("No pair")
solve()

```

     1 2 3 4
     5
    

    1 2
    


```python
def solve():
    s = input().strip()
    print("Yes" if s == s[::-1] else "No")
solve()

```

     madam
    

    Yes
    


```python
def solve():
    s1 = input().strip()
    s2 = input().strip()
    print("Yes" if sorted(s1) == sorted(s2) else "No")
solve()

```

     listen
     silent
    

    Yes
    


```python
def solve():
    arr = list(map(int, input().strip().split()))
    n = len(arr)
    total = n * (n + 1) // 2
    print(total - sum(arr))
solve()

```

     3 0 1
    

    2
    


```python
def solve():
    from collections import defaultdict
    def dfs(v):
        visited[v] = True
        recStack[v] = True
        for neighbor in graph[v]:
            if not visited[neighbor] and dfs(neighbor):
                return True
            elif recStack[neighbor]:
                return True
        recStack[v] = False
        return False

    n, e = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(e):
        u, v = map(int, input().split())
        graph[u].append(v)

    visited = [False]*n
    recStack = [False]*n
    for node in range(n):
        if not visited[node]:
            if dfs(node):
                print("Yes")
                return
    print("No")
solve()

```

     4 4
     0 1
     1 2
     2 3
     3 1
    

    Yes
    


```python
def solve():
    n = int(input())
    print(bin(n).count('1'))
solve()

```

     7
    

    3
    


```python
def solve():
    s = input().strip()
    from collections import Counter
    count = Counter(s)
    for ch in s:
        if count[ch] == 1:
            print(ch)
            return
    print("None")
solve()

```

     a b c a b d d
    

    c
    


```python
def solve():
    n = int(input())
    print("Yes" if n > 0 and (n & (n - 1)) == 0 else "No")
solve()

```

     16
    

    Yes
    


```python
def solve():
    import sys
    sys.setrecursionlimit(10000)
    def is_balanced(root):
        if not root:
            return 0, True
        lh, lb = is_balanced(root[1])
        rh, rb = is_balanced(root[2])
        balanced = lb and rb and abs(lh - rh) <= 1
        return max(lh, rh) + 1, balanced

    # Input format: (val, left_subtree, right_subtree) or None
    tree = eval(input())  
    print("Yes" if is_balanced(tree)[1] else "No")
solve()

```

     (1, (2, None, None), (3, None, None))
    

    Yes
    


```python
def solve():
    n = int(input())
    intervals = [tuple(map(int, input().split())) for _ in range(n)]
    intervals.sort(key=lambda x: x[1])
    count, end = 0, 0
    for s, e in intervals:
        if s >= end:
            count += 1
            end = e
    print(count)
solve()

```

     3
     1 3
     2 5
     4 7
    

    2
    


```python
def solve():
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def solve_nq(n):
        def backtrack(row=0):
            if row == n:
                result.append(board[:])
                return
            for col in range(n):
                if is_safe(board, row, col):
                    board[row] = col
                    backtrack(row + 1)
        board = [-1] * n
        result = []
        backtrack()
        return result

    n = int(input())
    solutions = solve_nq(n)
    print(len(solutions))
solve()

```

     4
    

    2
    


```python
def solve():
    coins = list(map(int, input().split()))
    amount = int(input())
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] = min(dp[i], dp[i - c] + 1)
    print(dp[amount] if dp[amount] != float('inf') else -1)
solve()

```

     1 2 5
     11
    

    3
    


```python
def solve():
    s = input().strip()
    word_dict = set(input().strip().split())
    dp = [False] * (len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break
    print("Yes" if dp[-1] else "No")
solve()

```

     leetcode
     leet code
    

    Yes
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

```


```python
df = pd.read_csv("lung_cancer.csv")  # make sure this file is in your JupyterLab folder
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Surname</th>
      <th>Age</th>
      <th>Smokes</th>
      <th>AreaQ</th>
      <th>Alkhol</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Wick</td>
      <td>35</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John</td>
      <td>Constantine</td>
      <td>27</td>
      <td>20</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Camela</td>
      <td>Anderson</td>
      <td>30</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alex</td>
      <td>Telles</td>
      <td>28</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diego</td>
      <td>Maradona</td>
      <td>68</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
!pip install matplotlib seaborn

```

    Collecting matplotlib
      Downloading matplotlib-3.10.3-cp312-cp312-win_amd64.whl.metadata (11 kB)
    Collecting seaborn
      Using cached seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting contourpy>=1.0.1 (from matplotlib)
      Downloading contourpy-1.3.2-cp312-cp312-win_amd64.whl.metadata (5.5 kB)
    Collecting cycler>=0.10 (from matplotlib)
      Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting fonttools>=4.22.0 (from matplotlib)
      Downloading fonttools-4.58.4-cp312-cp312-win_amd64.whl.metadata (108 kB)
    Collecting kiwisolver>=1.3.1 (from matplotlib)
      Downloading kiwisolver-1.4.8-cp312-cp312-win_amd64.whl.metadata (6.3 kB)
    Requirement already satisfied: numpy>=1.23 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from matplotlib) (2.3.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from matplotlib) (25.0)
    Collecting pillow>=8 (from matplotlib)
      Downloading pillow-11.2.1-cp312-cp312-win_amd64.whl.metadata (9.1 kB)
    Collecting pyparsing>=2.3.1 (from matplotlib)
      Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: pandas>=1.2 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from seaborn) (2.3.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: six>=1.5 in c:\users\stefi\miniconda3\envs\py312\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
    Downloading matplotlib-3.10.3-cp312-cp312-win_amd64.whl (8.1 MB)
       ---------------------------------------- 0.0/8.1 MB ? eta -:--:--
       - -------------------------------------- 0.3/8.1 MB ? eta -:--:--
       --- ------------------------------------ 0.8/8.1 MB 2.1 MB/s eta 0:00:04
       ------ --------------------------------- 1.3/8.1 MB 2.4 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       ------- -------------------------------- 1.6/8.1 MB 2.5 MB/s eta 0:00:03
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       --------- ------------------------------ 1.8/8.1 MB 477.1 kB/s eta 0:00:14
       ---------- ----------------------------- 2.1/8.1 MB 336.5 kB/s eta 0:00:18
       ---------- ----------------------------- 2.1/8.1 MB 336.5 kB/s eta 0:00:18
       ----------- ---------------------------- 2.4/8.1 MB 362.7 kB/s eta 0:00:16
       -------------- ------------------------- 2.9/8.1 MB 432.4 kB/s eta 0:00:12
       --------------- ------------------------ 3.1/8.1 MB 464.8 kB/s eta 0:00:11
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       ------------------ --------------------- 3.7/8.1 MB 526.8 kB/s eta 0:00:09
       -------------------- ------------------- 4.2/8.1 MB 247.2 kB/s eta 0:00:16
       -------------------- ------------------- 4.2/8.1 MB 247.2 kB/s eta 0:00:16
       ---------------------- ----------------- 4.5/8.1 MB 256.6 kB/s eta 0:00:15
       ----------------------- ---------------- 4.7/8.1 MB 268.6 kB/s eta 0:00:13
       ------------------------ --------------- 5.0/8.1 MB 280.4 kB/s eta 0:00:12
       ------------------------- -------------- 5.2/8.1 MB 293.2 kB/s eta 0:00:10
       --------------------------- ------------ 5.5/8.1 MB 304.5 kB/s eta 0:00:09
       ----------------------------- ---------- 6.0/8.1 MB 329.6 kB/s eta 0:00:07
       ------------------------------- -------- 6.3/8.1 MB 342.4 kB/s eta 0:00:06
       ----------------------------------- ---- 7.1/8.1 MB 380.6 kB/s eta 0:00:03
       ------------------------------------- -- 7.6/8.1 MB 405.3 kB/s eta 0:00:02
       ---------------------------------------- 8.1/8.1 MB 427.7 kB/s eta 0:00:00
    Using cached seaborn-0.13.2-py3-none-any.whl (294 kB)
    Downloading contourpy-1.3.2-cp312-cp312-win_amd64.whl (223 kB)
    Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Downloading fonttools-4.58.4-cp312-cp312-win_amd64.whl (2.2 MB)
       ---------------------------------------- 0.0/2.2 MB ? eta -:--:--
       --------- ------------------------------ 0.5/2.2 MB 3.4 MB/s eta 0:00:01
       ----------------------- ---------------- 1.3/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       ---------------------------- ----------- 1.6/2.2 MB 3.4 MB/s eta 0:00:01
       -------------------------------- ------- 1.8/2.2 MB 521.5 kB/s eta 0:00:01
       ---------------------------------------- 2.2/2.2 MB 619.5 kB/s eta 0:00:00
    Downloading kiwisolver-1.4.8-cp312-cp312-win_amd64.whl (71 kB)
    Downloading pillow-11.2.1-cp312-cp312-win_amd64.whl (2.7 MB)
       ---------------------------------------- 0.0/2.7 MB ? eta -:--:--
       ------- -------------------------------- 0.5/2.7 MB 3.4 MB/s eta 0:00:01
       ----------------------- ---------------- 1.6/2.7 MB 4.0 MB/s eta 0:00:01
       ------------------------------- -------- 2.1/2.7 MB 3.8 MB/s eta 0:00:01
       ---------------------------------------- 2.7/2.7 MB 3.5 MB/s eta 0:00:00
    Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
    Installing collected packages: pyparsing, pillow, kiwisolver, fonttools, cycler, contourpy, matplotlib, seaborn
    
       ---------------------------------------- 0/8 [pyparsing]
       ----- ---------------------------------- 1/8 [pillow]
       ----- ---------------------------------- 1/8 [pillow]
       ----- ---------------------------------- 1/8 [pillow]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       --------------- ------------------------ 3/8 [fonttools]
       -------------------- ------------------- 4/8 [cycler]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ------------------------------ --------- 6/8 [matplotlib]
       ----------------------------------- ---- 7/8 [seaborn]
       ----------------------------------- ---- 7/8 [seaborn]
       ---------------------------------------- 8/8 [seaborn]
    
    Successfully installed contourpy-1.3.2 cycler-0.12.1 fonttools-4.58.4 kiwisolver-1.4.8 matplotlib-3.10.3 pillow-11.2.1 pyparsing-3.2.3 seaborn-0.13.2
    


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("lung_cancer.csv")  
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
df.replace({'YES': 1, 'NO': 0, 'M': 1, 'F': 0}, inplace=True)
df.dropna(inplace=True)
X = df.drop(["RESULT", "NAME", "SURNAME"], axis=1)
y = df["RESULT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
acc = accuracy_score(y_test, y_pred)
prec = report['1']['precision']
rec = report['1']['recall']
f1 = report['1']['f1-score']
print("Performance Points ")
print(f"Accuracy Point : {acc*100:.2f}")
print(f"Precision Point: {prec*100:.2f}")
print(f"Recall Point   : {rec*100:.2f}")
print(f"F1-score Point : {f1*100:.2f}")
plt.figure(figsize=(5, 3))
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

    Performance Points 
    Accuracy Point : 94.44
    Precision Point: 100.00
    Recall Point   : 85.71
    F1-score Point : 92.31
    


    
![png](/pynotes/images/test_1_22_1.png)
    



```python

```


---
**Score: 20**