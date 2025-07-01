---
title: Basic5
date: 2025-07-01
author: Your Name
cell_count: 22
score: 20
---

```python

with open("sample.txt", "w") as file:
    file.write("Hello, Stefina!\nThis is a sample file created by Python.\nWelcome to file handling!")
with open("sample.txt", "r") as file:
    content = file.read()
    print("File Content:\n")
    print(content)

```

    File Content:
    
    Hello, Stefina!
    This is a sample file created by Python.
    Welcome to file handling!
    


```python
with open("output.txt", "w") as file:
    file.write(input("Enter text to write: "))

```

    Enter text to write:  AI Engineer
    


```python
with open("output.txt", "a") as file:
    file.write("\n" + input("Enter text to append: "))

```

    Enter text to append:  passionate in Deep leaning
    


```python
with open("sample.txt", "r") as file:
    lines = file.readlines()
    print("Line count:", len(lines))

```

    Line count: 3
    


```python
with open("sample.txt", "r") as file:
    words = file.read().split()
    print("Word count:", len(words))

```

    Word count: 14
    


```python
with open("sample.txt", "r") as file:
    text = file.read()
    print("Character count:", len(text))

```

    Character count: 82
    


```python
with open("sample.txt", "r") as file:
    words = file.read().split()
    print("Longest word:", max(words, key=len))

```

    Longest word: handling!
    


```python
with open("sample.txt", "r") as src, open("copy.txt", "w") as dest:
    dest.write(src.read())

```


```python
import os

filename = "sample.txt"
print("File exists" if os.path.exists(filename) else "File does not exist")

```

    File exists
    


```python
import os

filename = "newfile.txt"
if not os.path.exists(filename):
    open(filename, "w").close()
    print("File created")
else:
    print("File already exists")

```

    File created
    


```python

with open("oldname.txt", "w") as file:
    file.write("This is the file you will rename.")

```


```python
import os

if os.path.exists("oldname.txt"):
    os.rename("oldname.txt", "newname.txt")
    print("File renamed")
else:
    print("File 'oldname.txt' not found.")

```

    File renamed
    


```python
with open("sample.txt", "r") as file:
    text = file.read().lower()
    vowels = "aeiou"
    count = sum(1 for ch in text if ch in vowels)
    print("Vowel count:", count)

```

    Vowel count: 24
    


```python
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return a / b

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
op = input("Enter operation (+, -, *, /): ")

if op == '+':
    print(add(a, b))
elif op == '-':
    print(sub(a, b))
elif op == '*':
    print(mul(a, b))
elif op == '/':
    print(div(a, b))
else:
    print("Invalid operator")

```

    Enter first number:  20
    Enter second number:  30
    Enter operation (+, -, *, /):  /
    

    0.6666666666666666
    


```python
todo = []

while True:
    task = input("Enter task (or 'done' to exit): ")
    if task.lower() == 'done':
        break
    todo.append(task)

print("Your To-Do List:")
for i, task in enumerate(todo, 1):
    print(f"{i}. {task}")

```

    Enter task (or 'done' to exit):  23
    Enter task (or 'done' to exit):  20
    Enter task (or 'done' to exit):  22
    Enter task (or 'done' to exit):  done
    

    Your To-Do List:
    1. 23
    2. 20
    3. 22
    


```python

with open("numbers.txt", "w") as file:
    file.write("10 20 30 40")  
with open("numbers.txt", "r") as file:
    numbers = map(int, file.read().split())
    print("Sum:", sum(numbers))

```

    Sum: 100
    


```python

with open("numbers.txt", "w") as file:
    file.write("5 15 25 35") 
try:
    with open("numbers.txt", "r") as file:
        numbers = map(int, file.read().split())
        print("Sum:", sum(numbers))
except FileNotFoundError:
    print("Error: 'numbers.txt' file not found.")
except ValueError:
    print("Error: The file contains non-integer values.")

```

    Sum: 80
    


```python
import keyword

with open("sample.txt", "r") as file:
    text = file.read()
    words = text.split()
    count = sum(1 for word in words if word in keyword.kwlist)
    print("Keyword count:", count)

```

    Keyword count: 1
    


```python
import random

words = ["python", "hangman", "code"]
word = random.choice(words)
guesses = ''
turns = 6

while turns > 0:
    failed = 0
    for ch in word:
        if ch in guesses:
            print(ch, end=" ")
        else:
            print("_", end=" ")
            failed += 1
    print()
    if failed == 0:
        print("You Win!")
        break
    guess = input("Guess a letter: ")
    guesses += guess
    if guess not in word:
        turns -= 1
        print("Wrong! Attempts left:", turns)
        if turns == 0:
            print("You Lose! Word was:", word)

```

    _ _ _ _ _ _ _ 
    

    Guess a letter:  c
    

    Wrong! Attempts left: 5
    _ _ _ _ _ _ _ 
    

    Guess a letter:  p
    

    Wrong! Attempts left: 4
    _ _ _ _ _ _ _ 
    

    Guess a letter:  g
    

    _ _ _ g _ _ _ 
    

    Guess a letter:  h 
    

    Wrong! Attempts left: 3
    h _ _ g _ _ _ 
    

    Guess a letter:  a
    

    h a _ g _ a _ 
    

    Guess a letter:  n
    

    h a n g _ a n 
    

    Guess a letter:  m
    

    h a n g m a n 
    You Win!
    


```python
import random

num = random.randint(1, 100)
while True:
    guess = int(input("Guess the number (1-100): "))
    if guess < num:
        print("Too low!")
    elif guess > num:
        print("Too high!")
    else:
        print("Correct! You guessed it.")
        break

```

    Guess the number (1-100):  50
    

    Too low!
    

    Guess the number (1-100):  60
    

    Too low!
    

    Guess the number (1-100):  70
    

    Too low!
    

    Guess the number (1-100):  80
    

    Too low!
    

    Guess the number (1-100):  90
    

    Too high!
    

    Guess the number (1-100):  85
    

    Too high!
    

    Guess the number (1-100):  81
    

    Too low!
    

    Guess the number (1-100):  82
    

    Too low!
    

    Guess the number (1-100):  83
    

    Too low!
    

    Guess the number (1-100):  84
    

    Correct! You guessed it.
    


```python
import random

choices = ['rock', 'paper', 'scissors']
player = input("Choose rock, paper or scissors: ").lower()
computer = random.choice(choices)

print("Computer chose:", computer)

if player == computer:
    print("It's a tie!")
elif (player == 'rock' and computer == 'scissors') or \
     (player == 'scissors' and computer == 'paper') or \
     (player == 'paper' and computer == 'rock'):
    print("You win!")
else:
    print("You lose!")

```

    Choose rock, paper or scissors:  paper
    

    Computer chose: rock
    You win!
    


```python

```


---
**Score: 20**