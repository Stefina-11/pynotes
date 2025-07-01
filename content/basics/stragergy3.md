---
title: Stragergy3
date: 2025-07-01
author: Your Name
cell_count: 32
score: 30
---

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): pass

```


```python
from abc import ABC, abstractmethod

class Shape(ABC):
    def info(self):
        print("Shape info")

    @abstractmethod
    def area(self): pass

```


```python
from abc import ABC, abstractmethod

class Appliance(ABC):
    def plug_in(self):
        print("Appliance plugged in")

    @abstractmethod
    def operate(self): pass

```


```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self): pass

class Cat(Animal):
    def sound(self):
        print("Meow")

c = Cat()
c.sound()

```

    Meow
    


```python
from abc import ABC, abstractmethod

class Machine(ABC):
    @abstractmethod
    def start(self): pass

    @abstractmethod
    def stop(self): pass

class Fan(Machine):
    def start(self): print("Fan started")
    def stop(self): print("Fan stopped")

```


```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): pass

    @abstractmethod
    def volume(self): pass

```


```python
from abc import ABC, abstractmethod

class Meal(ABC):
    def prepare_meal(self):
        self.cook()
        self.serve()

    @abstractmethod
    def cook(self): pass

    def serve(self):
        print("Serve on plate")

class Pizza(Meal):
    def cook(self):
        print("Baking Pizza")

p = Pizza()
p.prepare_meal()

```

    Baking Pizza
    Serve on plate
    


```python
from abc import ABC, abstractmethod

class Base(ABC):
    def __init__(self):
        print("Base init")

    @abstractmethod
    def display(self): pass

```


```python
from abc import ABC, abstractmethod

class RemoteControl(ABC):
    @abstractmethod
    def power(self): pass

class TV(RemoteControl):
    def power(self):
        print("TV power toggled")

tv = TV()
tv.power()

```

    TV power toggled
    


```python
from abc import ABC, abstractmethod

class Utils(ABC):
    @staticmethod
    @abstractmethod
    def help(): pass

class HelpSystem(Utils):
    @staticmethod
    def help():
        print("This is the help section.")

HelpSystem.help()

```

    This is the help section.
    


```python
class User:
    def __init__(self, name):
        self.name = name

u = User("Stefina")
print(u.name)

```

    Stefina
    


```python
class Product:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Product: {self.name}"

p = Product("Laptop")
print(p)

```

    Product: Laptop
    


```python
class Point:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"Point({self.x})"

p = Point(10)
print(repr(p))

```

    Point(10)
    


```python
class File:
    def __del__(self):
        print("File object deleted")

f = File()
del f

```

    File object deleted
    


```python
class Book:
    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return self.pages

b = Book(250)
print(len(b))

```

    250
    


```python
class MyList:
    def __init__(self, data):
        self.data = data

    def __contains__(self, item):
        return item in self.data

ml = MyList([1, 2, 3])
print(2 in ml)

```

    True
    


```python
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.limit:
            self.num += 1
            return self.num
        else:
            raise StopIteration

for i in Counter(3):
    print(i)

```

    1
    2
    3
    


```python
class Box:
    def __init__(self, weight):
        self.weight = weight

    def __eq__(self, other):
        return self.weight == other.weight

b1 = Box(50)
b2 = Box(50)
print(b1 == b2)

```

    True
    


```python
class ID:
    def __init__(self, num):
        self.num = num

    def __hash__(self):
        return hash(self.num)

id1 = ID(123)
print(hash(id1))

```

    123
    


```python
class Cart:
    def __init__(self, items):
        self.items = items

    def __bool__(self):
        return bool(self.items)

cart = Cart(["item1"])
print(bool(cart))

```

    True
    


```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)

```

    True
    


```python
class Dog:
    def speak(self): return "Woof"

class Cat:
    def speak(self): return "Meow"

def pet_factory(pet="dog"):
    pets = dict(dog=Dog(), cat=Cat())
    return pets[pet]

pet = pet_factory("cat")
print(pet.speak())

```

    Meow
    


```python
class BoldText:
    def __init__(self, text):
        self.text = text

    def render(self):
        return f"<b>{self.text}</b>"

msg = BoldText("Hello")
print(msg.render())

```

    <b>Hello</b>
    


```python
class Publisher:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, sub):
        self.subscribers.append(sub)

    def notify(self):
        for sub in self.subscribers:
            sub.update()

class Subscriber:
    def update(self):
        print("Subscriber updated!")

p = Publisher()
s = Subscriber()
p.subscribe(s)
p.notify()

```

    Subscriber updated!
    


```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount

a = BankAccount("Stefina", 1000)
a.deposit(500)
a.withdraw(200)
print(a.balance)

```

    1300
    


```python
class Product:
    def __init__(self, name, qty):
        self.name = name
        self.qty = qty

class Inventory:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

    def show(self):
        for p in self.products:
            print(p.name, p.qty)

inv = Inventory()
inv.add_product(Product("Pen", 10))
inv.show()

```

    Pen 10
    


```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

class School:
    def __init__(self):
        self.students = []

    def add_student(self, s):
        self.students.append(s)

    def display(self):
        for s in self.students:
            print(s.name, s.grade)

s1 = Student("Ana", 90)
school = School()
school.add_student(s1)
school.display()

```

    Ana 90
    


```python
class Book:
    def __init__(self, title):
        self.title = title

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, b):
        self.books.append(b)

    def list_books(self):
        for b in self.books:
            print(b.title)

lib = Library()
lib.add_book(Book("Python 101"))
lib.list_books()

```

    Python 101
    


```python
class Patient:
    def __init__(self, name, disease):
        self.name = name
        self.disease = disease

class Hospital:
    def __init__(self):
        self.patients = []

    def admit(self, p):
        self.patients.append(p)

h = Hospital()
h.admit(Patient("John", "Flu"))
for p in h.patients:
    print(p.name, p.disease)

```

    John Flu
    


```python
class ATM:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrawn: {amount}")
        else:
            print("Insufficient balance")

atm = ATM(1000)
atm.withdraw(300)
print("Balance:", atm.balance)

```

    Withdrawn: 300
    Balance: 700
    


```python
fruits = ["apple", "banana", "cherry"]
print(fruits)
print(fruits[1])
fruits[0] = "mango"
fruits.append("orange")
fruits.insert(1, "grape")
fruits.remove("banana")
fruits.pop(1)
print(len(fruits))
for f in fruits:
    print(f)
print("apple" in fruits)
nums = [3, 5, 1, 4]
nums.sort()
nums.reverse()
copy_list = fruits.copy()
print(fruits + nums)
squares = [x*x for x in range(6)]
even = [x for x in range(10) if x % 2 == 0]
matrix = [[1, 2], [3, 4]]
print(matrix[1][0])
print(fruits.index("orange"))
nums = [1, 2, 2, 3]
print(nums.count(2))
fruits.clear()

colors = ("red", "green", "blue")
print(colors[0])
print(len(colors))
for c in colors:
    print(c)
temp = list(colors)
a, b, c = colors
nested = ((1, 2), (3, 4))
print(nested[1][0])
print(colors[1:])
print("green" in colors)
print(colors + ("yellow",))

s = {1, 2, 3}
s.add(4)
s.remove(2)
s.discard(5)
a = {1, 2}
b = {2, 3}
print(a.union(b))
print(a.intersection(b))
print(a.difference(b))
print(a.symmetric_difference(b))
print({1}.issubset(a))
print(a.issuperset({1}))
a.clear()
print(len(s))
for item in s:
    print(item)
frozen = frozenset([1, 2, 3])
unique = set([1, 2, 2, 3])

student = {"name": "John", "age": 20}
print(student["name"])
print(student.get("grade", "Not available"))
student["grade"] = "A"
student["age"] = 21
print(list(student.keys()))
print(list(student.values()))
print(list(student.items()))
for k, v in student.items():
    print(k, v)
print("name" in student)
student.pop("grade")
del student["age"]
student.clear()
copy_dict = student.copy()
d1 = {"a": 1}
d2 = {"b": 2}
d1.update(d2)
data = {"emp": {"name": "Ana", "id": 101}}
print(data["emp"]["name"])
squares = {x: x*x for x in range(5)}
print(len(d1))
print(student.setdefault("name", "Unknown"))
student.clear()

stack = []
stack.append(1)
stack.append(2)
stack.pop()
queue = [1, 2, 3]
queue.pop(0)
from collections import deque
stack = deque()
stack.append(5)
stack.pop()
queue = deque()
queue.append(10)
queue.popleft()
print(stack[-1] if stack else "Empty")
print(len(stack) == 0)
q = deque([1, 2, 3])
q.reverse()
class Stack:
    def __init__(self): self.items = []
    def push(self, item): self.items.append(item)
    def pop(self): return self.items.pop()
s = Stack()
s.push(10)
print(s.pop())
class Queue:
    def __init__(self): self.items = deque()
    def enqueue(self, item): self.items.append(item)
    def dequeue(self): return self.items.popleft()
q = Queue()
q.enqueue(1)
print(q.dequeue())
import heapq
pq = []
heapq.heappush(pq, 2)
heapq.heappush(pq, 1)
print(heapq.heappop(pq))
people = deque(["A", "B", "C"])
people.rotate(-1)
print(people)
expr = "(()())"
stack = []
balanced = True
for char in expr:
    if char == "(":
        stack.append(char)
    elif char == ")":
        if not stack:
            balanced = False
            break
        stack.pop()
print("Balanced" if balanced and not stack else "Unbalanced")
print(len(queue))

s = "hello"
print(s[::-1])
print(s == s[::-1])
print(''.join([ch for ch in s if ch.lower() not in 'aeiou']))
from collections import Counter
print(Counter(s))
print(len(set(s)) == len(s))
sentence = "hi there how are you"
words = sentence.split()
print('-'.join(words))
print(max(words, key=len))
a, b = "listen", "silent"
print(sorted(a) == sorted(b))
print(s.replace("l", "*"))

users = [{"id": 1}, {"id": 2}]
print(users[1]["id"])
grades = {"Math": [90, 85]}
print(grades["Math"][0])
d = {(1, 2): "value"}
print(d[(1, 2)])
mat = [[1, 2], [3, 4]]
print(mat[0][1])
class Person:
    def __init__(self, name): self.name = name
people = [Person("Sam")]
print(people[0].name)
objs = {"p1": Person("Lily")}
print(objs["p1"].name)
coords = {(0, 0), (1, 1)}
print((1, 1) in coords)
fs = {frozenset([1, 2]), frozenset([3, 4])}
print(fs)
keys = ['a', 'b']
vals = [1, 2]
print(dict(zip(keys, vals)))
nested = [[1, 2], [3, 4]]
flat = [num for sublist in nested for num in sublist]
print(flat)

```

    ['apple', 'banana', 'cherry']
    banana
    3
    mango
    cherry
    orange
    False
    ['mango', 'cherry', 'orange', 5, 4, 3, 1]
    3
    2
    2
    red
    3
    red
    green
    blue
    3
    ('green', 'blue')
    True
    ('red', 'green', 'blue', 'yellow')
    {1, 2, 3}
    {2}
    {1}
    {1, 3}
    True
    True
    3
    1
    3
    4
    John
    Not available
    ['name', 'age', 'grade']
    ['John', 21, 'A']
    [('name', 'John'), ('age', 21), ('grade', 'A')]
    name John
    age 21
    grade A
    True
    Ana
    2
    Unknown
    Empty
    True
    10
    1
    1
    deque(['B', 'C', 'A'])
    Balanced
    0
    olleh
    False
    hll
    Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
    False
    hi-there-how-are-you
    there
    True
    he**o
    2
    90
    value
    2
    Sam
    Lily
    True
    {frozenset({3, 4}), frozenset({1, 2})}
    {'a': 1, 'b': 2}
    [1, 2, 3, 4]
    


```python

```


---
**Score: 30**