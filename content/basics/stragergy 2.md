---
title: Stragergy 2
date: 2025-07-01
author: Your Name
cell_count: 70
score: 70
---

```python
class Person:
    def greet(self):
        print("Hello, I am a person!")

p = Person()
p.greet()

```

    Hello, I am a person!
    


```python

class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hi, I'm {self.name}")

p = Person("Alice")
p.greet()

```

    Hi, I'm Alice
    


```python
class Dog:
    species = "Canine"

    def __init__(self, name):
        self.name = name

d1 = Dog("Buddy")
d2 = Dog("Max")
print(d1.name, d1.species)
print(d2.name, d2.species)

```

    Buddy Canine
    Max Canine
    


```python
class Calculator:
    def add(self, a, b):
        return a + b

c = Calculator()
print(c.add(5, 3))

```

    8
    


```python

class Person:
    count = 0

    def __init__(self):
        Person.count += 1

    @classmethod
    def get_count(cls):
        return cls.count

p1 = Person()
p2 = Person()
print(Person.get_count())

```

    2
    


```python
class Math:
    @staticmethod
    def square(x):
        return x * x

print(Math.square(4))

```

    16
    


```python
class Secret:
    def __init__(self):
        self.__data = "Hidden"

    def reveal(self):
        return self.__data

s = Secret()
print(s.reveal())

```

    Hidden
    


```python
class Student:
    def __init__(self, marks):
        self._marks = marks

    @property
    def marks(self):
        return self._marks

    @marks.setter
    def marks(self, value):
        if 0 <= value <= 100:
            self._marks = value

s = Student(85)
print(s.marks)
s.marks = 95
print(s.marks)

```

    85
    95
    


```python
class Sample:
    """This is a sample class with a docstring."""
    pass

print(Sample.__doc__)

```

    This is a sample class with a docstring.
    


```python
class Car:
    def __init__(self, brand="Toyota"):
        self.brand = brand

c1 = Car()
c2 = Car("BMW")
print(c1.brand, c2.brand)

```

    Toyota BMW
    


```python
class Box:
    def __init__(self, volume):
        self.volume = volume

    def is_larger(self, other):
        return self.volume > other.volume

b1 = Box(10)
b2 = Box(7)
print(b1.is_larger(b2))

```

    True
    


```python
class Counter:
    count = 0

    def __init__(self):
        Counter.count += 1

c1 = Counter()
c2 = Counter()
print(Counter.count)

```

    2
    


```python
import copy

class Item:
    def __init__(self, name):
        self.name = name

item1 = Item("Pen")
item2 = copy.copy(item1)
print(item1.name, item2.name)

```

    Pen Pen
    


```python
class User:
    pass

u = User()
u.name = "John"
print(u.name)

```

    John
    


```python
class Fruit:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Fruit: {self.name}"

    def __repr__(self):
        return f"Fruit({self.name!r})"

f = Fruit("Apple")
print(str(f))
print(repr(f))

```

    Fruit: Apple
    Fruit('Apple')
    


```python
class Demo:
    def __init__(self):
        self.x = 10

d = Demo()
print(d.x)
del d.x
# print(d.x)  # This will raise AttributeError

```

    10
    


```python
a = [1, 2]
b = a
print(a is b)  # True, both refer to the same object

```

    True
    


```python
class Sample:
    def __init__(self, val):
        self.val = val

s = Sample(5)
print(s.__dict__)

```

    {'val': 5}
    


```python
class Data:
    def __init__(self, value):
        self.value = value

def show(obj):
    print(obj.value)

d = Data(99)
show(d)

```

    99
    


```python
class Animal:
    def __init__(self, name):
        self.name = name

def get_animal():
    return Animal("Tiger")

a = get_animal()
print(a.name)

```

    Tiger
    


```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    pass

d = Dog()
d.speak()

```

    Animal speaks
    


```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal(Animal):
    def walk(self):
        print("Mammal walks")

class Dog(Mammal):
    def bark(self):
        print("Dog barks")

d = Dog()
d.speak()
d.walk()
d.bark()

```

    Animal speaks
    Mammal walks
    Dog barks
    


```python
class Father:
    def skills(self):
        print("Gardening, Programming")

class Mother:
    def skills(self):
        print("Cooking, Art")

class Child(Father, Mother):
    pass

c = Child()
c.skills()  # MRO: Father’s method is called first

```

    Gardening, Programming
    


```python
class Parent:
    def __init__(self):
        print("Parent constructor")

class Child(Parent):
    def __init__(self):
        super().__init__()
        print("Child constructor")

c = Child()

```

    Parent constructor
    Child constructor
    


```python
class Parent:
    def show(self):
        print("Parent class")

class Child(Parent):
    def show(self):
        print("Child class")

c = Child()
c.show()

```

    Child class
    


```python
class A:
    def __init__(self):
        print("Constructor A")

class B(A):
    def __init__(self):
        super().__init__()
        print("Constructor B")

b = B()

```

    Constructor A
    Constructor B
    


```python
class A:
    value = 10

class B(A):
    pass

print(B.value)

```

    10
    


```python
class A:
    def method(self):
        print("Class A")

class B(A):
    def method(self):
        print("Class B")

class C(A):
    def method(self):
        print("Class C")

class D(B, C):
    pass

d = D()
d.method()

```

    Class B
    


```python
class Parent:
    def show(self):
        print("Parent class")

class Child1(Parent):
    pass

class Child2(Parent):
    pass

c1 = Child1()
c2 = Child2()
c1.show()
c2.show()

```

    Parent class
    Parent class
    


```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):
    pass

d = D()
d.show()  # B is prioritized before C in MRO

```

    B
    


```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.mro())

```

    [<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
    


```python
class Animal: pass
class Dog(Animal): pass

print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False

```

    True
    False
    


```python
class Car: pass
c = Car()
print(isinstance(c, Car))

```

    True
    


```python
class A:
    def __init__(self):
        print("A constructor")

class B(A):
    def __init__(self):
        print("B constructor")

b = B()  # A’s constructor is not called unless super() used

```

    B constructor
    


```python
class Parent:
    def __init__(self):
        self.__secret = "Hidden"

class Child(Parent):
    def reveal(self):
    
        pass

c = Child()
c.reveal()

```


```python
class A:
    def __init__(self):
        print("Init A")

class B(A):
    def __init__(self):
        super().__init__()
        print("Init B")

b = B()

```

    Init A
    Init B
    


```python
class A:
    @staticmethod
    def greet():
        print("Hello from A")

    @classmethod
    def identity(cls):
        print(f"I am {cls.__name__}")

class B(A): pass

B.greet()
B.identity()

```

    Hello from A
    I am B
    


```python
class Logger:
    def log(self, message):
        print(f"[LOG]: {message}")

class App(Logger):
    def run(self):
        self.log("App is running")

a = App()
a.run()

```

    [LOG]: App is running
    


```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass

class Dog(Animal):
    def sound(self):
        print("Bark")

d = Dog()
d.sound()

```

    Bark
    


```python
class Shape:
    def area(self):
        print("Area not defined")

class Circle(Shape):
    def area(self):
        print("Area of Circle")

class Square(Shape):
    def area(self):
        print("Area of Square")

shapes = [Circle(), Square()]
for shape in shapes:
    shape.area()

```

    Area of Circle
    Area of Square
    


```python
class Calculator:
    def add(self, a, b=0, c=0):
        return a + b + c

calc = Calculator()
print(calc.add(5))
print(calc.add(5, 10))
print(calc.add(5, 10, 15))

```

    5
    15
    30
    


```python
class Point:
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        return Point(self.x + other.x)

    def __str__(self):
        return f"Point({self.x})"

p1 = Point(10)
p2 = Point(20)
print(p1 + p2)

```

    Point(30)
    


```python
from functools import singledispatch

@singledispatch
def process(value):
    print("Default:", value)

@process.register(int)
def _(value):
    print("Integer:", value)

@process.register(str)
def _(value):
    print("String:", value)

process(10)
process("hello")

```

    Integer: 10
    String: hello
    


```python
class Animal:
    def speak(self):
        print("Animal sound")

class Dog(Animal):
    def speak(self):
        print("Bark")

d = Dog()
d.speak()

```

    Bark
    


```python

class Bird:
    def fly(self):
        print("Bird is flying")

class Airplane:
    def fly(self):
        print("Airplane is flying")

def lift_off(entity):
    entity.fly()

b = Bird()
a = Airplane()
lift_off(b)
lift_off(a)

```

    Bird is flying
    Airplane is flying
    


```python
class Animal:
    def make_sound(self):
        print("Some sound")

class Cat(Animal):
    def make_sound(self):
        print("Meow")

class Dog(Animal):
    def make_sound(self):
        print("Bark")

def sound(animal):
    animal.make_sound()

sound(Cat())
sound(Dog())

```

    Meow
    Bark
    


```python
class Circle:
    def draw(self):
        print("Drawing Circle")

class Square:
    def draw(self):
        print("Drawing Square")

shapes = [Circle(), Square()]
for shape in shapes:
    shape.draw()

```

    Drawing Circle
    Drawing Square
    


```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start(self):
        pass

class Car(Vehicle):
    def start(self):
        print("Car started")

v = Car()
v.start()

```

    Car started
    


```python
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self): pass

class Circle(Drawable):
    def draw(self):
        print("Drawing Circle")

c = Circle()
c.draw()

```

    Drawing Circle
    


```python
from abc import ABC, abstractmethod

class Machine(ABC):
    @abstractmethod
    def start(self): pass

    @abstractmethod
    def stop(self): pass

class Fan(Machine):
    def start(self):
        print("Fan started")

    def stop(self):
        print("Fan stopped")

f = Fan()
f.start()
f.stop()

```

    Fan started
    Fan stopped
    


```python
class Shape:
    def area(self):
        return 0

class Circle(Shape):
    def area(self):
        return 3.14 * 4 * 4

class Square(Shape):
    def area(self):
        return 4 * 4

shapes = [Circle(), Square()]
for shape in shapes:
    print(shape.area())

```

    50.24
    16
    


```python
print(len("hello"))
print(len([1, 2, 3]))

```

    5
    3
    


```python
class Add:
    def operation(self, a, b): return a + b

class Subtract:
    def operation(self, a, b): return a - b

def calc(op, a, b):
    return op.operation(a, b)

print(calc(Add(), 5, 3))
print(calc(Subtract(), 5, 3))

```

    8
    2
    


```python
class Book:
    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return self.pages

b = Book(120)
print(len(b))

```

    120
    


```python
class Box:
    def __init__(self, weight):
        self.weight = weight

    def __eq__(self, other):
        return self.weight == other.weight

b1 = Box(10)
b2 = Box(10)
print(b1 == b2)

```

    True
    


```python
class Player:
    def __init__(self, score):
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

p1 = Player(50)
p2 = Player(70)
print(p1 < p2)

```

    True
    


```python
class Greet:
    def __call__(self, name):
        print(f"Hello, {name}!")

g = Greet()
g("Stefina")

```

    Hello, Stefina!
    


```python
class MyList:
    def __init__(self):
        self.data = {}

    def __getitem__(self, key):
        return self.data.get(key, None)

    def __setitem__(self, key, value):
        self.data[key] = value

ml = MyList()
ml["a"] = 100
print(ml["a"])

```

    100
    


```python
class Animal:
    def speak(self):
        print("Animal")

class Cat(Animal):
    def speak(self):
        print("Meow")

def speak_animal(a: Animal):
    a.speak()

speak_animal(Cat())

```

    Meow
    


```python
class Payment:
    def pay(self):
        print("Generic Payment")

class CardPayment(Payment):
    def pay(self):
        print("Paid using Card")

class UpiPayment(Payment):
    def pay(self):
        print("Paid using UPI")

for p in [CardPayment(), UpiPayment()]:
    p.pay()

```

    Paid using Card
    Paid using UPI
    


```python
class Student:
    def __init__(self):
        self.name = "Stefina"         
        self.__marks = 95             
s = Student()
print(s.name)

```

    Stefina
    


```python
class Student:
    def __init__(self):
        self.__secret = "Hidden Info"

s = Student()
print(s._Student__secret) 

```

    Hidden Info
    


```python
class BankAccount:
    def __init__(self):
        self.__balance = 0

    def get_balance(self):
        return self.__balance

    def set_balance(self, amount):
        if amount > 0:
            self.__balance = amount

acc = BankAccount()
acc.set_balance(5000)
print(acc.get_balance())

```

    5000
    


```python
class Product:
    def __init__(self, price):
        self._price = price

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, val):
        if val >= 0:
            self._price = val

p = Product(100)
print(p.price)
p.price = 200
print(p.price)

```

    100
    200
    


```python
class Locker:
    def __init__(self):
        self.__pin = "1234"

    def check_pin(self, entered_pin):
        return self.__pin == entered_pin

l = Locker()
print(l.check_pin("1234"))

```

    True
    


```python
class Speed:
    def __init__(self):
        self.__speed = 0

    def increase(self):
        self.__speed += 10

    def show_speed(self):
        print(f"Speed: {self.__speed} km/h")

s = Speed()
s.increase()
s.show_speed()

```

    Speed: 10 km/h
    


```python
class File:
    def __init__(self):
        self._filename = "secret.txt"  # Protected by convention

f = File()
print(f._filename)

```

    secret.txt
    


```python
class Device:
    def _start(self):
        print("Device starting...")

class Phone(Device):
    def boot(self):
        self._start()

p = Phone()
p.boot()

```

    Device starting...
    


```python
class User:
    def __init__(self, password):
        self.__password = self.__encrypt(password)

    def __encrypt(self, pwd):
        return ''.join([chr(ord(c)+1) for c in pwd])

    def get_password(self):
        return self.__password

u = User("abc123")
print(u.get_password())

```

    bcd234
    


```python
class Vehicle:
    def __init__(self):
        self.__engine = "Petrol"

    def get_engine(self):
        return self.__engine

class Bike(Vehicle):
    pass

b = Bike()
print(b.get_engine())

```

    Petrol
    


---
**Score: 70**