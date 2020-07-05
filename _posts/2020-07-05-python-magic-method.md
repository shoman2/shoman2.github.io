---
layout: post
title:  "Python Magic Method 사용법"
subtitle: "Python Magic Method 사용법"
categories: development
tags: python
comments: true
---

- Python Magic Method 사용법에 대해 작성한 글입니다
- 키워드 : Python Magic Method, Python Special Method
	
	
	
---


### 매직 메소드
- Magic Method, Special Method라고도 불림
- 파이썬이 내부적으로 구현된(빌트인) 메소드를 구현함
	- 이미 만들어짐
- 더블 언더스코어를 사용한 것들이 Magic Method
- [데이터 모델](https://docs.python.org/ko/3.7/reference/datamodel.html#special-method-names)을 보면 자세히 나와있음

- int, float의 메소드를 확인해보자

```
# 기본형
print(int)
print(float)

# 모든 속성 및 메소드 출력
print(dir(int))
print(dir(float))
```

- dir 출력 결과를 보면, 더블 언더스코어로 시작하는 많은 메소드를 볼 수 있음
- `__new__`
	- 새로운 인스턴스를 만들 때 제일 처음으로 실행되는 메소드
	- 새로운 object를 return
- `__init__`
	- 인스턴스가 `__new__`로 생성된 후, 호출되는 메소드
- `__add__`
	- 더하는 메소드
	- 아래 두 결과는 동일함
	- 사용하기 쉽게 +에 `__add__`에 매핑되었다고 생각하면 됨
	
	
	```
	print(n+10)
	print(n.__add__(10))
	print(n.__doc__)
	print(n.__bool__(), bool(n))
	```
	
- `__doc__`
	- Docstring을 출력하는 메소드
- `__bool__`
	- Boolean 유무를 나타내는 메소드
	- bool(n)과 동일한 표현
- `__mul__`
	- 곱하는 메소드
	- \*에 매핑됨
- `__sub__`
	- 빼는 메소드
- `__le__`
	- 작거나 같다
	- little or equal(<=) 표현
- `__ge__`
	- 크거나 같다
	- greater or equal(>=) 표현
- `__del__`
	- 객체를 없어질 때 호출되는 메소드
	- 내부에 있는 레퍼런스 카운터가 0이 되면 삭제됨
- `__bytes__`
	- 객체를 나타내는 byte 문자열
- `__format__`
	- 객체를 나타내는 format을 지정하고 싶을 때 사용
- `__len__`
	- 객체의 길이를 반환
- `__iter__`
	- 컨테이너의 iterator를 반환
- `__reversed__`
	- 순서가 반대로 바뀌는 reversed() 함수 호출
- `__contains__`
	- item 인자를 받고, 존재하면 True, 아니면 False
	- `__contains__`가 정의도지 않으면 `__iter__`를 통해 이터레이션을 돌며 확인함


- 예시 Class #1
	- Class끼리 더하는건 쉽지 않음
	- 그러나 우리가 매직 메소드를 구현하면, Class끼리 더하는걸 정의할 수 있음(예를 들어 사람 Instance를 각각 더하면 두 사람의 키가 나오게하는!)
	- add할 경우 커스텀으로 함수를 수정할 수 있음(예를 들어 2개를 더할 경우 벌크 할인 느낌으로 * 0.8!

```
class Notebook:
    def __init__(self, name, price):
        self._name = name
        self._price = price
    
    def __str__(self):
        return f'{self.__class__.name} Class Info : {self._name}, {self._price}'
        
    def __add__(self, x):
        print("Called __add__")
        return (self._price + x._price) * 0.8
        
    def __sub__(self, x):
        print("Called __sub__")
        return self._price - x._price
        
    def __le__(self, x):
        print("Called __le__")
        if self._price <= x._price:
            return True
        else:
            return False

    def __ge__(self, x):
        print("Called __ge__")
        if self._price >= x._price:
            return True
        else:
            return False

# 인스턴스 생성  
xps13 = Notebook('XPS13', 100)
macbook_pro = Notebook('Macbook Pro', 200)

# add method를 구현하지 않았다면 아래처럼(그러지 추천하지 않음)
print(xps13._price + macbook_pro._price)

# add method를 구현했기 때문에 이렇게!
print(xps13 + macbook_pro)

print(xps13 >= macbook_pro)
print(macbook_pro - xps_13)
```



- 예시 Class #2
	- *args를 사용함
	- Vector.__doc__을 할 경우 아무것도 나오지 않음(작성하지 않았으니)

```
class Vector(object):
    def __init__(self, *args): # unpacking, packing 가정
        '''Create a vector, example : v = Vector(5,10)'''
        if len(args) == 0:
            self._x, self._y = 0, 0
        else:
            self._x, self._y = args

    def __repr__(self):
        '''Returns the vector infomations'''
        return 'Vector(%r, %r)' % (self._x, self._y)

    def __add__(self, other):
        '''Returns the vector addition of self and other'''
        return Vector(self._x + other._x, self._y + other._y)
    
    def __mul__(self, y):
        return Vector(self._x * y, self._y * y)

    def __bool__(self):
        return bool(max(self._x, self._y))
    
# Vector 인스턴스 생성
v1 = Vector(5,7)
v2 = Vector(23, 35)
v3 = Vector()

# 매직 메소드 출력
print(Vector.__init__.__doc__)
print(Vector.__repr__.__doc__)
print(Vector.__add__.__doc__)
print(v1, v2, v3)
print(v1 + v2)
print(v1 * 3)
print(v2 * 10)
print(bool(v1), bool(v2))
print(bool(v3)) 
```

<br />
<br />

---

### 그 외 매직 메소드
- WITH 키워드를 통해 블럭에 진입시, 컨텍스트 매니저와 함께 사용하는 매직 메소드
	- `__enter__`
		- with로 진입할 때 실행하는 메소드
	- `__exit__`
		- 블럭이 끝날 때 실행하는 메소드
- 비동기 관련 메소드
	- `__await__`
		- await 표현을 사용할 수 있는 객체를 만들 때 사용
		- iterator를 반환함
- 그 외에도 매우 많은 매직 메소드가 존재함
	- [A Guide to Python's Magic Methods](https://rszalski.github.io/magicmethods/) 참고!

<br />
<br />

---

### Reference	
- [인프런 파이썬 중급 프로그래밍](https://www.inflearn.com/course/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%A4%91%EA%B8%89-%EC%9D%B8%ED%94%84%EB%9F%B0-%EC%98%A4%EB%A6%AC%EC%A7%80%EB%84%90)
- [Python Class Document](https://docs.python.org/ko/3/tutorial/classes.html)
- [파이썬 더블 언더스코어: Magic Method](https://corikachu.github.io/articles/python/python-magic-method)
- [A Guide to Python's Magic Methods](https://rszalski.github.io/magicmethods/)