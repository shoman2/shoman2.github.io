---
layout: post
title:  "파이썬 중급 - Python Class 예제 및 사용법"
subtitle: "파이썬 중급 - Python Class 예제 및 사용법"
categories: development
tags: python
comments: true
---

- Python Class 예제 및 Python Class 사용법에 대해 작성한 글입니다
- 키워드 : Python Class, Python Class Example, Python Class Use Case
- 목차
	- [객체 지향 프로그래밍](#객체-지향-프로그래밍)
	- [절차 지향 프로그래밍 예시](#절차-지향-프로그래밍-예시)
	- [객체 지향 프로그래밍 살펴보기](#객체-지향-프로그래밍-살펴보기)
	- [메소드의 종류](#메소드의-종류)
	- [상속](#상속)
	- [Setter와 Getter, Property](#setter와-getter-property)
	- [추상 메소드](#추상-메소드)
	- [slots](#slots)

	
---


### 객체 지향 프로그래밍
- OOP
- 코드의 재사용, 코드 중복 방지를 위해 사용함
- 과거엔 규모가 큰 프로젝트는 함수 중심으로 개발함
	- 데이터가 방대해지고 복잡해져서 추후 개선이 어려움
	- 코드 중복, 협업시 문제가 발생하는 등 복잡해짐
- 클래스 중심
	- 데이터 중심 => 객체로 관리
	- 코드의 재사용, 코드 중복 방지, 유지보수, 대형 프로젝트에 사용 가능
- 항상 클래스를 사용하는게 좋다고 말할 순 없으며, 간단한 경우엔 절차 지향으로 프로그래밍해도 좋음(개발하려는 규모에 따라 다름)
- 절차 지향 프로그래밍
	- 물이 위에서 아래로 흐르는 것처럼, 순차적으로 처리되는 프로그래밍

	
### 절차 지향 프로그래밍 예시
- 객체 지향	프로그래밍을 하기 전에, 절차 지향 프로그래밍으로 하면 어떤지 확인하고 객체 지향으로 프로그래밍할 예정

- 일반적인 코딩
	- 하나의 형태를 만들고, Copy And Paste(복사 붙여넣기)로 추가 요소를 생성함
	- 스마트폰이 증가될수록 코드가 길어지게 됨
	- 스마트폰과 스마트폰 정보를 출력하려면 같이 출력해야 함

	```
	
	smartphone_1 = 'Iphone'
	smartphone_detail_1 = [
	    {'color' : 'White'},
	    {'price': 10000}
	]
	
	smartphone_2 = 'Galaxy'
	smartphone_detail_2 = [
	    {'color' : 'Blue'},
	    {'price': 8000}
	]
	
	smartphone_3 = 'Blackberry'
	smartphone_detail_3 = [
	    {'color' : 'Silver'},
	    {'price': 6000}
	]
	
	print(smartphone_1)
	print(smartphone_detail_1)
	```


- List를 사용한 방법
	- 위 방법보다 코드가 줄어듬
	- 변수를 인덱스로 접근 가능
		- 하지만 실수할 수 있음. 순서가 변경되면?
		- 데이터가 많아지면? 아이폰는 인덱스 1213123번이였는데 갤럭시가 삭제되서 인덱스가 1213122로 변했으면 어떻게 알 수 있을까?
	- 한번에 출력할 순 없음
	- 하나만 삭제하기 힘듬
		- smartphone\_list의 1번을 삭제하고, smartphone\_detail\_list의 1번도 삭제해줘야 함
		- 혹은 함수를 구현해야 함

	```
	smartphone_list = ['Iphone', 'Galaxy', 'Blackberry']
	smartphone_detail_list = [
	    {'color' : 'White', 'price': 10000},
	    {'color' : 'Blue', 'price': 8000},
	    {'color' : 'Silver', 'price': 6000}
	]
	
	del smartphone_list[1]
	del smartphone_detail_list[1]
	
	print(smartphone_list)
	print(smartphone_detail_list)
	```
	
	
- Dictionary를 사용한 방법
	- 중첩 문제(Key는 중복 X)
	- 정렬 문제(OrderedDict을 사용할 수 있음)
	- 키 조회 예외 처리를 생각해야 함
	- Dict 안의 값에 접근할 때는 ["key"]로 접근

	```
	smartphone_dicts = [
	    {'brand': 'Iphone', 'smartphone_detail': {'color' : 'White', 'price': 10000}},
	    {'brand': 'Galaxy', 'smartphone_detail': {'color' : 'Blue', 'price': 8000}},
	    {'brand': 'Blackberry', 'smartphone_detail': {'color' : 'Silver', 'price': 6000}}
	]
	
	del smartphone_dicts[1]
	print(smartphone_dicts)
	print(smartphone_dicts[0]["brand"])
	```	
	
<br />
<br />

---
	
### 객체 지향 프로그래밍 살펴보기
- 구조 설계한 후, 재사용성 증가
- 코드 반복 최소화됨
- 다양한 매직 메소드 활용할 수 있음
- 네임스페이스
	- 변수가 객체를 바인딩할 때, 그 둘 사이의 관계를 저장하고 있는 공간
	- a=2라고 할 때, 변수가 2라는 객체가 저장된 주소를 가지고 있는 상황
	- 파이썬의 클래스는 새로운 타입(객체)을 정의하기 위해 사용되고, 모듈과 마찬가지로 하나의 네임스페이스를 가짐
- `__init__` : 클래스 인스턴스 생성시 초기화하며 실행되는 부분
- class의 값을 보고 싶으면 `__dict__`을 사용
	- 네임스페이스를 확인
- python class str method
	- `__str__` 매직 메소드가 구현되어 있지 않은 상태에서 인스턴스를 print하면 object가 나옴
	- 비공식적으로 print문으로 출력하는 사용자 입장의 출력
	- print() 또는 str() 함수가 호출될 때 사용	- 기본적으로 str 메소드가 먼저 실행되며, str 메소드가 없으면 repr 메소드를 실행함
- `__repr__` : str과 비슷
	- 개발, 엔지니어 레벨에서 객체의 엄격한 타입을 표현할 땐 이 메소드를 사용
	- 객체 표현을 반환함
	- repr() 함수가 호출될 때 사용
- dir method
	- 해당 인스턴스가 가진 모든 attribute를 list 형태로 보여줌(값을 보여주진 않음)
- `__dict__`
	- 특정 네임스페이스만 보고 싶다면, `__dict__`를 사용
- `__doc__`
	- docstring을 출력


```
class Smartphone:
	"""
	Smartphone class
	"""
    def __init__(self, brand, details):
        self._brand = brand
        self._details = details

    def __str__(self):
        return f'str : {self._brand} - {self._details}'

    def __repr__(self):
        return f'repr : {self._brand} - {self._details}'
    

Smartphone1 = Smartphone('Iphone', {'color' : 'White', 'price': 10000})
Smartphone2 = Smartphone('Galaxy', {'color' : 'Black', 'price': 8000})
Smartphone3 = Smartphone('Blackberry', {'color' : 'Silver', 'price': 6000})

print(Smartphone1)
print(Smartphone1.__dict__)
print(Smartphone2.__dict__)
print(Smartphone3.__dict__)


# ID 확인 : 숫자가 모두 다름
print(id(Smartphone1))
print(id(Smartphone2))

print(Smartphone1._brand == Smartphone2._brand)
print(Smartphone1 is Smartphone2)

for x in Smartphone_list:
    print(repr(x))
    print(x)
    
print(Smartphone.__doc__)    
```
	
	
- self란?
	- self는 자기 자신을 뜻함. 인스턴스 자신
		- 인스턴스 : 클래스에 의해 만들어진 객체
	- self가 있는 것이 인스턴스 변수
	- 인자에 self를 받는 것은 인스턴스 메소드
		- detail_info 함수 구현
- 인스턴스의 `__class__`는 사용된 클래스를 출력함


```
class Smartphone:
    """
    Smartphone Class
    """
    def __init__(self, brand, details):
        self._brand = brandbrand
        self._details = details

    def __str__(self):
        return f'str : {self._brand} - {self._details}'

    def __repr__(self):
        return f'repr : {self._brand} - {self._details}'

    def detail_info(self):
        print(f'Current Id : {id(self)}')
        print(f'Smartphone Detail Info : {self._brand} {self._details.get('price'))}'


        
Smartphone1 = Smartphone('Iphone', {'color' : 'White', 'price': 10000})
Smartphone2 = Smartphone('Galaxy', {'color' : 'Black', 'price': 8000})
Smartphone3 = Smartphone('Blackberry', {'color' : 'Silver', 'price': 6000})
	
Smartphone1.detail_info
	
print(Smartphone1.__class__, Smartphone2.__class__)
# 부모는 같음
print(id(Smartphone1.__class__) == id(Smartphone3.__class__))
```	

- 클래스 변수
	- 클래스 내부에 선언된 변수
	- 클래스 변수는 클래스의 네임스페이스에 위치함
	- 모든 인스턴스가 공유하는 변수
	- `Smartphone1.__dict__`를 출력하면 클래스 변수는 보이지 않음
	- `dir(Smartphone1)`를 출력할 때는 클래스 변수가 보임
- 인스턴스 변수
	- self.name 같이 self가 붙은 변수
	- 인스턴스 변수는 인스턴스의 네임스페이스에 위치함
	- 인스턴스 네임스페이스에서 없으면 상위에서 검색
	- 즉, 동일한 이름으로 변수 생성 가능(인스턴스 검색 후 => 상위 클래스, 부모 클래스 변수)


```
class Smartphone:
    """
    Smartphone Class
    """
    # 클래스 변수
    smartphone_count = 0
    
    def __init__(self, brand, details):
        self._brand = brand
        self._details = details
        Smartphone.smartphone_count += 1

    def __str__(self):
        return f'str : {self._brand} - {self._details}'

    def __repr__(self):
        return f'repr : {self._brand} - {self._details}'

    def detail_info(self):
        print(f'Current Id : {id(self)}')
        print(f'Smartphone Detail Info : {self._brand} {self._details.get('price'))}'

    def __del__(self):
        Smartphone.smartphone_count -= 1
    
Smartphone1 = Smartphone('Iphone', {'color' : 'White', 'price': 10000})
Smartphone2 = Smartphone('Galaxy', {'color' : 'Black', 'price': 8000})
Smartphone3 = Smartphone('Blackberry', {'color' : 'Silver', 'price': 6000})
	
Smartphone1.detail_info
	
print(Smartphone1.__class__, Smartphone2.__class__)
# 부모는 같음
print(id(Smartphone1.__class__) == id(Smartphone3.__class__))
	
# 공유 확인
print(Smartphone.__dict__)
print(Smartphone1.__dict__)
print(Smartphone2.__dict__)
print(Smartphone3.__dict__)
print(dir(Smartphone1))
	
print(Smartphone1.cor_count)
print(Smartphone.smartphone_count)
```		

<br />
<br />

---

### 메소드의 종류
- 클래스 메소드(Python Class Method)
	- @classmethod 데코레이터를 사용
	- cls 인자를 받음
	- cls는 Smartphone를 뜻함(인스턴스 말고 클래스)
	- 클래스 변수 컨트롤할 때 사용
- 인스턴스 메소드(Python Instance Method)
	- Self가 들어간 경우
	- 객체의 고유한 속성 값을 사용
- 스태틱 메소드(Python Static Method)
	- 아무것도 인자를 받지 않음(self, cls 등)
	- 유연하게 사용함
	- [Meaning of @classmethod and @staticmethod for beginner?](https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner) 참고
	
	
```
class Smartphone:
    """
    Smartphone Class
    """
    # 클래스 변수
    Smartphone_count = 0
    
    # Instance Method
    # self : 객체의 고유한 속성 값 사용
    def __init__(self, brand, details):
        self._brand = brand
        self._details = details
        Smartphone.smartphone_count += 1

    def __str__(self):
        return f'str : {self._brand} - {self._details}'

    def __repr__(self):
        return f'repr : {self._brand} - {self._details}'

    def detail_info(self):
        print(f'Current Id : {id(self)}')
        print(f'Smartphone Detail Info : {self._brand} {self._details.get('price'))}'

    def get_price(self):
        return f'Before Smartphone Price -> brand : {self._brand}, price : {self._details.get('price')}'

    # Instance Method
    def get_price_culc(self):
        return f'After Smartphone Price -> brand : {self._brand}, price : {self._details.get('price') * Smartphone.price_per_raise}'

    # Class Method
    @classmethod
    def raise_price(cls, per):
        if per <= 1:
            print('Please Enter 1 or More')
            return
        cls.price_per_raise = per
        return 'Succeed! price increased.'

    # Static Method
    @staticmethod
    def is_iphone(inst):
        if inst._brand == 'Iphone':
            return f'OK! This Smartphone is {inst._brand}.'
        return 'Sorry. This Smartphone is not Iphone.'
    
    
Smartphone1 = Smartphone('Iphone', {'color' : 'White', 'price': 10000})
Smartphone2 = Smartphone('Galaxy', {'color' : 'Black', 'price': 8000})

# 기본 정보
print(Smartphone1)
print(Smartphone2)

# 전체 정보
Smartphone1.detail_info()
Smartphone2.detail_info()

# 가격 정보(인상 전)
print(Smartphone1.get_price())
print(Smartphone2.get_price())

# 가격 인상(클래스 메소드 미사용)
# 이렇게 직접 접근은 좋지 않아요
Smartphone.price_per_raise = 1.2

# 가격 정보(인상 후)
print(Smartphone1.get_price_culc())
print(Smartphone2.get_price_culc())

# 가격 인상(클래스 메소드 사용)
Smartphone.raise_price(1.6)

# 가격 정보(인상 후 : 클래스 메소드)
print(Smartphone1.get_price_culc())
print(Smartphone2.get_price_culc())

# Iphone 여부(스태틱 메소드 미사용)
def is_iphone(inst):
    if inst._brand == 'Iphone':
        return f'OK! This Smartphone is {inst._brand}.'
    return 'Sorry. This Smartphone is not Iphone.'

# 별도의 메소드 작성 후 호출
print(is_iphone(Smartphone1))
print(is_iphone(Smartphone2))

# Iphone 여부(스태틱 메소드 사용)
print('Static : ', Smartphone.is_iphone(Smartphone1))
print('Static : ', Smartphone.is_iphone(Smartphone2))

print('Static : ', Smartphone1.is_iphone(Smartphone1))
print('Static : ', Smartphone2.is_iphone(Smartphone2))

```	
	
<br />
<br />

---
	
### 상속	
- Class는 상속을 통해 자식 클래스에게 부모 클래스의 속성과 메소드를 물려줌
	- 예를 들어 Smartphone Class가 있고, Iphone Class, Galaxy Class 등이 있는 상황
	- Iphone과 Galaxy가 가치는 속성(attribute)는 다를 수 있음
	- 다중 상속도 가능함

```
class Smartphone:
    def __init__(self, brand, price):
        self._brand = brand
        self._price = price
      
    
    def __str__(self):
        return f'str : {self._brand} - {self._price}'

class Galaxy(Smartphone):
    def __init__(self, brand, price, country):
        self._brand = brand
        self._price = price
        self._country = country
 
    def __str__(self):
        return f'str : {self.__class__.__name__} 스마트폰은 {self._brand}에서 출시되었고, {self._country}에서 생산되었습니다. 가격은 {self._price}입니다'
    
iphone = Smartphone('IPhone', 7000)
print(iphone)
galaxy = Galaxy('Galaxy', 5000, 'South Korea')
print(galaxy)    
```

<br />
<br />

---

### Setter와 Getter, Property
- 객체의 속성(attribute)를 다룰 때 사용
- getter, setter는 객체의 속성을 읽고 변경할 때 사용함
	- 참고로 객체(Object)는 속성(Attribute)와 Method로 구현
- 자바 같은 객체 지향 언어에서 외부에서 바로 접근할 수 없는 private 객체 속성을 지원함
	- 이런 언어에선 private 속성의 값을 읽고(get) 변경(set)하기 위해 getter와 setter를 사용함
	- 파이썬은 모든 메서드가 public이기 때문에 getter와 setter 메소드가 없지만, 사용할 수는 있음

```
class Smartphone:
    def __init__(self, brand, price):
        self._brand = brand
        self._price = price
        
    def get_price(self):
        return self._price
        
    def set_price(self, price):
        self._price = price
    	
```

- property
	- 파이썬은 속성에 직접 접근을 막기 위해 property를 사용함
	- 데코레이터로 감싸서 사용함

```
class Smartphone:
    def __init__(self, brand, price):
        self._brand = brand
        self._price = price
 
    @property
    def price(self):
        return self._price
	    
    @price.setter
    def price(self, price):
        print(f"변경 전 가격 : {self._price}")
        self._price = price
        print(f"변경 후 가격 : {self._price}")

Smartphone1 = Smartphone("Iphone", 1000)
Smartphone1.price = 10000
```

- property를 사용하면 value의 제한을 들 수 있음
	- 예를 들면, Smartphone class에서 가격이 0원 미만일 경우 에러를 발생시킬 수 있음(property가 아니고 get, set으로 구현하면 에러가 발생하진 않음)
- 사용하는 목적
	- 변수를 변경할 때 제한사항을 두고 싶은 경우
	- getter, setter 함수를 만들지 않고 간단히 접근하기 위함

```
class Smartphone:
    def __init__(self, brand, price):
        self._brand = brand        self._price = price
	    
    @property
    def price(self):        return self._price
	    
    @price.setter
    def price(self, price):
        if price < 0:
            raise ValueError("Price below 0 is not possible")
        print(f"변경 전 가격 : {self._price}")
        self._price = price
        print(f"변경 후 가격 : {self._price}")

Smartphone1 = Smartphone("Iphone", 1000)
Smartphone1.price = 10000   
Smartphone1.price = -1000
```	



<br />
<br />

---

### 추상 메소드
- Class를 만들었다면 Class에서 정의된 메소드를 그대로 사용하지 않을 수 있음
	- 상속받고, 어떤 메소드는 추가하고 어떤 메소드는 오버라이드할 수 있음
	- 통일된 Class 체계를 구축하며 확장 기능을 가능하게 만드는 것이 Class
- 이런 개념을 토대로, Class를 만들 때 반드시 구현해야 할 메소드를 명시할 수 있음
- 아래 코드는 Sonata에서 func1 함수를 구현하지 않아 오류가 발생함

```
class Smartphone:
    def func1(cls):
        raise NotImplementedError()

class Iphone(Smartphone):
    pass
    
    
iphone = Iphone()
iphone.func1() # Error 발생
```

- 조금 더 Strict하고 세련된 방법은 `@abc.abstractmethod`를 사용하는 방법
	- abc는 abstract base class의 약자
	- Class의 인자로 metaclass=abc.ABCMeta를 지정하고,
		- Class에 데코레이터로 abc.abstractmethod를 사용하면, class가 호출되는 순간 구현해야 하는 메소드가 구현되었는지를 확인함
		- 참고 : metaclass는 단순히 클래스를 만들 때 사용됨. 클래스의 클래스
	- 바로 에러가 발생하기 때문에 인스턴스화시키지 않고, 추상화에 이점이 존재

```
import abc

class Smartphone(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def func1(cls):
        raise NotImplementedError()

class Iphone(Smartphone):
    def func2(self):
        pass


iphone = Iphone() # Error 발생
```


<br />
<br />

---

### slots
- 파이썬의 클래스들은 인스턴스 속성을 저장함
- 파이썬에서 객체의 인스턴스 속성을 저장하기 위해 dictionary를 사용함
	- 이런 방식은 속성을 추가하거나 삭제할 때 유용함
	- `obj.__dict__`으로 접근하는 경우
- 그러나 dictonary 자료 구조는 메모리를 많이 낭비함
	- 파이썬 객체 생성시 모든 속성을 메모리에 할당하려고 함
	- 메모리 할당량이 많으면 많은 RAM을 낭비하게 됨
- 이런 문제를 해결하기 위해 `__slots__`를 사용해 특정 속성에만 메모리를 할당하도록 함
- [ipython memory usage](https://github.com/ianozsvald/ipython_memory_usage)를 사용하면 `__slots__`를 사용해서 메모리 사용량이 얼마나 개선되는지 확인할 수 있음

- `__slots__` 없이 짠 코드

```
class Smartphone:
    def __init__(self, brand, price):
        self._brand = brand
        self._price = price
        self.set_up()
	    
    # 코드 생략
    # set_up은 init시 실행하는 함수라고 생각
```

- `__slots__`를 사용해 짠 코드

```
class Smartphone:
    __slots__ = ['_brand', '_price']
   
    def __init__(self, brand, price):        self._brand = brand        self._price = price
        self.set_up()
	    
    # 코드 생략
    # set_up은 init시 실행하는 함수라고 생각
```    



- 추후 추가적으로 다룰 이야기
	- MetaClass


<br />
<br />

---

### Reference	
- [인프런 파이썬 중급 프로그래밍](https://www.inflearn.com/course/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%A4%91%EA%B8%89-%EC%9D%B8%ED%94%84%EB%9F%B0-%EC%98%A4%EB%A6%AC%EC%A7%80%EB%84%90)
- [Python Class Document](https://docs.python.org/ko/3/tutorial/classes.html)