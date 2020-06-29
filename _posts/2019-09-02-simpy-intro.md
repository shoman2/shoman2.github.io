---
layout: post
title:  "Python SimPy 사용법 - 파이썬으로 시뮬레이션 만들기"
subtitle: "Python SimPy 사용법 - 파이썬으로 시뮬레이션 만들기"
categories: data
tags: simulation
comments: true
---


- Python SimPy 프레임워크를 사용해 파이썬으로 시뮬레이션 만드는 방법에 대해 작성한 글입니다
	- Discrete Event Simulation Python Framework


---

### SimPy
- SimPy는 process 기반 discreate-event 시뮬레이션 프레임워크
    - 코루틴을 잘 사용한 예시
    - SimPy의 process는 Python 제네레이터 함수로 정의되고, 고객, 차량 에이전트 같은 active component를 모델링할 때 사용될 수 있음
    - 제한된 용량을 모델링하기 위해 shared resource도 제공함(서버나 체크아웃 카운터, 터널 등)
    - 이론적으로 SimPy로 continuous simulation을 할 수 있지만, 도와주는 기능은 없음
    - 반면 SimPy는 공유된 자원끼리 상호작용이 없는 고정된 크기의 시뮬레이션엔 오버 스펙임
    - [SimPy Document](https://readthedocs.org/projects/simpy/downloads/pdf/latest/)
    - [영상 자료](https://www.youtube.com/watch?v=Bk91DoAEcjY)
    - 소스코드 : [Gitlab](https://gitlab.com/team-simpy/simpy)
- 짧은 예제
    - 2개의 시계가 다른 time interval로 움직이는 예제

	```
	import simpy
	    
	def clock(env, name, tick):
	    while True:
	        print(name, env.now)
	        yield env.timeout(tick)
	    
	env = simpy.Environment()
	env.process(clock(env, 'fast', 0.5))
	env.process(clock(env, 'slow', 1))
	    
	env.run(until=2)
	```

### 설치
```
pip3 install simpy
```

### Introduction
- 사실 SimPy는 단순한 비동기 이벤트 dispatcher임
    - 주어진 시뮬레이션 시간에 이벤트를 생성하고 예약함
    - 이벤트는 우선 순위, 시뮬레이션 시간, event id에 의해 정렬됨
    - 이벤트는 콜백 list를 가지고 있으며, event loop에 의해 실행됨
    - 이벤트는 return value를 가질 수 있음
- 구성 요소
    - <img src="https://www.dropbox.com/s/0d64nggr9ifd034/Screenshot 2019-09-01 23.57.27.png?raw=1">
    - 1) Process
    - 2) Resource
        - 2-1) Resources
            - Resource
            - PriorityResource
            - PreemptiveResource
        - 2-2) Containers
        - 2-3) Stores
            - Store
            - FilterStore
            - PriorityStore
    - 3) Environment

---

### 1) Process

- customer, vehicles 등 일반적인 agent를 모델링
- 순서가 있는 형태의 프로세스
- Event 인스턴스를 yield하는 제네레이터 함수를 만듬
- Generator 형태로 생성
    - 제네레이터 내부에서 특정 activity가 수행될 때 resource에서 request를 날려 resource를 일정 시간동안 사용
- process의 이벤트 시간이 모두 지나면, 이벤트의 value를 받을 수 있음
- 중요한 이벤트 유형 : `Timeout`
    - 이 유형은 일정 시간이 지난 후 트리거됨. 주어진 시간동안 휴면 상태를 유지함
    - `Environment.timeout()`으로 사용할 수 있음
	
	```
	import simpy
	    
	def car(env):
		"""
		자동차 프로세스
		주차하고 여행을 떠남
		parking과 driving 상태를 스위칭함
		"""
	    while True:
	        print('Start parking at %d' % env.now)
	        parking_duration = 5
					# 환경에서 timeout 이벤트를 발생시킴(parking_duration동안 휴면)
	        yield env.timeout(parking_duration)
	    
	        print('Start driving at %d' % env.now)
	        trip_duration = 2
	        yield env.timeout(trip_duration)
	    
	env = simpy.Environment()
	env.process(car(env))
	env.run(until=15)
```
	
- <img src="https://www.dropbox.com/s/m5r2i3x0awuwbw4/Screenshot 2019-09-02 00.03.57.png?raw=1">
- <img src="https://www.dropbox.com/s/lpkx2fovalw16bb/Screenshot 2019-09-02 00.04.11.png?raw=1">
- <img src="https://www.dropbox.com/s/i1rv04i2527jm5c/Screenshot 2019-09-02 00.04.23.png?raw=1">

---

### 2) Resource

- 용량이 제한된 일종의 컨테이너
- 프로세스는 자원에 무언가를 넣거나 얻으려고 시도함
- 리소스가 꽉 찼거나 비어있으면 대기열에서 대기해야 함
- 모든 리소스엔 최대 용량과 두 개의 대기열이 있음
    - 대기열 1개는 리소스를 넣기 위한 프로세스용 : put ⇒ request
    - 다른 하나는 꺼내기를 위한 프로세스용 : get ⇒ release
- 프로세스가 resource에 request하고, 사용한 후 release함
    - 예를 들어 주유소에 차량이 도착하고, fule pump가 되면 떠남
- 리소스 구조
	
	```
	BaseResource(capacity):
	   put_queue
	   get_queue
	    
	   put(): event
	   get(): event
   ```

- 총 3가지로 구현됨
    - `Resources`
        - Resource : 양수여야 하고 기본값은 1
            - 현재 사용자 또는 대기중인 사용자 목록, 리소스 용량을 검색할 수 있음
            
            ```
            import simpy
            
            res = simpy.Resource(env, capacity=1)
            
            def print_stats(res):
                print('%d of %d slots are allocated.' % (res.count, res.capacity))
                print('  Users:', res.users)
                print('  Queued events:', res.queue)
            
            def user(res):
                print_stats(res)
                with res.request() as req:
                    yield req
                    print_stats(res)
                print_stats(res)
            
            procs = [env.process(user(res)), env.process(user(res))]
            env.run()
            ```

            - <img src="https://www.dropbox.com/s/515rnaqx78rt9ev/Screenshot 2019-09-02 00.05.24.png?raw=1">
        - PriorityResource
            - 프로세스가 각 요청에 우선 순위를 제공할 수 있음
            - 더 중요한 요청 먼저 리소스에 액세스함
            - 숫자가 작을수록 우선 순위가 높음

            ```
            import simpy
            
            def resource_user(name, env, resource, wait, prio):
                yield env.timeout(wait)
                with resource.request(priority=prio) as req:
                    print('%s requesting at %s with priority=%s' % (name, env.now, prio))
                    yield req
                    print('%s got resource at %s' % (name, env.now))
                    yield env.timeout(3)
            
            env = simpy.Environment()
            res = simpy.PriorityResource(env, capacity=1)
            p1 = env.process(resource_user(1, env, res, wait=0, prio=0))
            p2 = env.process(resource_user(2, env, res, wait=1, prio=0))
            p3 = env.process(resource_user(3, env, res, wait=2, prio=-1))
            env.run()
            # p3이 p2보다 늦게 리소스 요청했지만 우선 순이가 높아 더 일찍 리소스 사용함
            ```
            
        - PreemptiveResource
            - 종종 새로운 요청이 기존 자원을 내쫓고 점유해야할 수 있음
            - PriorityResource에서 상속하고 preempt flag를 추가 request함(True로)
            - PreemptiveResource는 preemption보다 priorities가 더 우선순위가 높음
            - low priority, preemptive가 high priority를 뛰어넘을 수 없음

            ```
            import simpy
            
            def resource_user(name, env, resource, wait, prio):
                yield env.timeout(wait)
                with resource.request(priority=prio) as req:
                    print('%s requesting at %s with priority=%s' % (name, env.now, prio))
                    yield req
                    print('%s got resource at %s' % (name, env.now))
                    try:
                        yield env.timeout(3)
                    except simpy.Interrupt as interrupt:
                        by = interrupt.cause.by
                        usage = env.now - interrupt.cause.usage_since
                        print('%s got preempted by %s at %s after %s' %
                              (name, by, env.now, usage))
            
            env = simpy.Environment()
            res = simpy.PreemptiveResource(env, capacity=1)
            p1 = env.process(resource_user(1, env, res, wait=0, prio=0))
            p2 = env.process(resource_user(2, env, res, wait=1, prio=0))
            p3 = env.process(resource_user(3, env, res, wait=2, prio=-1))
            env.run()
            ```
            
    - `Containers`
        - 미분화된 대량 생산, 소비를 모델링할 수 있음
        - 물 같은 연속적인 것과 사과와 같은 불연속적인 것이 사용 가능
        - 주유소의 가스 / 휘발유 탱크를 모델링할 때 사용할 수 있음
        - 현재 level을 검색할 수 있음. capacity(GasStation.monitor_tank()
        
        ```
        import simpy
        
        class GasStation:
            def __init__(self, env):
                self.fuel_dispensers = simpy.Resource(env, capacity=2)
                self.gas_tank = simpy.Container(env, init=100, capacity=1000)
                self.mon_proc = env.process(self.monitor_tank(env))
        
            def monitor_tank(self, env):
                while True:
                    if self.gas_tank.level < 100:
                        print('Calling tanker at %s' % env.now)
                        env.process(tanker(env, self))
                    yield env.timeout(15)
        
        def tanker(env, gas_station):
            yield env.timeout(10)  # Need 10 Minutes to arrive
            print('Tanker arriving at %s' % env.now)
            amount = gas_station.gas_tank.capacity - gas_station.gas_tank.level
            yield gas_station.gas_tank.put(amount)
        
        def car(name, env, gas_station):
            print('Car %s arriving at %s' % (name, env.now))
            with gas_station.fuel_dispensers.request() as req:
                yield req
                print('Car %s starts refueling at %s' % (name, env.now))
                yield gas_station.gas_tank.get(40)
                yield env.timeout(5)
                print('Car %s done refueling at %s' % (name, env.now))
        
        def car_generator(env, gas_station):
    		"""
    		차량을 생성하는 함수
    		"""
            for i in range(4):
                env.process(car(i, env, gas_station))
                yield env.timeout(5)
        
        env = simpy.Environment()
        gas_station = GasStation(env)
        car_gen = env.process(car_generator(env, gas_station))
        env.run(35)
        ```
        
    - `Stores` : object의 생산과 소비를 모델링 할 수 있음
        - 일반적인 Store
        
        ```
        import simpy
        
        def producer(env, store):
            for i in range(100):
                yield env.timeout(2)
                yield store.put('spam %s' % i)
                print('Produced spam at', env.now)
        
        
        def consumer(name, env, store):
            while True:
                yield env.timeout(1)
                print(name, 'requesting spam at', env.now)
                item = yield store.get()
                print(name, 'got', item, 'at', env.now)
        
        
        env = simpy.Environment()
        store = simpy.Store(env, capacity=2)
        
        prod = env.process(producer(env, store))
        consumers = [env.process(consumer(i, env, store)) for i in range(2)]
        
        env.run(until=5)
        ```
        
        - FilterStore : store에서 object를 가져갈 때 커스텀 함수를 사용하는 store
            - 다양한 속성이 있는 기계 공장을 모델링함
            - FilterStore의 items에 값을 저장함!!
        
		        ```
		        import simpy
		        
		        from collections import namedtuple
		        
		        Machine = namedtuple('Machine', 'size, duration')
		        m1 = Machine(1, 2)  # Small and slow
		        m2 = Machine(2, 1)  # Big and fast
		        
		        env = simpy.Environment()
		        machine_shop = simpy.FilterStore(env, capacity=2)
		        machine_shop.items = [m1, m2]  # Pre-populate the machine shop
		        
		        def user(name, env, ms, size):
		            machine = yield ms.get(lambda machine: machine.size == size)
		            print(name, 'got', machine, 'at', env.now)
		            yield env.timeout(machine.duration)
		            yield ms.put(machine)
		            print(name, 'released', machine, 'at', env.now)
		        
		        
		        users = [env.process(user(i, env, machine_shop, (i % 2) + 1))
		                 for i in range(3)]
		        env.run()
		        ```
        
        - PriorityStore : 우선 순위에 따라 필터링할 수 있는 store
            - inspector 프로세스가 maintainer 프로세스가 우선 순위에 따라 복구되도록 기록함
            
            ```
            import simpy
	    
			env = simpy.Environment()
			issues = simpy.PriorityStore(env)
			    
			def inspector(env, issues):
			    for issue in [simpy.PriorityItem('P2', '#0000'),
			                  simpy.PriorityItem('P0', '#0001'),
			                  simpy.PriorityItem('P3', '#0002'),
			                  simpy.PriorityItem('P1', '#0003')]:
			        yield env.timeout(1)
			        print(env.now, 'log', issue)
			        yield issues.put(issue)
			    
			def maintainer(env, issues):
			    while True:
			        yield env.timeout(3)
			        issue = yield issues.get()
			        print(env.now, 'repair', issue)
			    
			_ = env.process(inspector(env, issues))
			_ = env.process(maintainer(env, issues))
			env.run()
            ```

	        
---

### 3) Environment

- simulation 하려는 환경
- 환경은 이벤트 목록에 이벤트들을 저장하고 현재 시뮬레이션 시각을 추적함
- <img src="https://www.dropbox.com/s/ztbxm58ebklyxgx/Screenshot 2019-09-02 00.01.45.png?raw=1">
- <img src="https://www.dropbox.com/s/f05xzo8ql4v8uhp/Screenshot 2019-09-02 00.02.30.png?raw=1">

---

### 예시

- 차량 인스턴스 예시

	```
	import simpy
	    
	class Car:
	    def __init__(self, env):
	        self.env = env
	        # Start the run process everytime an instance is created.
	        self.action = env.process(self.run())
	    
	    def run(self):
	        while True:
	            print('Start parking and charging at %d' % self.env.now)
	            charge_duration = 5
	            # We yield the process that process() returns
	            # to wait for it to finish
	            yield self.env.process(self.charge(charge_duration))
	    
	            # The charge process has finished and
	            # we can start driving again.
	            print('Start driving at %d' % self.env.now)
	            trip_duration = 2
	            yield self.env.timeout(trip_duration)
	    
	    def charge(self, duration):
	        yield self.env.timeout(duration)
	    
	env = simpy.Environment()
	car = Car(env)
	env.run(until=15)
	```

- 자동차가 완전히 충전될 때까지 기다리지 않고 도중에 충전 중단하고 운전할 경우
    - interrupt()를 호출해 실행 중 프로세스를 중단할 수 있음
    
    ```
    import simpy
    
    def driver(env, car):
        yield env.timeout(3)
        car.action.interrupt()
        
    class Car:
        def __init__(self, env):
            self.env = env
            self.action = env.process(self.run())
    
        def run(self):
            while True:
                print('Start parking and charging at %d' % self.env.now)
                charge_duration = 5
                # We may get interrupted while charging the battery
                try:
                    yield self.env.process(self.charge(charge_duration))
                except simpy.Interrupt:
                    # When we received an interrupt, we stop charging and
                    # switch to the "driving" state
                    print('Was interrupted. Hope, the battery is full enough ...')
    
                print('Start driving at %d' % self.env.now)
                trip_duration = 2
                yield self.env.timeout(trip_duration)
    
        def charge(self, duration):
            yield self.env.timeout(duration)
            
    env = simpy.Environment()
    car = Car(env)
    env.process(driver(env, car))
    env.run(until=15)
    ```
    
- 다음 글에선 같이 시뮬레이션 예제를 구현해보는 글을 작성할 예정입니다 :) 
