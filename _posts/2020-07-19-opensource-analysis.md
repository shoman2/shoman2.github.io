---
layout: post
title:  "Python 오픈소스(Open Source) 분석 방법"
subtitle: "파이썬 Open source 분석하는 개인적인 방법"
categories: development
tags: python
comments: true
---


- Python Open Source 분석 방법에 대해 작성한 글입니다
	- 이 글이 진리는 아니고, 제가 개인적으로 시도했던 방법에 대해 작성했습니다! 
	- 더 좋은 방법을 알고 계시면, 댓글로 알려주시면 좋을 것 같습니다 :)
	- 특히, 파이썬 오픈소스에 집중했습니다
- 목차
	- [Open Source란?](#open-source란)
	- [오픈소스를 분석하면 좋은 점](#오픈소스를-분석하면-좋은-점)
	- [기본 지식](#기본-지식)
	- [오픈소스 선택하기](#오픈소스-선택하기)
	- [공식 문서를 토대로 오픈소스 파악하기](#공식-문서를-토대로-오픈소스-파악하기)
	- [Clone 또는 Fork](#clone-또는-fork)
	- [큰 숲 바라보기 - 아키텍쳐 분석](#큰-숲-바라보기---아키텍쳐-분석)
	- [디테일 분석 - 함수, 클래스 등](#디테일-분석---함수-클래스-등)
	- [팁](#팁)
	- [같이 분석해보기 - SimPy](#같이-분석해보기---simpy)
	- [마치며](#마치며)
	
		

<br />

---

<br />


### Open Source란?
- [Wiki](https://ko.wikipedia.org/wiki/%EC%98%A4%ED%94%88_%EC%86%8C%EC%8A%A4_%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4)에 의하면 오픈소스 소프트웨어는 소스 코드를 공개해 누구나 제한없이 사용할 수 있는 오픈소스 라이선스를 만족하는 소프트웨어를 뜻함
	- 라이선스는 Apache License, GNU, MIT, BSD, GNU 등 다양한 라이선스가 존재함
- 데이터 분석을 한다면 자주 사용할 Pandas, Numpy, 머신러닝 오픈소스인 scikit learn, 딥러닝 오픈소스 중 유명한 Tensorflow, PyTorch, MXNet 등을 이야기할 수 있음
	- 언어별로 오픈소스가 다양함
- 아마 프로그래밍을 하면, 오픈소스를 자주 접하게 됨


<br />

---

<br />


### 오픈소스를 분석하면 좋은 점
- (제가 생각하는) 오픈소스를 분석하면 좋은 점은 다음과 같음
- 1) 본인이 익숙한 라이브러리의 내부를 알 수 있는 점
- 2) 코드 퀄리티 향상
	- 파이썬 프로그래밍 방식
	- Low Level에선 코드를 어떻게 추상화하는지 파악 가능
- 3) 오픈소스 기여를 위한 사전 단계
	- 오픈소스에서 어떻게 협업하는지 간접적으로 알 수 있고, Pull Request나 Issue 등록하며 Github에 익숙해질 수 있음
- 4) 다양한 지식을 추가로 습득할 수 있음
	- 오픈소스에서 추상화해둔 코드를 보며, 그 안에 필요한 키워드를 찾을 수 있음
- 5) 오픈소스의 버그를 찾을 수 있음
	- 오픈소스 코드를 디버깅하며 이 부분이 이상한데? 등을 거슬러 올라가면 버그를 찾을 수 있음
- 6) 메타인지를 가질 수 있음
	- 메타인지는 생각에 대한 생각으로 나타낼 수 있는데, 코드의 코드로 이걸 왜 이렇게 구현했지? 다른 방식으로 구현하면 어떨까? 등을 고민해볼 수 있음

<br />

---

<br />

### 기본 지식
- 프로그래밍 언어 지식은 어느정도 필요함
	- 그렇다고 프로그래밍 언어의 모든 것을 알아야 하는 것은 아님
	- 우선 간단히 파이썬 언어의 특징을 이해하면 좋고, 오픈소스에서 나타나는 파이썬 코드를 통해 학습하면 됨
- IDE
	- 오픈소스를 분석할 때, Github Repo에서 이동할 수 있지만 빠르게 탐색하기 위해 파이참 등을 잘 활용하면 좋음
- 오픈소스 선택하기 위해 오픈소스에 대한 이해가 필요함
	- 예를 들어 Pandas는 데이터를 조작하기 위해 사용하는 오픈소스다! 라는 오픈소스의 핵심을 이해하고 있어야 함(최소한!)
- 오픈소스에서 필요로 하는 지식은 있어야 함
	- 예를 들어 Apache Spark를 분석하려고 하는데, 병렬 처리 관련 이론을 모르면 어려울 수 있음
	- 공식 Document를 읽어보고 추가 공부!
	- 꽤 큰 오픈소스라면 논문이 있는 경우도 존재함 => 논문 읽고 정리도 좋음


<br />

---

<br />


### 오픈소스 선택하기
- 1) 만약 처음이라면 우리가 많이 아는 유명한 오픈소스를 선택하는 방법도 있지만, 상대적으로 덜 유명한 오픈소스를 선택하는 방법도 있음
	- 덜 유명한 오픈소스는 우선 아키텍쳐가 상대적으로 간단할 수 있음
	- Tensorflow나 Pandas 같은 오픈소스는 꽤 오래 되었기 때문에, 초심자 입장에서 바로 접근하기에 진입 장벽이 있을 수 있음
	- 혹은 초반 버전 선택하는 방법도 있음
- 2) 혹은 본인이 많이 사용하고 있는 라이브러리
	- 회사 업무로 Airflow를 많이 사용하고 있고, 그 안을 더 깊게 이해하고 싶은 경우라면 당연히 좋음! (어려울 수 있지만)
- 3) 얼마나 활발하게 개발되고 있는지
	- 최근에 올라온 Commit이 언제인지, Issue에 올라온 내용이 답변이 달리는지 등
	- 활발한 프로젝트일수록 오픈소스 분석하다 어려운 내용을 물어보면 빠른 답변을 기대할 수 있음
	- 보조적으로 라이브러리의 Version이 몇인지 확인해서 활발하게 개발되는지를 볼 수 있음
- 4) 기업이 주도해서 만드는지 or 개인이나 작은 팀이 만드는지
	- 3)과도 연관되는 내용인데 기업에서 주도할 경우 해당 오픈소스를 관리할 팀이 따로 존재하곤 함 => 답변을 받을 수 있는 기대 가능
- **제일 중요한 것은 2) 본인이 많이 사용하고, 지속적으로 관심을 가질 수 있는지가 중요함**


<br />

---

<br />


### 공식 문서를 토대로 오픈소스 파악하기
- 오픈소스를 분석하기 전, 혹시 이 오픈소스가 처음이라면 공식 문서를 통해 라이브러리의 사용 방법을 익혀두면 좋음
- 어떤 코드를 먼저 살펴볼지 방향을 잡을 때 도움이 됨
- Getting Started, User Guide, API Reference 등이 있는데 Getting Started에 많은 정보들이 포함됨
	- 예를 들어, 오픈소스 Overview엔 어떤 목적으로 만들어졌고, 어떤 특징이 있는지 나옴
	- 여기에 나오는 예시 코드를 주로 먼저 파악함
- Version History엔 현재 오픈소스가 몇 버전이고, 어떤 기능이 언제 추가되었는지 확인할 수 있음
- 이 시기에 공식 Document, 창시자가 블로그에 올린 글, 각종 컨퍼런스에서 발표한 유튜브 영상 들을 보고 창시자의 "철학"을 파악함




<br />

---

<br />


### Clone 또는 Fork
- Github, Gitlab, BitBucket 등에 오픈소스가 저장되어 있음
- 인터넷 창에서 확인할 수 있지만, 조금 더 쉽게 이동하고 필요할 경우 커스텀 로그를 남기기 위해 개인 컴퓨터로 파일을 Clone한다
	- 오픈소스 분석이 아닌 기여를 하고 싶은 경우엔 Fork를 한 후, Pull Request를 날리면 됨

<br />

---

<br />


### 큰 숲 바라보기 - 아키텍쳐 분석
- 개인의 Local로 소스를 Clone한 후, 본격적인 분석을 시작함
- 대부분 오픈소스의 이름의 폴더가 있음
	- 그 안에 핵심 코드들이 저장됨
	- 패키지를 만들어본 경험이 있다면 이게 당연하지만, 모르는 분들을 위해 캡쳐
<img src="https://www.dropbox.com/s/b0r69fi0gfm8196/Screenshot%202020-07-19%2017.27.25-tile.png?raw=1">

- 처음엔 오픈소스의 아키텍쳐, 숲을 먼저 확인한 후 디테일한 부분으로 이동함
- 간단하게 할 수 있는 것은 폴더별로 어떤 파일이 있는지 확인하고, 공식 문서 등에서 확인했던 메소드가 어디에 저장되어 있는지 파악함
- 큰 아키텍쳐를 draw.io나 키노트 등에서 그려봄

<br />

---

<br />


### 디테일 분석 - 함수, 클래스 등
- 큰 숲을 파악한 후, 디테일한 부분을 분석함
- 클래스엔 어떤 정보를 담고 있고, 어디서 활용되며, 함수는 어떤 방식으로 동작하는지 등을 파악함
	- 모든 함수를 다 파악하겠다! 라고 생각하지 않고 조금씩 시도함
	- Class는 최상위에 어떤 클래스가 있고, 그걸 어디서 상속받는지 등을 확인함
- 이 때, IDE에서 breakpoint를 걸어두고 함수를 실행해봄
	- 직접 눈으로 어던 값이 들어가있는지 확인해보는게 이해에 큰 도움이 됨(모두 실행하긴 힘들지만..)
- 어떤 Input, Output이 오는지 감을 잡고 싶으면 Test 코드를 확인해봄
	- 보통 Test Code엔 어떤 Input이 들어와서 Output이 뭐가 나오는지 정의되어 있어 코드를 이해할 때 도움을 줌
- 네이밍을 어떻게 했는지도 파악하며, 특정 메소드는 Protected method고, Private method 인지 등을 파악함
- 잘 되어있는 오픈소스라면 주석을 잘 남겼을테니, 주석을 주의깊게 보는 것도 유용함
- 나라면 이렇게 짜볼 수 있지 않을까?도 생각해보고 메모해둠 
- 추후 분석한 내용을 까먹지 않기 위해 노션 등에 기록해둠



<br />

---

<br />


### 팁
- **포기하지 말기** (★★★★★)
	- 오픈소스 분석은 익숙하지 않다면, 매우 힘들고 고된 작업일 수 있음
	- 하지만 이 과정에서 조금씩 성장하고 있습니다!
	- 혼자 어렵다면 여러 사람들과 함께 분석하는 방법도 있음
	- 힘들어도 한 싸이클을 꼭 돌아보기!!
- 분석하며 문서, 이미지로 정리하기
	- 위에도 말했지만, 이미지화하거나 문서로 정리해둬야 추후에 분석한 내용을 다시 볼 수 있음
- 창시자의 문서, 유튜브 영상 보기
	- 창시자의 철학, 창시자가 생각한 베이스라인을 이해할 수 있음
- 이해가 안될 경우엔 디버거의 breakpoint 사용하기
- Test Code 참고하기
	- Test Code와 디버거를 잘 활용하면 유용함!
- 모든 것을 알려고 하지말고, 관심있는 부분부터 조금씩 파악하기
- 코드에서 학습하기. 새로운 것들!
	- 네이밍 공부
	- 코드 스타일
	- 타입 힌팅 등
- 코드를 무조건 많이 보는 것을 추천
	- 인쇄해서 보는 분들도 있다고 들음
	- 코드를 보며 코드를 보는 안목을 기를 수 있음

	

<br />

---

<br />

### 같이 분석해보기 - SimPy
- SimPy 전체 분석 내용은 아니고, 일부분 분석할 때 작성해둔 메모입니다
- SimPy는 Discrete Event Simulation Python Framework로 시뮬레이션을 만들 때 활용할 수 있는 오픈소스
- [Gitlab](https://gitlab.com/team-simpy/simpy/-/tree/master/)에 소스가 저장됨
- SimPy 소개는 [Python SimPy 사용법 - 파이썬으로 시뮬레이션 만들기](https://zzsza.github.io/data/2019/09/02/simpy-intro/)에서 확인할 수 있음
	- 코루틴을 잘 활용한 프레임워크로 SimPy의 process는 Python Generator로 정의됨
- 예제 코드
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

	- 이 코드를 보면 다음과 같은 생각을 할 수 있음
		- `simpy.Environment` 코드를 봐야겠다
		- `env.process`와 `env.run` 코드도 봐야겠다
- simpy 코드를 clone
	
	```
	git clone https://gitlab.com/team-simpy/simpy.git
	```
	
- 아키텍쳐 파악
	- 생각보다 파일 개수가 적음
	
<center>	
<img src="https://www.dropbox.com/s/3gq8uzdzmk5wvh5/Screenshot%202020-07-19%2018.23.21.png?raw=1" width="300" height="500">
</center>

- Mac에서 [Tree](https://coding-start.tistory.com/108)를 설치해 트리 구조를 출력하고 정리 시작
	- 다행히 SimPy는 파일 위에 파일에 대한 정리가 이미 작성되어 있어서 수월한 편

	```
	├── __init__.py : SimPy에서 가장 많이 사용되는 구성 요소를 single namespace로 통합(편의상 진행)
	├── core.py : discrete 시뮬레이션 환경을 위한 핵심 요소
	├── events.py : SimPy에서 사용되는 기본 이벤트 타입이 저장됨. 모든 이벤트 기본 클래스는 `Event`
	├── exceptions.py : Exception
	├── resources : SimPy의 리소스들, resoucre, container, store가 있음
	│   ├── __init__.py
	│   ├── base.py : SimPy의 shared resource 타입, simpy.events의 Event를 뭔가 상속해서 만듬
	│   ├── container.py : 프로세스간에 물질을 공유하기 위한 리소스. 가스 스테이션 같은 것
	│   ├── resource.py : priorities와 preemption가 있다고 함
	│   └── store.py : unlimited amout of objects를 지원하는 리소스
	├── rt.py : Real Time의 약자인듯, 실시간으로 동기화하는 이벤트 실행 환경
	└── util.py : 유틸리티성 함수
	```
	
	- 위에 작성한 것처럼, 큰 틀을 빠르게 파악함(다 파악하는데 5분도 소요되지 않음. 자세히 읽으려 하지 말고, 대충 이런 것들이 있나? 정도의 느낌으로 파악)	
- simpy.Environment 코드 파악
	- src/simpy로 들어감
	- environment.py 같은 것은 보이지 않으니 core.py를 확인
	- Environment Class를 확인할 수 있음
	- <img src="https://www.dropbox.com/s/00dd2s0abeph9xf/Screenshot%202020-07-19%2018.47.29.png?raw=1">
- 코드를 보고 알게된 내용
	- Environment는 initial_time을 받는구나
		- 어? 근데 initial_time: SimTime은 뭐지? 보통 이 자리엔 타입이 들어가는데?
		- 위에 코드 정의를 보니 `SimTime = Union[int, float]`이 있구나!
		- Union이 뭔지 찾아보니 [typing.Union](https://python.flowdas.com/library/typing.html#typing.Union)이 있네! 여러 타입을 가지고 있는 공용체구나. 즉 SimTime은 int 또는 float을 가지고 있군!
	- _queue에 List로 Tuple을 저렇게 넣기도 하는구나
		- 변하면 안되서 Tuple로 사용했을까?
	- _eid는 count 함수를 사용하네. 이건 또 무슨 함수일까
	- _active_proc에 Optional[Process]라고 되어있는데 Process는 어떻게 정의되어있나 봐야겠네
	- BoundClass는 또 뭐지?
		- 위에 정의를 보니 클래스가 메소드처럼 동작하도록 하는구나! 
		- `__get__()` descriptor는 기본적으로 `function.__get__()`과 동일하고, cls는 descriptor instnace에 바인딩한다고 함
		- descriptor는 또 뭘까?
			- [디스크립터 사용법 안내서](https://docs.python.org/ko/3/howto/descriptor.html)를 자세히 봐야겠다
			- Generic이란 용어도 나오네. 이거도 나중에 확인해야지
	- 일단 돌아가서 Environment 클래스를 다시 보자
		- <img src="https://www.dropbox.com/s/xlje7lx7wu3cjzq/Screenshot%202020-07-19%2018.59.43.png?raw=1">
	- @property를 이렇게 사용했구나
	- type_checking으로 분기를 타는데, process 메소드가 구현되어 있네! 이게 env.process였나
		- Process(self, generator)라서 제네레이터를 가지고 뭔가 하는구나
		- Command 키를 누르고 Process를 클릭해서 어떻게 생겼는지 보자!
		- events.py로 이동함
		- <img src="https://www.dropbox.com/s/rn22545u2uhdagz/Screenshot%202020-07-19%2019.02.02.png?raw=1">
	- 오 hasattr() 수업 시간에 배운건데, throws가 있는지 확인하네?
		- Cython 이야기도 나오네.. 참 알아야 하는 것들이 많은데, 일단 체크해두자! 어쨋든 이게 없으면 ValueError가 발생하네
	- 돌아와서 Environment의 schedule 함수는 heappush를 사용하네. heapque 내장 모듈에 대해 역시 잘 알아야겠다. 자료구조 공부해야지!
	- env.run을 보기로 했으니, run 메소드를 확인해보자
	- <img src="https://www.dropbox.com/s/1q1uoyitl7peboj/Screenshot%202020-07-19%2019.08.33.png?raw=1">
	- docstring만 봐도 어떤 일을 하는지 잘 이해할 수 있겠다.. 역시 문서화가 중요해
	- 주어진 기준 until을 충족할 때까지 step()을 실행함
	- 스텝을 거치면서 실행하나보다
- 일단 이정도로 첫날 분석을 마무리하고, 추가적으로 공부할 내용 정리해보면!
	- Descriptor 확인하기
	- heapque 다시 공부하기
	- Cython 사용 방법
	- typing 모듈 확인하기
	
	
<br />

---

<br />

### 마치며
- 오픈소스를 분석하는 개인적인 방법에 대해 정리했음
- 이 방법이 최선의 방법은 아니고, 개인적으로 진행하는 방법!
	- 아직 다듬어지진 않아서, 더 개선이 필요할 것 같음
	- 같이 분석해보기쪽은 정리를 깔끔하게 하기보단, 약간 산만하게 분석하는 느낌
	- 하지만 코드를 안보는 것보단 더 좋다고 생각!
- 현재 Scikit-Learn, Pandas, SimPy, Ray 등의 오픈소스를 분석해두었는데, 이젠 분석만 하지않고 오픈소스에 기여하는 방법도 고민할 예정
- 제일 중요한 것은 **포기하지 말고 꾸준히 하는 것!** 한 사이클이라도 꼭 돌아보면 좋을 것 같아요(작은 단위의 분석부터 시작!)



<br />

---

<br />

### 오픈소스 분석을 위해 봤던 자료들
- [오픈소스를 시작하려는 히치하이커를 위한 안내서](https://speakerdeck.com/dongjin/the-hitchhikers-guide-to-open-source)
- [김형준님의 오픈소스: 코드 분석 어떻게 하나?](https://www.popit.kr/%EC%98%A4%ED%94%88%EC%86%8C%EC%8A%A4-%EC%BD%94%EB%93%9C-%EB%B6%84%EC%84%9D-%EC%96%B4%EB%96%BB%EA%B2%8C-%ED%95%98%EB%82%98/)
- [오픈소스로 개발 실력 쌓기](https://www.slideshare.net/kthcorp/h3-2012-15042338)
- [작은 오픈소스 프로그램을 분석해서 실력을 쌓아 나간 사례](https://okky.kr/article/408595)
- [소스코드 분석에 왕도는 있다?의 댓글](https://kldp.org/node/67385#comment-295224)
- [Contributing to pandas documentation for the first time - lessons from open source](https://www.youtube.com/watch?v=qGPaRTG17ts)
- [곽중선님의 오픈소스 분석 가이드](https://www.facebook.com/sunny.kwak.90/posts/372869479504711)