---
layout: post
title:  "Github Action 사용법 정리"
subtitle: "Github Action 사용 방법 및 크롤링 자동화"
categories: development
tags: devops
comments: true
---
	 
- Github Action 사용법 및 cron 사용 방법에 대해 정리한 글입니다
	- Github Action으로 YES24 IT 신간을 파이썬으로 크롤링 후 Issue에 업로드하는 예제가 있습니다
		- [Github Action with Python](https://github.com/zzsza/github-action-with-python)
	- Github action with cron, Github action python 

---

### Github Action
- [공식 홈페이지](https://github.com/features/actions), [공식 문서](https://help.github.com/en/actions)   
- 소프트웨어 workflow를 자동화할 수 있도록 도와주는 도구
- Workflow의 대표적인 예
	- 1) Test Code
		- ex) 특정 함수의 return 값이 어떻게 나오는지 확인하는 테스트 코드
		- ex) df의 타입이 pd.DataFrame이 맞는가?
		- ex) value1에 특정 값이 들어가는가?
		- 쿼리를 날리고 데이터가 맞는지 정합성 체크하는 것도 일종의 테스트
	- 2) 배포
		- 서버에 새로운 기능, 버전 등을 배포
	- 3) 기타 자동화하고 싶은 스크립트
		- 주기적으로 데이터를 수집해 처리
	- 4) 다양한 파이썬 버전에서 실행되는지 확인
- CI나 CD는 아래 글을 참고해도 좋음
	- [Google Cloud Build를 사용한 CI/CD
](https://zzsza.github.io/gcp/2020/05/09/google-cloud-build/)
- 가격
	- Public repo : 무료, 단 limit이 있음
	- Private repo : [링크](https://help.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-billing-for-github-actions) 참고
- 사용할 수 있는 한도
    - Workflow는 하나의 Repo에 최대 20개까지 등록 가능
    - Workflow 안에 존재하는 Job은 6시간동안 실행될 수 있고, 초과시 자동으로 중지됨
    - Github Free는 Storage 한도 500MB, 월에 실행 시간 2,000분
		- Github Pro는 Storage 한도 1GB, 월에 실행  시간 3,000분

		
<br />

---

<br />


### Github Action Core 개념
- Github Action을 이해하기 위해서 알아야 하는 개념은 **Workflow, Event, Job, Step, Action, Runner** 등이 있음 
- 1) Workflow
	- 여러 Job으로 구성되고, Event에 의해 트리거될 수 있는 자동화된 프로세스
	- 최상위 개념
	- Workflow 파일은 YAML으로 작성되고, Github Repository의 `.github/workflows` 폴더 아래에 저장됨
- 2) Event
	- Workflow를 Trigger(실행)하는 특정 활동이나 규칙
	- 예를 들어 다음과 같은 상황을 사용할 수 있음
		- 특정 브랜치로 Push하거나
		- 특정 브랜치로 Pull Request하거나
		- 특정 시간대에 반복(Cron)
		- Webhook을 사용해 외부 이벤트를 통해 실행
	- 자세한 내용은 [Events that trigger workflows](https://help.github.com/en/actions/reference/events-that-trigger-workflows#external-events-repository_dispatch) 참고
- 3) Job
	- Job은 여러 Step으로 구성되고, 가상 환경의 인스턴스에서 실행됨
	- 다른 Job에 의존 관계를 가질 수 있고, 독립적으로 병렬 실행도 가능함
- 4) Step
	- Task들의 집합으로, 커맨드를 날리거나 action을 실행할 수 있음
- 5) Action
	- Workflow의 가장 작은 블럭(smallest portable building block)
	- Job을 만들기 위해 Step들을 연결할 수 있음
	- 재사용이 가능한 컴포넌트
	- 개인적으로 만든 Action을 사용할 수도 있고, Marketplace에 있는 공용 Action을 사용할 수도 있음
		- [Github Marketplace](https://github.com/marketplace?type=actions)와 [Github Actions Repository](https://github.com/actions/)에서 확인 가능
- 6) Runner
	- Gitbub Action Runner 어플리케이션이 설치된 머신으로, Workflow가 실행될 인스턴스
	- Github에서 호스팅해주는 Github-hosted runner와 직접 호스팅하는 Self-hosted runner로 나뉨
	- Github-hosted runner는 Azure의 Standard\_DS2\_v2로 vCPU 2, 메모리 7GB, 임시 스토리지 14GB
		- [Self-hosted runners](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/adding-self-hosted-runners) 참고



<br />

---

<br />

   
## Github Action 생성하는 흐름
- 1) 코드 작성
- 2) 코드 작성 후, Workflow 정의
- 3) 정상 작동하는지 Test


### Workflow 정의하기
- 기본적인 방법 : `.github/worfklows` 폴더 안에 `.yml` 파일을 생성 => 템플릿 활용하면 좋음
- Github Repo에서 Actions 클릭
	- <img src="https://www.dropbox.com/s/xuefekjnxhpopu4/Screenshot%202020-05-30%2015.08.15.png?raw=1">
	- <img src="https://www.dropbox.com/s/9xayfkoe50ypmg3/Screenshot%202020-05-30%2015.08.46.png?raw=1">
- Set up this workflow 클릭하면 간단한 workflow 생성할 수 있음
	- <img src="https://www.dropbox.com/s/w2710nhfvigb433/Screenshot%202020-05-30%2015.09.40.png?raw=1">

- yml 파일 예시	
	- Master 브랜치에 Push 또는 Pull Request가 올 경우 실행되는 CI란 이름을 갖는 Workflow

	```
	name: CI
	
	on:
	  push:
	    branches: [ master ]
	  pull_request:
	    branches: [ master ]
	
	jobs:
	  build:
	    runs-on: ubuntu-latest
	
	    steps:
	    - uses: actions/checkout@v2
	
	    - name: Run a one-line script
	      run: echo Hello, world!
	
	    - name: Run a multi-line script
	      run: |
	        echo Add other actions to build,
	        echo test, and deploy your project.
	```    
	
- name : Workflow의 이름을 지정
- on

	```
	on:
	  push:
	    branches: [ master ]
	  pull_request:
	    branches: [ master ]
	```
	    
	- Event에 대해 작성하는 부분
	- 어떤 조건에 Workflow를 Trigger 시킬지
	- push(Branch or Tag), pull_request, schedule을 사용할 수 있음
		- 단, 다른 CI/CD 도구에 있는 즉시 실행 버튼은 없음(추후에 생길 수도?)
	- 단일 Event를 사용할 수도 있고, array로 작성할 수도 있음
	
		```
		on: push
		# 또는
		on: [pull_request, issues]
		```
	
- jobs

	```
	jobs:
		  build:
		    runs-on: ubuntu-latest
		
		    steps:
		    - uses: actions/checkout@v2
		    - name: Run a one-line script
		      run: echo Hello, world!

		    - name: Run a multi-line script
		      run: |
		        echo Add other actions to build,
		        echo test, and deploy your project.
	```

	- Workflow는 다양한 Job으로 구성됨
	- 여러 Job이 있을 경우, Default로 병렬 실행
	- build라는 job을 생성하고, 그 아래에 2개의 step이 존재하는 구조
	- runs-on은 어떤 OS에서 실행될지 지정
	- strategy - matrix 인자를 사용하면 어떤 파이썬 버전에서 테스트할지 확인할 수 있음. 자세한 내용은 밑 예제를 통해 확인할 수 있음
	- 위 예제에선 run할 때 env(환경 변수)를 설정하는 예제가 없는데, 밑에서 환경 변수를 등록하는 방법에 대해 안내할 예정
	- steps의 uses는 어떤 액션을 사용할지 지정함. 이미 만들어진 액션을 사용할 때 지정

 	
<br />

---

<br />

### Github Action with Python
- 간단한 예제
	- Workflow 예시 중 Python으로 작성된 것은 Publish Python Package, Python application, Python package 등이 있음
	- 여기선 **Python package**를 예시로 사용함
- Github Repository 생성 후, hello.py 생성 후 Push
	
	```
	# hello.py
	print("Hello world")
	```

- Github Repository에서 Actions 클릭 - New workflow - Continuous integration workflows 더보기 - Python package - Set up the workflow
	- <img src="https://www.dropbox.com/s/o61viwvro0edmh4/Screenshot%202020-06-06%2013.56.19.png?raw=1">

- 생성된 yaml 파일 아래에 pip install하는 부분과 pytest를 지우고, python3 hello.py 실행하도록 수정
    - 템플릿 코드를 조금만 수정함
    - <img src="https://www.dropbox.com/s/fk9igo6mb6wwn1t/Screenshot%202020-06-04%2023.30.07.png?raw=1">
	- python3 hello.py를 다양한 파이썬 버전에서 실행함
		- strategy: matrix: python-version에서 3.5, 3.6, 3.7, 3.8를 지정함
		
		```
		strategy: 
			matrix: 
				python-version: [3.5, 3.6, 3.7, 3.8]
		```	

- Master로 Push하면 Latest commit 왼쪽에 노란색 아이콘이 생김. 이건 현재 작업 중임을 의미함
    - <img src="https://www.dropbox.com/s/kcvvq91nfltnarw/Screenshot%202020-06-04%2023.38.53.png?raw=1">
- 아이콘(노란색 아이콘 또는 초록색 체크)을 클릭하면 작업 상태를 확인할 수 있음
    - <img src="https://www.dropbox.com/s/ho00pazf0gr180w/Screenshot%202020-06-04%2023.40.03.png?raw=1">
- Details를 클릭하면
	- 로그를 확인할 수 있음
	- <img src="https://www.dropbox.com/s/3lufc3oxkyicrla/Screenshot%202020-06-06%2014.09.20.png?raw=1">

	
---	

- 간단하게 Python 스크립트를 실행한다면, Python application을 사용해도 충분함
	- <img src="https://www.dropbox.com/s/edigs0wz3chcrv1/Screenshot%202020-06-06%2014.01.46.png?raw=1">
	
	```
	name: Python application
	
	on:
	  push:
	    branches: [ master ]
	  pull_request:
	    branches: [ master ]
	
	jobs:
	  build:
	
	    runs-on: ubuntu-latest
	
	    steps:
	    - uses: actions/checkout@v2
	    - name: Set up Python 3.7
	      uses: actions/setup-python@v2
	      with:
	        python-version: 3.7
	    - name: Install dependencies
	      run: |
	        python -m pip install --upgrade pip
	        pip install flake8 pytest
	        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	    - name: Lint with flake8
	      run: |
	        # stop the build if there are Python syntax errors or undefined names
	        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	    - name: Test echo
	      run: |
	        echo "hi" 
	    - name: Test python -h
	      run: |
	        python3 -h
	```    

<br />

---

<br />

### YES24의 IT 신간 도서 40개를 주기적으로 확인하는 Github Action 만들기
- [Github Action with Python](https://github.com/zzsza/github-action-with-python)에서 코드를 확인할 수 있음
	- 매일 오전 9시에 YES24 IT 신간 40개를 찾아, 해당 책과 가격을 Github Issue에 업로드
	- <img src="https://www.dropbox.com/s/tb0g3k1osfn7vqe/Screenshot%202020-06-04%2023.59.05.png?raw=1">
	- <img src="https://www.dropbox.com/s/rio9vo7g2ok6jk2/Screenshot%202020-06-04%2023.59.36.png?raw=1">
- 코드는 매우 간단해서, 코드보단 Action Workflow에 대해 다룸

	```
	name: yes24_crawler
	
	on:
	  schedule:
	    - cron: '0 0 * * *'
	
	jobs:
	  build:
	    runs-on: ubuntu-latest
	    steps:
	    - uses: actions/checkout@v2
	    - name: Set up Python
	      uses: actions/setup-python@v2
	      with:
	        python-version: 3.7
	    - name: Install dependencies
	      run: |
	        python -m pip install --upgrade pip
	        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	    - name: Run main.py
	      run: |
	        python main.py
	      env:
	        MY_GITHUB_TOKEN: ${{ secrets.MY_GITHUB_TOKEN }}
	```
	
- schedule의 cron에 cron 표현을 등록. UTC 시간대라 한국 시간으론 9시를 의미함
	- 단, 테스트해보니 약간의 지연 시간이 있음(이건 사용량이 많아서 그럴수도..?)
- requirements.txt에 있는 패키지를 설치하고 main.py를 실행함
	- 여기서 run 아래에 env 인자에 환경 변수를 넘겨줌
	- 환경 변수를 넘겨줄 때 Secrets에 저장함	

<br />

---

<br />

### Secret Key 저장하기
- 클라우드의 저장소에 데이터를 저장하거나, Dockerhub에 접근하는 등을 할 때 고유한 Key가 필요할 수 있음
	- 이런 경우 Secrets에 등록해서 사용할 수 있음
- 아래 흐름은 위에 있는 YES24 크롤링처럼 Issue에 작성하는 경우의 예고, Github Access Token이 필요하지 않는 작업이면 2)부터 확인하면 됨
- 1) Github Access Token 생성하기
	- Github Issue를 남기기 위해 Github Access Token이 필요함
	- <img src="https://www.dropbox.com/s/m38bj7yd5b5465y/Screenshot%202020-06-06%2014.06.16.png?raw=1">
	- Settings - Developer settings - Personal access tokens - Generate new token 클릭
	- 체크박스에 필요한 권한 체크. repo, workflow 설정 후 저장
	- Value 복사하기
- 2) Secret Key 등록하기
	- <img src="https://www.dropbox.com/s/daxbpcnjwybi5hm/Screenshot%202020-06-06%2014.06.53.png?raw=1">
	- Repository의 Settings - 왼쪽에 Secrets 클릭
	- New secret 클릭하고 Name과 Value 입력하기
	- Repository access 쪽으로 가면, access policy를 정할 수 있음
	- workflow에서 사용할 땐 ${{ secrets.이름 }} 으로 사용하고, python에선 os.getenv(변수 이름)으로 사용
- 더 자세한 내용은 [Creating and storing encrypted secrets](https://help.github.com/en/actions/configuring-and-managing-workflows/creating-and-storing-encrypted-secrets) 참고
	
	
	
<br />

---

<br />

	
### Job의 의존 관계 설정하기
- yaml 파일에서 jobs쪽에 needs를 사용해 의존 관계를 설정할 수 있음

	```
	jobs:
	  job1:
	  job2:
	    needs: job1
	  job3:
	    needs: [job1, job2]
	```
	
- job1은 job2이 시작되기 전에 성공적으로 완료되어야 하고, job3은 job1과 job2이 모두 완료될 때까지 기다림	
- 더 자세한 내용은 [공식 문서](https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idneeds) 참고

<br />

---

<br />

### Artifact 저장하기
- Artifact는 job끼리 데이터를 공유하거나, workflow가 완료된 후 데이터를 저장할 수 있게 해줌
- [upload-artifact](https://github.com/actions/upload-artifact)와 [download-artifact](https://github.com/actions/download-artifact) Action을 사용함
- Workflow에서 생성된 파일을 Upload하고, 다른 job에서 사용할 경우 업로드된 파일을 Download해서 사용함
- 예제는 총 3개의 job으로 구성되고, job_1에서 math-homework.txt 파일을 생성하고 업로드
	- job_2는 job_1에서 업로드한 파일을 다운로드하고 연산한 후, 다시 업로드
	- job_3은 job_2에서 업로드한 파일을 다운받아 출력
- 코드에서 유심히 볼 부분은 uses: actions/upload-artifact@v1, uses: actions/download-artifact@v1 이 부분
	- uses에 어떤 액션을 사용할지 지정해서 인자만 입력해 활용함(라이브러리처럼)
- 결과
	- <img src="https://www.dropbox.com/s/o3323hd05h6ae9e/Screenshot%202020-06-06%2014.49.49.png?raw=1">

- workflow yaml 파일	
	
	```
	name: Share data between jobs
	
	on: [push]
	
	jobs:
	  job_1:
	    name: Add 3 and 7
	    runs-on: ubuntu-latest
	    steps:
	      - shell: bash
	        run: |
	          expr 3 + 7 > math-homework.txt
	      - name: Upload math result for job 1
	        uses: actions/upload-artifact@v1
	        with:
	          name: homework
	          path: math-homework.txt
	
	  job_2:
	    name: Multiply by 9
	    needs: job_1
	    runs-on: windows-latest
	    steps:
	      - name: Download math result for job 1
	        uses: actions/download-artifact@v1
	        with:
	          name: homework
	      - shell: bash
	        run: |
	          value=`cat homework/math-homework.txt`
	          expr $value \* 9 > homework/math-homework.txt
	      - name: Upload math result for job 2
	        uses: actions/upload-artifact@v1
	        with:
	          name: homework
	          path: homework/math-homework.txt
	
	  job_3:
	    name: Display results
	    needs: job_2
	    runs-on: macOS-latest
	    steps:
	      - name: Download math result for job 2
	        uses: actions/download-artifact@v1
	        with:
	          name: homework
	      - name: Print the final result
	        shell: bash
	        run: |
	          value=`cat homework/math-homework.txt`
	          echo The result is $value
	```

- 더 자세한 내용은 [Persisting workflow data using artifacts](https://help.github.com/en/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts) 참고

<br />

---

<br />


## 정리
- Github Action은 Workflow 자동화를 위한 도구
- CI/CD에서 활용할 수 있고, cron으로 스케쥴을 설정할 수 있음
	- 단, 즉시 실행 버튼은 없음
	- cron은 한국 시간 오전 9시로 지정하면 9시 15분쯤 실행되는 지연 현상이 발견됨
- 암호화된 변수를 사용하고 싶으면 Secrets에 저장 가능
- 의존 관계를 표시할 땐 job에서 needs 키워드 사용
- Data를 공유하고 싶은 경우 upload-artifact와 download-artifact 사용
- 관련 자료가 많이 없어 공식 문서를 잘 보면 좋음
	- [Github Actions Documentation](https://help.github.com/en/actions)
	- [Workflow syntax for GitHub Actions](https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- 누군가 만들어둔 action을 활용하면 쉽게 자동화 가능
	- [Github Actions Repository](https://github.com/actions)에서 확인 가능함
	- [upload-artifact](https://github.com/actions/upload-artifact), [download-artifact](https://github.com/actions/download-artifact), [cache](https://github.com/actions/cache) 등

<br />

---

<br />

### Reference
- [Github Actions Documentation](https://help.github.com/en/actions)
- [Workflow syntax for GitHub Actions](https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [An Introduction to Github Actions](https://gabrieltanner.org/blog/an-introduction-to-github-actions)
- [자동으로 스터디 모집 글을 모아 알림을 주는 파이썬 크롤러 만들기(with Github Actions)](https://baek.dev/post/17/)