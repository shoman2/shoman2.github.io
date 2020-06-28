---
layout: post
title:  "Flask에서 Unit Test하기"
subtitle: "Flask에 Unit Test하기"
categories: development
tags: python
comments: true
---
	 
- Flask에서 Unit Test 하는 방법에 대해 작성한 글입니다

---

## Test
### 테스트 자동화의 중요성
- 시스템 테스트에서 가장 중요한 것은 테스트의 자동화
	- 사람이 직접 실행하는 매뉴얼 테스트만 거칠 경우, 사이드 이펙트가 생길 수 있음
- 테스트를 최대한 자동화해서 테스트가 반복적으로, 자주 실행될 수 있도록 해야하며 항상 정확하게, 빠지는 부분이 없도록 테스트가 실행되도록 하는 것이 굉장히 중요함
- 테스트 방법
	- 1) UI test / End-to-End test
	- 2) integration test
	- 3) unit test 

### UI test / End-To-End test
- UI Test는 시스템의 UI(User Interface)를 통해서 테스트하는 것
- 웹이라면 웹 브라우저를 통해 웹사이트에 접속하고, UI에 직접 입력하고 클릭하는 등을 통해 기능이 정상적으로 작동하는지 테스트
- 장점
	- 사용자가 실제로 시스템을 사용하는 방시과 가장 동일하게 테스트
- 단점
	- 시간이 가장 많이 소요되는 테스트
	- 프론트엔드 ~ 백엔드까지 모든 시스템을 실행시키고 연결해야 함
	- 자동화하기 가장 까다로움
- Selenium 같은 UI Test 프레임워크를 사용해 어느정도 자동화가 가능하지만 100% 자동화는 어려움
	- 특히 화면 렌더링에서 문제가 발생
- 단점 때문에 전체 테스트 중 대략 10% 정도만 UI test 방식을 통해 실행하는 것을 추천 ( 주로 마지막에 테스트 )

### Integration test
- 여태까지 미니터 API를 개발하며 해왔던 방식
- API 서버를 실행시키고 HTTP Request를 실행해 Response가 올바른지 파악
- 테스트하고자 하는 서버를 실제로 실행시키고 테스트 HTTP 요청을 실행해 테스트해보는 방식
- 하나의 시스템만 실행해서 UI test에 비해 테스트 설정이나 실행 시간이 더 짧고 간단
- 하지만 unit test에 비해 자동화에 걸리는 공수가 더 크고 실행 속도도 더 느릴 수 밖에 없음
- 전체 테스트 중 대략 20% 정도만 할당하는 것을 추천    

### Unit test
- 시스템을 테스트한다는 개념보다는 코드를 직접 테스트하는 개념
	- 즉, 코드로 코드를 테스트함
- 실행하기 쉬우며 실행 속도도 빠름
	- 디버깅도 비교적 쉬움
	- 함수 단위로 테스트해서 파악이 쉬울 수밖에 없음
- 단점은 함수 단위로 테스트하다보니 전체적인 부분을 테스트하기엔 제한적일 수 밖에 없음
	- 이런 단점을 integration test와 UI test를 통해 보완
- 전체 테스트의 70%를 unit test

### pytest
- 파이썬 내장 라이브러리인 uniitest보다 사용하기 직관적인 pytest를 사용할 예정
- 설치

	```
	pip3 install pytest
	```
	
- pytest에선 test_라고 되어있는 파일들만 테스트 파일로 인식하고, 함수도 test_라고 prefix가 있어야만 test할 함수로 인식함
- 예시 (test_multiply_by_two.py)
	
	```
	def multiply_by_two(x):
		return x * 2
		
	def test_multiply_by_two():
		assert multiply_by_two(4) == 7
	``` 
	
- 터미널에서 pytest를 입력해 실행하면 오류가 발생


### 미니터 API unit test
- 순서
	- test_endpoints.py에 unit test 코드 구현
	- 테스트용 데이터베이스 생성
- config.py

	```
	test_db = {
    'user': 'test',
    'password': '1234',
    'host': 'localhost',
    'port': 3306,
    'database': 'test_db'
	}
	
	test_config = {
	    "DB_URL": f"mysql+mysqlconnector://{test_db['user']}:{test_db['password']}@{test_db['host']}:{test_db['port']}/{test_db['database']}?charset=utf8"
	}
	``` 	

- Flask는 unit test에서 엔드포인트들을 테스트할 수 있는 기능을 제공함(test_client)
- pytest.fixture 데코레이터를 사용하면 같은 이름의 함수의 리턴값을 해당 인자에 넣어줌

	```
	@pytest.fixture
	def api():
	    app = create_app(config.test_config)
	    app.config['TESTING'] = True
	    api = app.test_client()
	
	    return api
	```

- test ping

	```
	def test_ping(api):
	    resp = api.get("/ping")
	    assert b'pong' in resp.data
	```
	
- test tweet
	- tweet을 생성하기 위해선 사용자가 있어야 함
	- 해당 사용자로 인증 절차를 거친 후, access token으로 tweet 엔드포인트를 호출해야 함

	```
	def test_tweet(api):
	    new_user = {
	        "email": "zzsza@naver.com",
	        "passowrd": "1234",
	        "name": "변성윤",
	        "profile": "test profile"
	    }
	    resp = api.post(
	        "/sign-up",
	        data=json.dumps(new_user),
	        content_type="application/json"
	    )
    	assert resp.status_code == 200

	    # get id fo the new uesr
	    resp_json = json.loads(resp.data.decode("utf-8"))
	    new_user_id = resp_json["id"]
	
	    # login
	    resp = api.post(
	        "/login",
	        data=json.dumps({"email": "zzsza@naver.com", "password": "1234"}),
	        content_type="application/json"
	    )
	    resp_json = json.loads(resp.data.decode("utf-8"))
	    access_token = resp_json["access_token"]
	
	    # tweet
	    resp = api.post(
	        "/tweet",
	        data=json.dumps({"tweet": "Hello World"}),
	        content_type="application/json",
	        headers={"Authorization": access_token}
	    )
	    assert resp.status_code == 200
	
	    # tweet check
	    resp = api.get(f"/timeline/{new_user_id}")
	    tweets = json.loads(resp.data.decode("utf-8"))
	
	    assert resp.status_code == 200
	    assert tweets == {
	        "user_id": 1,
	        "timeline" : [
	            {
	                "user_id": 1,
	                "tweet": "Hello World"
	            }
	        ]
	    }
	``` 	
	
- test에서 사용된 데이터를 테스트 종료 후 삭제해줘야 다른 테스트에 영향을 끼치지 않음
	- pytest에선 setup_function과 teardown_function을 사용하면 됨

	```
	def setup_function():
	    ## Create a test user
	    hashed_password = bcrypt.hashpw(
	        b"test password",
	        bcrypt.gensalt()
	    )
	    new_users = [
	        {
	            'id'              : 1,
	            'name'            : '변성윤',
	            'email'           : 'zzsza@naver.com',
	            'profile'         : 'test profile',
	            'hashed_password' : hashed_password
	        }
	    ]
	    database.execute(text("""
	        INSERT INTO users (
	            id,
	            name,
	            email,
	            profile,
	            hashed_password
	        ) VALUES (
	            :id,
	            :name,
	            :email,
	            :profile,
	            :hashed_password
	        )
	    """), new_users)
	``` 	
	
	```
	def teardown_function():
	    database.execute(text("SET FOREIGN_KEY_CHECKS=0"))
	    database.execute(text("TRUNCATE users"))
	    database.execute(text("TRUNCATE tweets"))
	    database.execute(text("TRUNCATE users_follow_list"))
	    database.execute(text("SET FOREIGN_KEY_CHECKS=1"))
	```
	
### Reference
- [깔끔한 파이썬 탄탄한 백엔드](http://www.yes24.com/Product/goods/68713424)
