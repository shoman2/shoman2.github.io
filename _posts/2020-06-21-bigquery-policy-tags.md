---
layout: post
title:  "BigQuery Column 보안을 위한 Policy Tags 사용법"
subtitle: "BigQuery Column 보안을 위한 정책 태그 사용법"
categories: gcp
tags: BigQuery
comments: true
---

- BigQuery Column 수준의 보안을 가능하게 해주는 Policy Tags 사용 방법에 대해 작성한 글입니다
	- 글 작성하는 시점(20년 6월 21일)에 아직 Beta 기능이므로 추후에 기능이 변경될 수 있습니다

---


### BigQuery Policy Tags
- 한국어론 정책 태그라고 번역됨
- Policy Tags는 [Google Cloud Platform Data Catalog](https://cloud.google.com/data-catalog?hl=ko) 서비스의 일부
	- Data Catalog는 메타데이터 관리 서비스
- 필요한 상황
	- 특정 열을 권한에 따라 접근을 다르게 하고 싶은 경우
	- 데이터 보안이 필요한 경우
		- 예시 1) 직원 데이터를 저장한 경우 연봉 같은 민감한 정보는 인사팀 또는 연봉을 결정하는 임원급 이상에게만 볼 수 있도록 설정
		- 예시 2) 개인 정보 취급자에겐 개인 정보를 접근할 수 있도록 설정
- **민감한 컬럼에 세분화된 액세스 권한 부여함**
- 계층을 나눠서 권한 설정 가능
	- <img src="https://www.dropbox.com/s/u02e1hfkqnnfwgu/Screenshot%202020-06-21%2011.55.00.png?raw=1">
	- High, Medium, Low 유형으로 데이터를 나누고 그 아래에 어떤 데이터가 들어간지 정의함


<br />

---

### BigQuery Policy Tags 설정하는 방법
- 1) Policy Tag 정의(=생성)
- 2) BigQuery Column에 Policy Tag 할당
- 3) Policy Tag를 사용할 유저에게 권한 부여


<br />

---

### 1) Policy Tag 정의
- [Google Cloud Platform의 콘솔](https://console.cloud.google.com/)에서 Data Catalog 클릭
	- [Data Catalog 링크](https://console.cloud.google.com/marketplace/details/google/datacatalog.googleapis.com)
	- <img src="https://www.dropbox.com/s/3nj1iu7921ojwir/Screenshot%202020-06-21%2012.04.00.png?raw=1" width="200" height="400">
- 처음 접속시 API 사용 설정 필요
	- <img src="https://www.dropbox.com/s/3m08h1yaiybobji/Screenshot%202020-06-21%2012.05.08.png?raw=1">
- Data Catalog 메인에서 "정책 태그" 아래에 있는 정책 태그 생성 및 관리 클릭
	- 참고로 이 화면에서 가장 많이 쿼리된 BigQuery 테이블 및 뷰를 볼 수 있음(많이 사용되는 테이블 확인)
	- <img src="https://www.dropbox.com/s/xz0fsrjrvoyztcw/Screenshot%202020-06-21%2012.05.51.png?raw=1">
- Policy Tag UI
	- <img src="https://www.dropbox.com/s/7hyp9h874pc0yfv/Screenshot%202020-06-21%2012.07.14.png?raw=1">
- 만들기 클릭하면 Policy Tag를 생성할 수 있음
	- <img src="https://www.dropbox.com/s/6409llilvyhf7i4/Screenshot%202020-06-21%2012.10.50.png?raw=1">
- 위 이미지에선 크게 High / Medium이란 최상위 Policy Tag를 생성하고, High 아래에 name, email Policy Tag를 생성했음
	- email 아래에 또 하위 Policy Tag를 생성할 수 있음
	- 여기선 간단하게 name만 사용할 예정
- 만들어진 Policy Tags 확인
	- 메타데이터와 위에서 설정한 정책들을 볼 수 있음
	- <img src="https://www.dropbox.com/s/ujebhxkw9sg9ten/Screenshot%202020-06-21%2012.13.31.png?raw=1">
- "액세스 제어 적용"을 클릭해 Policy Tag가 적용되면 특정 권한을 가진 사람만 접근 되도록 설정



<br />

---


### 2) BigQuery Column에 Policy Tag 할당
- Policy Tag를 적용하는 방법은 빅쿼리 스키마를 수정하는 방법과 같음
- 사용할 Table로 이동해 데이터 확인
	- user 테이블은 channel_id, user_id, channel_name, user_name이 저장됨
	- user_name만 Policy Tag 적용할 예정ㅇ
	- <img src="https://www.dropbox.com/s/coty8t9rp3jirp5/Screenshot%202020-06-21%2012.16.44.png?raw=1">
- 테이블을 클릭하고 "스키마"로 이동
	- 스키마쪽에 가면 "정책 태그"라는 열이 보임
	- <img src="https://www.dropbox.com/s/f429ueebbhyj78h/Screenshot%202020-06-21%2012.17.18.png?raw=1">
- 스키마 수정 클릭
	- <img src="https://www.dropbox.com/s/r2vsim00b1wjc7q/Screenshot%202020-06-21%2012.18.06.png?raw=1">
- 사용할 필드를 클릭하면 정책 태그 추가 버튼이 활성화됨. 정책 태그 추가 버튼 클릭
	- 아까 저장된 Policy Tag들이 보임. name을 클릭하고 선택 - 저장
	- <img src="https://www.dropbox.com/s/2jnogdlxb40qhz3/Screenshot%202020-06-21%2012.18.59.png?raw=1">
- 다시 테이블의 스키마쪽으로 접근
	- 액세스 권한이 없으므로 1열이 제한되었다고 나옴(저도 권한 설정을 아직 안한 상태)
	- <img src="https://www.dropbox.com/s/upgbr5ociy27v6d/Screenshot%202020-06-21%2012.20.03.png?raw=1">
- 미리보기를 클릭하면 Policy Tag가 적용된 열은 보이지 않음
	- <img src="https://www.dropbox.com/s/ywxezyca495nf6f/Screenshot%202020-06-21%2012.21.08.png?raw=1">
- 강제로 쿼리를 날리려고 하면 에러 메세지 발생
	- User does not have permission to access policy tag "policy_tag_test: name" on column
	- <img src="https://www.dropbox.com/s/vfk0nzowqopognt/Screenshot%202020-06-21%2012.21.56.png?raw=1">
- 삭제는 BigQuery 웹 콘솔에서 스키마 수정 - 정책 태그 옆에 있는 x 버튼 클릭 후 저장하면 됨

	
<br />

---


	
### 3) Policy Tag를 사용할 유저에게 권한 부여
- 이제 BigQuery Column에 정책 태그를 할당했으니 유저에게 권한 부여가 필요함
- Policy Tag 관련해서 사용되는 Role은 딱 3개
	- ㄱ) roles/datacatalog.policyTagsAdmin(한국어 : Data Catalog 관리자)
	- ㄴ) roles/datacatalog.categoryAdmin(한국어 : 정책 태그 관리자)
	- ㄷ) roles/datacatalog.policyTagFineGrainedReader(한국어 : 세분화된 권한의 리더)
- ㄱ) roles/datacatalog.policyTagsAdmin(한국어 : Data Catalog 관리자)
	- Data Catalog의 모든 요소에 접근할 수 있는 권한
- ㄴ) roles/datacatalog.categoryAdmin(한국어 : 정책 태그 관리자)
	- Policy Tag 생성, 수정, 삭제 권한
	- 즉, **Policy Tag를 생성하고 할당하는 사람에게 필요한 권한** 
- ㄷ) roles/datacatalog.policyTagFineGrainedReader(한국어 : 세분화된 권한의 리더)
	- Policy Tag가 적용된 Column에 접근할 수 있는 권한
	- 즉, **Policy Tag가 적용된 열을 볼 수 있는 권한**
	- 이 권한은 Policy Tag마다 권한을 줄 수 있음
- 자세한 설명은 [Data Catalog 역할](https://cloud.google.com/iam/docs/understanding-roles?hl=ko#data-catalog-roles) 참고
- 아까 봤던 Policy Tag 페이지로 이동
	- Policy Tag를 체크하고 우측 정보 패널에서 "구성원 추가" 클릭 후, 세분화된 권한 설정
	- <img src="https://www.dropbox.com/s/dycfck8zcmc60gi/Screenshot%202020-06-21%2012.34.19.png?raw=1">
- name Policy Tag에 snugyun01@gmail.com이 접근 가능해짐
	- <img src="https://www.dropbox.com/s/6qjfen487ur0511/Screenshot%202020-06-21%2012.35.19.png?raw=1">
- email Policy Tag쪽은 유저를 설정하지 않았기 때문에 snugyun01@gmail.com은 권한이 없음
	- <img src="https://www.dropbox.com/s/uw6zw0wukma74s9/Screenshot%202020-06-21%2012.36.09.png?raw=1">
- 이제 다시 빅쿼리 user 테이블로 이동
	- Policy Tag가 적용되어 있지만 아까처럼 제한되었다는 메세지는 없음
	- <img src="https://www.dropbox.com/s/0me31xbadlugshk/Screenshot%202020-06-21%2012.37.16.png?raw=1">
- 쿼리를 날려도 정상적으로 나옴
	- <img src="https://www.dropbox.com/s/pbyvn60ts5ipakb/Screenshot%202020-06-21%2012.38.01.png?raw=1">
	
	

<br />

---
	
	
### Python에서 BigQuery Policy Tag 적용하기
- BigQuery 웹 UI에서 Policy Tag를 적용할 수 있지만, **WriteDisposition 옵션이 WRITE\_TRUNCATE일 경우 테이블을 지우고 다시 생성해서 스키마와 관련된 정보가 삭제됨**
- [Airflow BigQuery Operator](https://zzsza.github.io/mlops/2019/04/17/airflow-bigquery-operator/)에서 WRITE_TRUNCATE를 사용하면 Policy Tag를 한번 더 설정해야 함
	- 또는 WRITE_TRUNCATE를 쓰지 않고, 데이터만 DELETE하고 WRITE_APPEND하는 방법도 있지만 개인적으론  Policy Tag를 한번 더 할당하는 명시적인 방법을 선호함
- 20년 6월 현재 [google-cloud-bigquery](https://googleapis.dev/python/bigquery/latest/index.html) 1.25.0 버전부터 Policy Tags를 지원한다고 하지만, 예제도 없고 사용하기도 조금 번거로움(기존 스키마를 복사하고, SchemaField를 새롭게 만들고 append해서 반영하는 방식)
	- google-cloud-bigquery 대신 [google-api-python-client](https://github.com/googleapis/google-api-python-client)를 활용함
	- [BigQuery API](http://googleapis.github.io/google-api-python-client/docs/dyn/bigquery_v2.html)를 활용해 Request를 날림
	- 기존 Table의 스키마를 get하고, patch로 기존 데이터를 업데이트함
	- [tables.patch](https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/patch) 참고함
	- 이 때, Policy Tag 분류 ID와 Policy Tag ID를 알아야 하는데, 이건 Policy Tag UI에서 확인할 수 있음
	
	


<br />

---


### 정리하면
- Policy Tags(정책 태그)는 Data Catalog의 기능 중 하나로, 빅쿼리의 열 수준 보안을 가능하게 함
- Policy Tag 설정하는 방법
	- 1) Policy Tag 정의
	- 2) BigQuery Column에 Policy Tag 할당
	- 3) 유저에게 권한 부여
		- 데이터를 볼 권한은 roles/datacatalog.policyTagFineGrainedReader(한국어 : 세분화된 권한의 리더)를 부여하면 됨
- WriteDisposition 옵션에서 WRITE_TRUNCATE를 사용하면 저장된 Policy Tag가 사라지니 한번 더 설정해야 함


<br />

---


### 개선이 필요한 부분(개인 생각)
- Policy Tag를 사용하지 않다가 적용하면 기존에 SELECT * 이런 쿼리에서 에러가 발생할 수 있음
	- 그 사람이 볼 수 있는 권한을 가진 열들을 *로 표현할 수는 없나?
- 스키마쪽에도 특정 열을 권한 설정했다는 것을 보여주는 것보다, 아예 안 보이는게 좋지 않을까?
- 누가 이 Policy Tag 권한을 가지고 있는지는 보이지만, 어떤 열에 Policy Tag가 설정되어 있는진 직접 다 확인해야 함
- 가격 책정 문서의 불명확함
	- 가격 책정은 [BigQuery 가격 책정](https://cloud.google.com/bigquery/pricing?hl=ko)과 [Data Catalog 가격 책정](https://cloud.google.com/data-catalog/pricing?hl=ko)을 참고하라고 하는데, 설명을 계속 읽어도 와닿지 않음..
- 현재 베타니 추후엔 이런 내용은 개선될거라 믿습니다! :)


<br />

---


### Reference
- [BigQuery 열 수준 보안 소개](https://cloud.google.com/bigquery/docs/column-level-security-intro?hl=ko)
- [BigQuery 열 수준 보안으로 액세스 제한](https://cloud.google.com/bigquery/docs/column-level-security?hl=ko)
- [python-datacatalog](https://github.com/googleapis/python-datacatalog) : 지금은 아직 많이 발전하지 않았지만, 나중엔 이 라이브러리에서 Policy Tag를 관리하게 될 것 같네요