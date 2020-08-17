# Data Engineering 필요성

모든 비즈니스가 동일한 데이터 분석 환경을 갖출 수 없으며, 성장 단계에 따라 선택과 집중을 해야하는 분석환경이 다름

따라서 데이터 엔지니어링이 비즈니스 사이드와 잘 커뮤니케이션 하여 의사결정해야함

---

#### 페이스북과 같은 유저 경험이 중요한 비즈니스의 경우, 처음부터 데이터 시스템 구축이 성공의 열쇠가 됨.

**예시)** 

유저정보

- User ID
- Last Login

컨텐츠 정보

- Content ID
- Content Type
- Categories

유저액션

- Impression / Click
- Position in Feed
- Comment, Like, etc.

---

####  ECommerce는 마케팅/CRM/Logistics 데이터 분석을 통해 전략을 수립

예시)

- 마케팅 채널별 비용 데이터
- CRM 데이터
- 각종 물류 데이터

#### -->처음부터 모든 인력을 갖추고 분석 환경을 갖출 수는 없기 때문에 성장 단계 별로 필요한 분석 환경을 갖추는 것이 Key

# 데이터 아키텍쳐 구축 시 고려 사항

#### 1) 비즈니스 모델 상 가장 중요한 데이터는 무엇인가 ? 

![image-20200817225741201](/Users/tkim29/Library/Application Support/typora-user-images/image-20200817225741201.png)

#### 2) 데이터 거버넌스

 1) 원칙(Principle)

- 데이터를 유지 관리하기 위한 가이드
- 보안, 품질, 변경관리

 2) 조직(Organization)

- 데이터를 관리할 조직의 역할과 책임
- 데이터 관리자, 데이터 아키텍트

 3) 프로세스(Process)

- 데이터 관리를 위한 시스템
- 작업 절차, 모니터 및 측정

#### 3) 유연하고 변화 가능한 환경 구축

- 특정 기술 및 솔루션에 얽매여져 있지 않고 새로운 테크를 빠르게 적용할 수 있는 아키텍쳐를 만드는 것
- 생성되는 데이터 형식이 변화할 수 있는 것처럼 그에 맞는 툴들과 솔루션들도 빠르게 변화할 수 있는 시스템을 구축하는 것

#### 4) 실시간(Real-Time) 데이터 핸들링 가능한 시스템

- 밀리세컨 단위의 스트리밍 데이터가 됬건 하루에 한번 업데이트 되는 데이터가 됬건 데이터 아키텍쳐는 모든 스피드의 데이터를 핸들링 및 커버 해야함
  - Real Time Streaming Data Processing
  - Cronjob
  - Serverless Triggered Data Processing

#### 5) 시큐리티(Security)

- 내부와 외부 모든 곳에서부터 발생할 수 있는 위험 요소들을 파악하여 어떻게 데이터를 안전하게 관리할 수 있는지 아키텍쳐 안에 포함

#### 6) 셀프 서비스 환경 구축

- 데이터 엔지니어 한명만 액세스가 가능한 데이터 시스템은 <u>확장성</u>이 없는 데이터 분석환경이 되어버리기 때문에 분석가, 프론엔드 개발자 등의 사용자들을 고려하여 확장성을 갖춘 환경을 구축

# 데이터 시스템의 옵션들

#### 1) API의 시대

- 마케팅,CRM,ERP등 다양한 플랫폼 및 소프트웨어들은 API를 통해 데이터를 주고 받을 수 있는 환경을 구축하여 생태계를 생성

#### 2) RDB (관계형 DB)

- 데이터의 관계도를 기반으로 한 디지털 DB로 데이터의 저장을 목적으로 생겨남
- SQL이라고 하는 스탠다드 방식을 통해 자료를 열람하고 유지
- 현재 가장 많이 쓰고 있는 데이터 시스템 (매우 오래됨)

#### 3) NoSQL Database

- Not Only SQL 의 약자
- Unstructured, SchemaLess Data (비정형데이터) 다루는 DB
- Scale Horizaontally (예시 : 메신저/카톡 등)
- Highly scalable / Less expensive to maintain

#### 4) Distributed Storage System/MapReduce를 통한 병렬 처리

**스파크:**

- Hadoop의 진화 버전으로 빅데이터 분석 환경에서 리얼타임 데이터를 프로세싱하기에 더 최적화
- Java, Python, Scala를 통한 API를 제공하여 애플리케이션 생성
- SQL Query 환경을 서포트하여 분석가들에게 더 각광 받음

#### 5) Serverless Framework

- Triggered by http requests, db events, queuing services
- Pay as you use (항상 서버를 띄워놓지 않음)
- Form of functions
- 3rd party App 밀 다양한 API를 통해 데이터를 수집 정제하는데 유용

