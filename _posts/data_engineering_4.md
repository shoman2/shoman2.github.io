# SQL Data Types

SQL Data Types

- Numeric
  - bit
  - tinyint
  - smallint
  - int
  - bigint
  - decimal
  - numeric
  - float
  - real
- Date/TIme
  - DATE - YYYY-MM-DD
  - TIME
  - DATETIME
  - TIMESTAMP
  - YEAR
- Character/String
  - CHAR
  - VARCHAR
  - TEXT
- Unicode Character/String
- Binary
- Miscellaneous
- Boolean

# Relational DB

- DB = Organized Collection of DATA
  - ex) financial records, medical records, inventories
  - most common type ? Relational Database
- Relational DB
  - 2차원 형태(테이블)로 데이터를 표현
  - RDB를 수정 및 생성하는 시스템을 RDBMS라고 부른다
- 특징?
  - 모든 데이터를 2차원 테이블로 표현
  - Row 와 컬럼으로
  - 테이블들은 상호 관계를 갖게 된다 (Entity Relationship) 모델
  - Columns (Variables), Rows(Observations/Records)
  - **Normalization(Reduce Redundancy)**
    - Design Technique으로 볼 수도 있음
  - 대표적인 DB Types
    - MySQL, PostgresSQL, MariaDB

# AWS - RDS에 연결하기

1. Free Tier로 가입해서 셋팅 가능하다.
2. Public Availability Open 해줘야하고
3. Inbound 규칙에 MySQL 또한 add rule 해줘야 정상작동함
4. 물론 사전적으로 AWS CLI셋팅을 해야하며
5. 그 이후 Endpoint 주소로 연결 후 admin아이디와 비번 치면 바로 접속되더라
6. MySQL CLI 명령어
   1. mysql -h [ENDPOINT 주소] -P [포트번호 3306] -D [직접적속 DB명] -u [ADMIN ID] -p
   2. 비밀번호입력 하면 접속됨



# ERD (Entity Relationship Diagram)

엔터티 관계 다이어그램은 데이터 모델링 설계 과정에서 사용하는 모델로, 약속된 기호를 이용하여 데이터 베이스 구조를 쉽게 이해 할 수 있다.

**ERD의 기본 요소 3가지**

- Entities: 사람, 장소, 물건, 사건, 개념등 어떠한 개체
- Attributes: 엔터티의 속성(사람: 성, 이름, 성별, 나이 등)
- Relationships: 엔터티간의 관계

**Symbols and Notations**

![image-20200823212450881](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20200823212450881.png)

* **결국 각 테이블 별 관계를 파악하는게 주된 목적**

![image-20200823212700578](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20200823212700578.png)

# Primary Key & Unique Key

- 프라이머리 키는 테이블에 단 하나밖에 없는 유니크한 구별 값을 의미 (Unique & Not Null)
- Foreign Key는 다른 테이블의 키 값 (Can contain more than one, Null values OK)

- Unique Key는 또 다른 인덱싱 키 (Can accept NULL value, More than one unique key)