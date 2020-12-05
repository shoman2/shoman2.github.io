---
layout: post
title:  "Data Engineering "
subtitle:   "NoSQL"
categories: data
tags: engineering
comments: true
---
# Data Engineering - NoSQL

#### 1)Not Only SQL - 관계형 데이터베이스와의 극명한 차이

##### **다이나믹 스키마**

- 구조를 정의하지 않고도 Documents, Key Values 등을 생성
- 각각의 다큐먼트는 유니크한 구조로 구성 가능
- 데이터베이스들마다 다른 Syntax
- 필드들을 지속적으로 추가 가능

##### Scalability

- SQL DBs are vertically scalable - CPU, RAM, SSD
- NoSQL DBs are horizontally scalable - Sharding / Partitioning

#### 2)Partitions?

##### 데이터 매니지먼트, 퍼포먼스 등 다양한 이유로 데이터를 나누는 일

- Vertical Partition: 테이블을 더 작은 테이블들로 나누는 작업으로써 노멀라이제이션 후에도 경우에 따라 컬럼을 나누는 파티션 작업을 수행함
- Horizaontal Partition: 스키마 또는 스트럭쳐 자체를 카피하여 데이터 자체를 Sharded Key로 분리. NoSQL에서 무조건 사용되는 형태

#### 3)AWS DynamoDB (NoSQL)

- Create TABLE in AWS CONSOLE
- Pip install boto3
