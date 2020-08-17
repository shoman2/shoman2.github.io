---
layout: post
title:  "Data Engineering Pipeline"
subtitle:   "Pipeline"
categories: data
tags: Engineering
comments: true
---
# 데이터 파이프라인 이란?

#### 1)데이터 파이프라인 이란 ?

데이터를 한 장소에서 다른 장소로 옮기는 것

- API 에서 DB로
- DB에서 다른 DB로
- DB에서 BI Tool로 시각화

#### 2)데이터 파이프라인이 필요한 경우

다양한 데이터 소스들로부터 많은 데이터를 생성하고 저장하는 서비스

- **데이터 사일로** 가 있는 경우 -- 마케팅, 어카운팅, 세일즈, 오퍼레이션 등 각 영역의 데이터가 서로 고립되어 있는 상황을 데이터 사일로라고 부름 
- 실시간 혹은 높은 수준의 데이터 분석이 필요한 비즈니스 모델인 경우
- 클라우드 환경으로 데이터 저장되는 경우

#### 3)데이터 파이프라인 예시

![image-20200817233047001](/Users/tkim29/github_blog/shoman2.github.io/assets/img/docs/image-20200817233047001.png)

#### 4)데이터 파이프라인 구축시 고려 사항

- Scalability: 데이터가 기하급수적으로 늘어났을때도 작동 여부
- Stability: 에러, 데이터 플로우 등 다양한 모니터링 및 관리 방안
- Security: 데이터 이동간 보안에 대한 리스크

# 자동화(Automation) 이해

#### 데이터 프로세싱에서 자동화란?

필요한 데이터를 추출, 수집, 정제하는 프로세싱을 최소의 사람 인풋으로 머신을 통해 운영하는 것을 의미

예) Spotify 데이터를 하루에 한번 API를 통해서 클라우드 데이터베이스로 가져오는 Task라면 사람이 직접 매번 작업하는 것이 아니라 크론탭 등 머신 스케쥴링을 통해 자동화 가능

#### 자동화를 위한 고려사항

<u>1)데이터 프로세싱 스텝들</u>

<u>2)에러 핸들링 및 모니터링 (얼마나 걸렸는지 리포트)</u>

<u>3)트리거/스케줄링 (얼마나 자주 긁어올 것인지)</u>

#### 1)데이터 프로세싱 스텝들

![image-20200817233901641](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20200817233901641.png)

- 각 스텝별로 어떻게 작동시킬 것인지 고민해야함

#### 2)에러 핸들링 및 모니터링

파이썬 로깅 패키지

```python
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.debug('This messege should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
```

Cloud Logging Systems

- AWS Cloudwatch
- AWS Data Pipeline Errors

#### 3)트리거 & 스케쥴링

![image-20200817234336611](/Users/tkim29/github_blog/shoman2.github.io/assets/img/image-20200817234336611.png)

