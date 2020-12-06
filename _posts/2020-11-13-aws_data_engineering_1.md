---
layout: post
title:  "Data Engineering - AWS 프리티어 사용해보기 "
subtitle:   "AWS Set Up - Free Tier"
categories: data
tags: engineering
comments: true
---
# AWS 프리티어 구축 (for Data Engineering) 

#### 1.AWS 가입 후 둘러보기

![image-20200818211137479](https://shoman2.github.io/assets/img/image-20200818211137479.png)

여러가지 서비스들이 보인다. 특히 EC2, S3 정도는 눈에 익혀놔야한다. 무조건 쓰기 때문에..



#### 2.AWS CLI 둘러보기

AWS사이트에 들어오지 않고 AWS Command Line Interface 를 통해서 AWS와 통신하는 방법에 대해 알아본다.

1)AWS CLI Installation 명령어를 아래와 같이 날린다.

```shell
pip3 install awscli --upgrade --user
```



2)Console 창으로 들어가서 IAM을 설정한다. (유저 추가) / 권한은 어드민으로 주고 생성

3)iTerm켜서 CLI를 잘 설치해줘야한다. 갠적으로 bash가 아닌 zsh를 사용해서 좀 애를 먹었다.

https://docs.aws.amazon.com/cli/latest/userguide/install-macos.html

요기를 참조해서 따라하니 다행히 설치 성공해서 CLI를 통해 AWS와 송신이 가능하게 셋팅이 완료

