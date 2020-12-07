---
layout: post
title:  "Data Engineering - 기초도구 설명 ATOM & SHELL"
subtitle:   "Basic Set up"
categories: data
tags: engineering
comments: true
---
# Data Engineering 기초 Set up
#### 1. 텍스트 에디터 사용 (어떤걸 쓰던 자기가 편한게 장땡)
1)**ATOM** 1.50.0 for macOS - 처음 사용해보는데 경험을위해 한번 깔아보자.

- 구글에 Atom 검색하니 가장 상위에 뜸

- 다운로드 후 설치 (아래 이미지 참조, 컬러가 예쁘다.)

  ![image-20200818202946149](https://shoman2.github.io/assets/img/image-202008182029461491.png)

- 맥 터미널에서 codes라는 폴더를 생성(mkdir codes)후 atom . 을 입력하면 아톰이 자동으로 실행되면서 해당 디렉토리에서 작업을 진행 할 수 있게됨 

#### 2-1. UNIX 기본 커맨드 정리

- ls: 현재 경로안에 어떤 파일이 있는지
- mkdir: 폴더 만들기
- cd [폴더명]: 폴더명으로 들어가기
- pwd : 현재 경로 체크
- cd .. : 이전 경로로 이동
- cp : 파일 복사
- rm : 파일 지우기

- cp -r : 폴더지우기
- cat : 해당 파일 전체 출력하기

#### 2-2. Shell Scripting 기본

```python
import sys


def main():

    print(sys.argv[1])

if __name__ == '__main__':
    main()

#example.py 파일로 저장하기
```

```shell
#!/bin/bash
python3 example.py 1
python3 example.py 2

#run.sh로 저장하기
```

```shell
./run.sh #실행시 permission denied가 나온다.

chmod +x run.sh #이렇게 해서 권한제한을 풀어주고

./run.sh #재실행하면 example.py 1 , example.py 2 모두 순차적으로 실행된다.
```

<u>AWS S3 Shell Command 예시</u>

 ```shell
#!/bin/bash
rm *.zip
zip lisztfever.zip -r *

aws s3 rm s3://areha/lisztfever/lisztfever.zip
aws s3 cp ./lisztfever.zip s3://areha/lisztfever/lisztfever.zip
aws lambda update-function-code --function-name lisztfever --s3-bucket areha --s3-key lisztfever/lisztfever.zip
 ```

