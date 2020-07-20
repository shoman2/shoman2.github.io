---
layout: post
title:  "Github Profile Readme로 프로필 꾸미기"
subtitle: "Github Profile Readme로 프로필 꾸미기"
categories: development
tags: tools
comments: true
---

- Github Profile을 이쁘게 디자인할 수 있는 방법에 대해 작성한 글입니다
	- 키워드 : Desing Github Profile using README.md, Add a README to Github Profile
- 생각보다 등록하는 방법이 간단해서, 가볍게 정리합니다


---


### Github Profile
- 7월 9일경 페이스북에서 Github에 새로운 Feature가 나왔다는 글을 봤습니다
- 글은 [Design Github profile using README.md](https://dev.to/web/design-github-profile-using-readme-md-8al)로, Github의 우측 공간에 자신의 프로필을 올릴 수 있는 기능이 나왔습니다!
- 간단히 제 최종 프로필을 보여드리면 아래와 같이 우측 공간에 프로필이 생깁니다
	- 제 Github : [https://github.com/zzsza](https://github.com/zzsza)

<img src="https://www.dropbox.com/s/7kcp1mt6jzyk60v/Screenshot%202020-07-10%2023.11.03.png?raw=1">



<br />

---

<br />

### 등록하는 방법
- 자신의 Github로 이동합니다

<img src="https://www.dropbox.com/s/3uq0i8acbw6cesn/Screenshot%202020-07-10%2021.24.02.png?raw=1">


- Create a new repository 누른 후, Repo 이름을 자신의 닉네임으로 설정합니다
	- 닉네임을 입력하면 "You found a secret!, zzsza/zzsza is a special repository that you can use to add a README.md to your Github profile. Make sure it's public and initializa it with a README to get started" 라는 문구가 나타납니다
	- 이 Repo는 Private하면 (당연하게도) 보이지 않겠네요!

<img src="https://www.dropbox.com/s/tbahv71fffzr0kt/Screenshot%202020-07-10%2021.24.22.png?raw=1">


- Repo를 만들면(README를 init한다고 체크하면) Hi there!이란 내용의 README가 생성됩니다
	- zzsza/zzsza is a special repository. Its README will appred on your public profile! Send feedback이란 문구가 우측에 보이네요

<img src="https://www.dropbox.com/s/86bwofzj0t6gkz9/Screenshot%202020-07-10%2021.24.50.png?raw=1">


- 이제 Github로 이동하면 아래처럼 보입니다

<img src="https://www.dropbox.com/s/ehv4nadfmtk7i55/Screenshot%202020-07-10%2021.25.22.png?raw=1">


<br />

---

<br />

### Github Profile 꾸미기
- 이제 자유롭게 Github Profile의 README를 수정하시면 됩니다!
- 자신이 표현하고 싶은 내용을 담으면 되는데, 자신의 경력이나 호기심있는 부분을 올릴 수 있습니다
- 관련 레포가 조금씩 생기고 있는데, 
	- 자신의 현재 상태를 알려주는 personal dashboard인 [Github Now](https://github.com/beyondcode/github-now) : 스포티파이와 연동하거나, 구글 캘린더와 연동 가능
- 저는 제 Github에 방문한 사람 수가 궁금해서 [hits](https://hits.seeyoufarm.com/)를 붙여봤습니다
	- 다른 hits도 있지만 이 hits는 [TODAY/TOTAL]을 표현해서, 더 자세한 정보를 알 수 있습니다
	- hits 홈페이지로 이동한 후, URL을 입력해줍니다
	- 그 후 MARKDOWN 아래에 있는 Copy를 눌러주세요

	<img src="https://www.dropbox.com/s/u2h7jl2ymlci9td/Screenshot%202020-07-10%2021.46.46.png?raw=1">
	
- `div align=center`를 사용해서 가운데에 위치하도록 했습니다

	```
	<div align=center>
	
	[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzzsza)](https://hits.seeyoufarm.com) 
	
	</div>
	```	

- 이제 마치 싸이월드처럼 제 프로필에 방문한 사람 수를 알 수 있습니다!(싸이월드 추억..)

<img src="https://www.dropbox.com/s/p8oriovo556rnps/Screenshot%202020-07-10%2021.50.26.png?raw=1">


- 그 외에도 이미지 추가, Table 표현도 가능합니다
	- Table 생성은 [Tables Generator](https://www.tablesgenerator.com/markdown_tables)에서 쉽게 생성할 수 있습니다 

- 또한, [shields.io](https://shields.io/)에서 Badge를 생성할 수 있습니다
	- [Simple Icons](https://simpleicons.org/)에 있는 아이콘을 쉽게 사용할 수 있습니다

[![Tech Blog Badge](http://img.shields.io/badge/-Tech%20blog-black?style=flat-square&logo=github&link=https://zzsza.github.io/)](https://zzsza.github.io/)

[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/seong-yun-byeon-8183a8113/)](https://www.linkedin.com/in/seong-yun-byeon-8183a8113/)

[![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://www.youtube.com/c/kyleschool)](https://www.youtube.com/c/kyleschool)

[![Facebook Badge](https://img.shields.io/badge/facebook-1877f2?style=flat-square&logo=facebook&logoColor=white&link=https://www.facebook.com/zzsza)](https://www.facebook.com/zzsza)


[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:snugyun01@gmail.com)](mailto:snugyun01@gmail.com)


- img.shields.io/badge/ 뒤에 `작성할 단어-색상?스타일옵션` 이렇게 뱃지를 생성할 수 있습니다
- 색상 코드는 주로 [simpleicons](https://simpleicons.org/)에서 참고했고, [shields](https://shields.io/)에서 생성할 수 있습니다

- Badge 코드

	
	```
	[![Tech Blog Badge](http://img.shields.io/badge/-Tech%20blog-black?style=flat-square&logo=github&link=https://zzsza.github.io/)](https://zzsza.github.io/)
	
	[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/seong-yun-byeon-8183a8113/)](https://www.linkedin.com/in/seong-yun-byeon-8183a8113/)
	
	[![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://www.youtube.com/c/kyleschool)](https://www.youtube.com/c/kyleschool)
	
	[![Facebook Badge](https://img.shields.io/badge/facebook-1877f2?style=flat-square&logo=facebook&logoColor=white&link=https://www.facebook.com/zzsza)](https://www.facebook.com/zzsza)
	
	
	[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:snugyun01@gmail.com)](mailto:snugyun01@gmail.com)
	
	```

<br />

---

<br />

### 프로필이 길면 좋을까?
- 프로필이 자세하면 Github Repo가 덜 보일텐데, 이게 좋을까? 싶은 생각에 테스트했습니다
- 6줄 정도면 이런 느낌이고

<img src="https://www.dropbox.com/s/uekj5hlpegm7tom/Screenshot%202020-07-10%2021.58.32.png?raw=1">

<br />


- 36줄은 이런 느낌입니다

<img src="https://www.dropbox.com/s/tzttka8etyooen6/Screenshot%202020-07-10%2022.00.50.png?raw=1">

<br />

- 블로그 프로필에 있는 글을 복사해 프로필이 더 길면 어떤 모습인지 확인했습니다
	- 너무 길면 Github Repo가 보이지 않아, 적당해야 할 것 같습니다

<img src="https://www.dropbox.com/s/rhf4kmihjnwz1tw/Screenshot%202020-07-10%2022.11.10.gif?raw=1">


<br />

---

<br />


### 최종본
- 저는 간단히 1줄로 저를 소개하고, 관심있는 내용을 추가했습니다! 아이콘을 활용했고, 최대한 심플하게 작성했습니다
	- 이메일은 Github Main 왼쪽에 나와서 추가적으로 작성하지 않고, 아이콘으로 대체했습니다(알록달록 이쁘네요..!)
	- 제가 작성한 형태를 가져가고 싶으시면, 자유롭게 제 repo를 fork하셔서 코드를 가져가도 좋을 것 같아요 :)

<img src="https://www.dropbox.com/s/7kcp1mt6jzyk60v/Screenshot%202020-07-10%2023.11.03.png?raw=1">


- 포트폴리오로 사용하실 분들은, 더 자세히 작성하셔도 좋을 것 같습니다!
	- 한국어/영어 버전을 만들어서 누를 수 있게 해도 좋을 것 같네요 :)
- 앞으로 더 창의적인 Github Profile이 나올 것 같네요!



- 20년 7월 20일
	- 다양한 분들의 프로필을 보다가, Github Stats도 발견했습니다!
	- [Github Stats Card](https://github.com/anuraghazra/github-readme-stats)에서 확인할 수 있습니다
	- username을 수정해주시면 됩니다!
	
	```
	[![Anurag's github stats](https://github-readme-stats.vercel.app/api?username=username)](https://github.com/anuraghazra/github-readme-stats)
	```

[![kyle's github stats](https://github-readme-stats.vercel.app/api?username=zzsza)](https://github.com/anuraghazra/github-readme-stats)
