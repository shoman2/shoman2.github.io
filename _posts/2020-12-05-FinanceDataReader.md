---
layout: post
title:  "FinanceDataReader 패키지 "
subtitle:   "주가정보 시각화"
categories: data
tags: dl
comments: true
---
# FinanceDataReader Package

**FinanceDataReader 라는 엄청난 패키지가 어떤분이 개발하셨는지는 몰라도 재무데이터 모으는 나같은 사람에겐 실무에 큰 도움이 되었다. 진심 감사드린다.  이 패키지를 설치하고 임포트해서 간단한 주가 시계열 분석을 진행해보고자 한다. 나도 제대로 써보는 건 처음이라.. 그래도 누군가 이 포스트를 보고 도움이 되기를 간절히 바란다. 내가 하고픈 건, 원하는 주식 또는 상장ETF의 종가 Movement를 아래 그래프처럼 시각화 해보고자 한다.**


```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = (15,15)
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.formatter.limits"] = -10000, 10000

df.plot()

```


    <AxesSubplot:xlabel='Date'>

![png](https://github.com/shoman2/shoman2.github.io/assets/img/output_2_1.png)



**우선, 패키지를 임포트 하고 버젼을 확인해본다.**


```python
import FinanceDataReader as fdr

fdr.__version__

```


    '0.9.10'



**사용 가이드에 따라서 차분히 하나씩 해보고자 한다. 국내 상장사는 참고로 코스피, 코스닥, 코넥스가 존재한다. 이 3가지 마켓에 상장된 회사들의 리스트는 아래 명령어를 통해 한번에 DataFrame형태로 불러올 수 있다.**


```python
KRX_LISTING = fdr.StockListing('KRX') #참고로 NYSE를 입력하면 미국시장도 소환가능!
```

**어떤 정보를 가져왔는지 개요는 아래와 같다. 깔끔하게 약 10가지 정보를 DataFrame으로 제공해준다.**


```python
KRX_LISTING.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2597 entries, 0 to 2596
    Data columns (total 10 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   Symbol          2597 non-null   object        
     1   Market          2597 non-null   object        
     2   Name            2597 non-null   object        
     3   Sector          2397 non-null   object        
     4   Industry        2377 non-null   object        
     5   ListingDate     2397 non-null   datetime64[ns]
     6   SettleMonth     2397 non-null   object        
     7   Representative  2397 non-null   object        
     8   HomePage        2229 non-null   object        
     9   Region          2397 non-null   object        
    dtypes: datetime64[ns](1), object(9)
    memory usage: 223.2+ KB



```python
KRX_LISTING
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Market</th>
      <th>Name</th>
      <th>Sector</th>
      <th>Industry</th>
      <th>ListingDate</th>
      <th>SettleMonth</th>
      <th>Representative</th>
      <th>HomePage</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>060310</td>
      <td>KOSDAQ</td>
      <td>3S</td>
      <td>특수 목적용 기계 제조업</td>
      <td>반도체 웨이퍼 캐리어</td>
      <td>2002-04-23</td>
      <td>03월</td>
      <td>박종익, 김세완 (각자 대표이사)</td>
      <td>http://www.3sref.com</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>1</th>
      <td>095570</td>
      <td>KOSPI</td>
      <td>AJ네트웍스</td>
      <td>산업용 기계 및 장비 임대업</td>
      <td>렌탈(파렛트, OA장비, 건설장비)</td>
      <td>2015-08-21</td>
      <td>12월</td>
      <td>이현우</td>
      <td>http://www.ajnet.co.kr</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>2</th>
      <td>006840</td>
      <td>KOSPI</td>
      <td>AK홀딩스</td>
      <td>기타 금융업</td>
      <td>지주사업</td>
      <td>1999-08-11</td>
      <td>12월</td>
      <td>채형석, 이석주(각자 대표이사)</td>
      <td>http://www.aekyunggroup.co.kr</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>3</th>
      <td>054620</td>
      <td>KOSDAQ</td>
      <td>APS홀딩스</td>
      <td>기타 금융업</td>
      <td>인터넷 트래픽 솔루션</td>
      <td>2001-12-04</td>
      <td>12월</td>
      <td>정기로</td>
      <td>http://www.apsholdings.co.kr</td>
      <td>경기도</td>
    </tr>
    <tr>
      <th>4</th>
      <td>265520</td>
      <td>KOSDAQ</td>
      <td>AP시스템</td>
      <td>특수 목적용 기계 제조업</td>
      <td>디스플레이 제조 장비</td>
      <td>2017-04-07</td>
      <td>12월</td>
      <td>김영주</td>
      <td>http://www.apsystems.co.kr</td>
      <td>경기도</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2592</th>
      <td>000547</td>
      <td>KOSPI</td>
      <td>흥국화재2우B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2593</th>
      <td>000545</td>
      <td>KOSPI</td>
      <td>흥국화재우</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2594</th>
      <td>003280</td>
      <td>KOSPI</td>
      <td>흥아해운</td>
      <td>해상 운송업</td>
      <td>외항화물운송업(케미컬탱커)</td>
      <td>1976-06-29</td>
      <td>12월</td>
      <td>이환구</td>
      <td>http://www.heung-a.com</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>2595</th>
      <td>037440</td>
      <td>KOSDAQ</td>
      <td>희림</td>
      <td>건축기술, 엔지니어링 및 관련 기술 서비스업</td>
      <td>설계 및 감리용역</td>
      <td>2000-02-03</td>
      <td>12월</td>
      <td>정영균, 이목운, 허철호, 염두성 (각자대표)</td>
      <td>http://www.heerim.com</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>2596</th>
      <td>238490</td>
      <td>KOSDAQ</td>
      <td>힘스</td>
      <td>특수 목적용 기계 제조업</td>
      <td>OLED Mask 인장기, OLED Mask 검사기 등</td>
      <td>2017-07-20</td>
      <td>12월</td>
      <td>김주환</td>
      <td>http://www.hims.co.kr</td>
      <td>인천광역시</td>
    </tr>
  </tbody>
</table>
<p>2597 rows × 10 columns</p>
</div>



**약 2597개 회사들이 검색되고 있다. 이게 실제 맞는 숫자인지는 모르겠으나 코드 뒷단을 뜯어보니 krx marketdata에서 가져오는 것처럼 보여서 아마도 최신이 계속 반영되는 원천을 가지고 있는 듯하다. 여튼 다행 **

**테스트 삼아서 2가지 종목 005930과 000660의 가격추이를 그려본다. 참고로 잘 아는 삼성전자와 하이닉스다.**


```python
df1 = fdr.DataReader('005930', '2019-10-01', '2020-12-04') #삼성전자
df2 = fdr.DataReader('000660', '2019-10-01', '2020-12-04') #하이닉스
```


```python
df1['Close'].plot()
df2['Close'].plot()
```




    <AxesSubplot:xlabel='Date'>




![png](https://github.com/shoman2/shoman2.github.io/assets/img/output_12_1.png)

**최근에 크게 가격이 상승한 것으로 보여진다. 특히 하이닉스. 이런형태로 각 종목별 레이블링과 기간에따른 종가 Movement를 조금더 디테일하게 그려보자. 삼성전자, 한글과컴퓨터, 그리고 각종 ETF의 움직임을 동시에 그려보도록 한다.**


```python
stock_list = [
  ["삼성전자", "005930"],
  ["한글과컴퓨터", "030520"],
  ["TIGER 2차전지테마", "305540"],
  ["KODEX 200", "069500"],
  ["TIGER 소프트웨어" ,"157490"],
  ["TIGER 현대차그룹+펀더멘털 ()","138540"],
  ["KINDEX 미국S&P 500", "360200"]
]

```

**stock_list 안에다가 종목이름과 종목코드를 입력하고 데이터 프레임형태로 정리한다.**


```python
import pandas as pd

df_list = [fdr.DataReader(code, '2019-11-01', '2020-12-31')['Close'] for name, code in stock_list]
#len(df_list)

```


```python
# pd.concat()로 합치기

df = pd.concat(df_list, axis=1)
df.columns = [name for name, code in stock_list] 
df.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>삼성전자</th>
      <th>한글과컴퓨터</th>
      <th>TIGER 2차전지테마</th>
      <th>KODEX 200</th>
      <th>TIGER 소프트웨어</th>
      <th>TIGER 현대차그룹+펀더멘털 ()</th>
      <th>KINDEX 미국S&amp;P 500</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-01</th>
      <td>51200</td>
      <td>10050</td>
      <td>7144</td>
      <td>27273</td>
      <td>8493</td>
      <td>16388</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-04</th>
      <td>52300</td>
      <td>10050</td>
      <td>7204</td>
      <td>27665</td>
      <td>8498</td>
      <td>16437</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-05</th>
      <td>52700</td>
      <td>10050</td>
      <td>7234</td>
      <td>27851</td>
      <td>8478</td>
      <td>16617</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-06</th>
      <td>53300</td>
      <td>10150</td>
      <td>7209</td>
      <td>27865</td>
      <td>8483</td>
      <td>16471</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-07</th>
      <td>52900</td>
      <td>11350</td>
      <td>7100</td>
      <td>27831</td>
      <td>8528</td>
      <td>16563</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-08</th>
      <td>52100</td>
      <td>11550</td>
      <td>7189</td>
      <td>27738</td>
      <td>8478</td>
      <td>16549</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-11</th>
      <td>51600</td>
      <td>11200</td>
      <td>7149</td>
      <td>27524</td>
      <td>8458</td>
      <td>16480</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-12</th>
      <td>52600</td>
      <td>11350</td>
      <td>7129</td>
      <td>27768</td>
      <td>8458</td>
      <td>16553</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-13</th>
      <td>52500</td>
      <td>11400</td>
      <td>7035</td>
      <td>27543</td>
      <td>8368</td>
      <td>16432</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-11-14</th>
      <td>52800</td>
      <td>10750</td>
      <td>7030</td>
      <td>27763</td>
      <td>8707</td>
      <td>16446</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**시계열 형태로 인덱스값과 각 종목별 종가가 깔끔히 정리가 되었다. 이를 시각화 해보자**


```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = (15,15)
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.formatter.limits"] = -10000, 10000

df.plot()

```




    <AxesSubplot:xlabel='Date'>




![png](https://github.com/shoman2/shoman2.github.io/assets/img/output_19_1.png)



```python

```

**기타 다른 ETF 종목들도 동일한 방식으로 그려보면 아래와 같다.**


```python
stock_list2= [
  ["ARIRANG 신흥국MSCI", "195980"],
  ["KODEX 골드선물(H)", "132030"],
  ["TIGER 미국S&P500 선물(H)" ,"143850"],
    ["KODEX 200", "069500"],
    #["TIGER 소프트웨어" ,"157490"],
    #["KOSEF 국고채10년","148070"],
    #[" KODEX 단기채권PLUS", "214980"]
]
```


```python
import pandas as pd

df_list2 = [fdr.DataReader(code, '2019-11-01', '2020-12-31')['Close'] for name, code in stock_list2]
#len(df_list)

```


```python
# pd.concat()로 합치기

df2 = pd.concat(df_list2, axis=1)
df2.columns = [name for name, code in stock_list2] 
df2.tail(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ARIRANG 신흥국MSCI</th>
      <th>KODEX 골드선물(H)</th>
      <th>TIGER 미국S&amp;P500 선물(H)</th>
      <th>KODEX 200</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-23</th>
      <td>11795</td>
      <td>12885</td>
      <td>41490</td>
      <td>34810</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>11800</td>
      <td>12550</td>
      <td>41920</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>11850</td>
      <td>12410</td>
      <td>42320</td>
      <td>34795</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>11850</td>
      <td>12445</td>
      <td>42335</td>
      <td>35140</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>11930</td>
      <td>12435</td>
      <td>42250</td>
      <td>35220</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>11760</td>
      <td>12185</td>
      <td>42005</td>
      <td>34685</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>11805</td>
      <td>12270</td>
      <td>42535</td>
      <td>35235</td>
    </tr>
    <tr>
      <th>2020-12-02</th>
      <td>11885</td>
      <td>12450</td>
      <td>42520</td>
      <td>35850</td>
    </tr>
    <tr>
      <th>2020-12-03</th>
      <td>11960</td>
      <td>12615</td>
      <td>42670</td>
      <td>36105</td>
    </tr>
    <tr>
      <th>2020-12-04</th>
      <td>12085</td>
      <td>12650</td>
      <td>42730</td>
      <td>36750</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = (15,15)
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.formatter.limits"] = -10000, 10000

df2.plot()

```




    <AxesSubplot:xlabel='Date'>




![png](https://github.com/shoman2/shoman2.github.io/assets/img/output_25_1.png)



**정말 좋은 패키지 같다. 놀라울정도로 깔끔한 데이터를 끌어올 수 있는 듯 하다. 강추강추!**
