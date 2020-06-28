---
layout: post
title:  "AI Platform Notebooks 사용 방법 및 소개"
subtitle:  "AI Platform Notebooks 사용 방법 및 소개"
categories: gcp
tags: ai-platform
comments: true
---

- Google Cloud Platform의 AI Platform Notebooks에 대해 작성한 글입니다



---


### AI Platform
- AI Platform은 Google Cloud Platform에서 만든 서비스로, 머신러닝 모델을 프러덕션용으로 구현, 배포할 때 도와주는 제품
	- Serving, TFX, Explainable AI, TPU
	- AI Hub, Kubeflow
	- 다양한 도구들을 사용할 수 있고, 점점 추가되고 있음
- Google에서 주력으로 밀고 있는 느낌
	- Google Datalab은 이제 사용하지 않고, AI Platform Notebooks을 더 권장하는듯
- BigQuery나 Cloud Storage에 저장된 데이터를 활용해 전처리하고(BigQuery나 Dataflow 등 활용) 학습한 후, Serving
	- 그 파이프라인도 자동화

<img src="https://www.dropbox.com/s/gq1rjkizzx4orgs/Screenshot%202020-05-10%2013.59.27.png?raw=1">

- 가격
	- Kubeflow, AI Hub, Notebooks은 무료로 사용할 수 있음
	- AI Platform Training, AI Platform Prediction, Compute Engine, Google Kuberenetes Engine, BigQuery, Cloud Storage만 부과
- 앞으로 AI Platform 관련 글을 꾸준히 올릴 예정
	- [mlops-examples-on-gcp](https://github.com/zzsza/mlops-examples-on-gcp)에 코드도 올릴 예정


---

### AI Platform Notebooks
- [제품 설명 문서](https://cloud.google.com/ai-platform-notebooks?hl=ko), [공식 문서](https://cloud.google.com/ai-platform/notebooks/docs?hl=ko)
- JupyterLab의 Managed Service라고 생각하면 됨
	- Google Cloud Platform의 Compute Engine에 Jupyter 설치, 라이브러리 설치 등(혹은 Docker 활용)을 직접 할 수 있지만 이런 작업은 클라우드에 익숙하지 않은 사람은 힘들 수 있음
	- 직접 설치하는 방법은 [Google Cloud Platform에서 Tensorflow, Jupyter Notebook 설치 및 startup script 설정](https://zzsza.github.io/gcp/2018/06/14/install-tensorflow-jupyter-in-gcp/) 참고  
	- 특히 GPU 설정, 인증 등은 처음엔 까다로울 수 있음
	- VPN 설정 등도 제공함
- 비용
	- Notebooks 자체를 만드는 비용은 들지 않고, 리소스를 사용하는 비용만 냄(Compute Engine, AI Platform Training, AI Platform Prediction, Google Kuberenetes Engine, BigQuery, Cloud Storage 비용) 

- 다음 내용을 순서대로 진행함
	- 1) Notebook 인스턴스 생성하기
		- 라이브러리 설치하기
	- 2) 커스텀 Conatiner로 인스턴스 만들기
	- 3) Github 연결하기

---


### 1) Notebook 인스턴스 생성하기
- 우선, Google Cloud Platform에 가입하고, 결제 활성화, Compute Engine API 사용 설정해야 함
	- [링크](https://console.cloud.google.com/flows/enableapi?apiid=compute.googleapis.com&hl=ko&_ga=2.208308498.146998283.1588999422-154659433.1588690943)에 들어가면 Compute Engine API 사용 설정할 수 있음
- [링크](https://console.cloud.google.com/ai-platform/notebooks/instances)를 클릭해 AI Platform Notebooks으로 이동

<img src="https://www.dropbox.com/s/mkas7awu4k8cc0s/Screenshot%202020-05-10%2014.51.45.png?raw=1">

- 새 인스턴스 클릭

<img src="https://www.dropbox.com/s/k8wfkob5bqtp65k/Screenshot%202020-05-10%2014.52.28.png?raw=1">

- R, Python, TensorFlow, PyTorch, XGBoost, CUDA, Customize instance 등이 있음
	- Python 클릭시 이름, 지역, 맞춤설정 등이 있음
		- 맞춤 설정에선 머신의 성능, GPU 유무, 부팅 디스크, 네트워크, 권한 등을 설정 가능
		- <img src="https://www.dropbox.com/s/s53sypsv0954cpi/Screenshot%202020-05-10%2014.56.39.png?raw=1">
	- TensorFlow Enterprise 2.1 누르면 WIthout GPUs와 With 1 NVIDIA Tesla T4가 나옴
		- T4 클릭
		- NVIDIA GPU 드라이버 자동 설치를 체크해야 나중에 GPU 수를 조정할 수 있음
		- 맞춤 설정은 Python과 동일
		- <img src="https://www.dropbox.com/s/0em6lx6v4amsgrg/Screenshot%202020-05-10%2014.58.02.png?raw=1">
	- Customize instance는 2) 커스텀 Container로 인스턴스 만들기에서 다룰 예정
- 여기서 GPU 선택할 때, IAM Admin - Quotas에 GPU 할당량이 있는지 확인해야 함
	- [링크](https://console.cloud.google.com/iam-admin/quotas?hl=ko)

	<img src="https://www.dropbox.com/s/vs8b5fdb0532wh8/Screenshot%202020-05-10%2019.58.22.png?raw=1">

- 원하는 region의 GPU Quota를 수정하고 Notebook을 생성
	- us-east1 region 할당량을 수정함
- 우선 GPU 없는 인스턴스 생성
	- Python 클릭후 노트북 생성 

	<img src="https://www.dropbox.com/s/eptigp9kptlu3ty/Screenshot%202020-05-10%2020.07.40.png?raw=1">

- 인스턴스 이름 클릭시 Compute Engine 설정으로 이동함
- JUPYTERLAB 열기 클릭

<img src="https://www.dropbox.com/s/qlaqkjoqaq09je3/Screenshot%202020-05-10%2020.09.04.png?raw=1">
	

- !pip3 list로 설치된 라이브러리 확인
	- gevent, pyqt 등 다양한 라이브러리가 설치되어 있음

<img src="https://www.dropbox.com/s/skpgsv3p1k7o1ex/Screenshot%202020-05-10%2020.10.09.png?raw=1">



### 2) 커스텀 Container로 인스턴스 만들기
- 터미널에서 gcloud 명령어로 진행
- Docker Image가 필요함
- gcloud config 확인
	- 사용하려는 gcloud config가 Active인지 확인	
	```
	gcloud config configurations list
	```
	
- config_name인 설정 활성화(이미 사용하려는 설정이 IS_ACTIVE True면 할 필요 없음)

	```
	gcloud config configurations activate <config_name>
	```	

- Project 설정

	```
	PROJECT_ID=[YOUR PROJECT ID]
	
	gcloud config set project $PROJECT_ID
	```	
	
- gcloud로 API 사용 설정

	```
	gcloud services enable \
	cloudbuild.googleapis.com \
	container.googleapis.com \
	cloudresourcemanager.googleapis.com \
	iam.googleapis.com \
	containerregistry.googleapis.com \
	containeranalysis.googleapis.com \
	ml.googleapis.com \
	dataflow.googleapis.com 
	```

- Cloud Build service account가 Editor 권한을 갖도록 설정

	```
	PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
	CLOUD_BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
	gcloud projects add-iam-policy-binding $PROJECT_ID \
	  --member serviceAccount:$CLOUD_BUILD_SERVICE_ACCOUNT \
	  --role roles/editor
	```	
	
- requirements.txt 생성

	```
	cat > requirements.txt << EOF
	kfp==0.2.5
	fire
	gcsfs
	EOF
	```

- Dockerfile 생성
	- Docker에 대해 잘 모르면 [Docker와 쿠버네티스의 이해](https://zzsza.github.io/development/2018/04/17/docker-kubernetes/), [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) 참고

	```
	cat > Dockerfile << EOF
	FROM gcr.io/deeplearning-platform-release/base-cpu
	SHELL ["/bin/bash", "-c"]
	RUN apt-get update -y && apt-get -y install kubectl
	COPY requirements.txt .
	RUN python -m pip install -U -r requirements.txt 
	EOF
	```

- Container Registry에 빌드한 Image Push

	```
	IMAGE_NAME=kfp-dev
	TAG=latest
	IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
	
	gcloud builds submit --timeout 15m --tag ${IMAGE_URI} .
	```

- gcloud compute instance 명령어로 생성
	- ZONE은 [리전과 영역](https://cloud.google.com/compute/docs/regions-zones?hl=ko) 문서 참고함
	
	```
	ZONE=us-east1-b
	INSTANCE_NAME=mlops-examples-ai-platform-notebook
		
	IMAGE_FAMILY="common-container"
	IMAGE_PROJECT="deeplearning-platform-release"
	INSTANCE_TYPE="n1-standard-4"
	METADATA="proxy-mode=service_account,container=$IMAGE_URI"
		
	gcloud compute instances create $INSTANCE_NAME \
	    --zone=$ZONE \
	    --image-family=$IMAGE_FAMILY \
	    --machine-type=$INSTANCE_TYPE \
	    --image-project=$IMAGE_PROJECT \
	    --maintenance-policy=TERMINATE \
	    --boot-disk-device-name=${INSTANCE_NAME}-disk \
	    --boot-disk-size=100GB \
	    --boot-disk-type=pd-ssd \
	    --scopes=cloud-platform,userinfo-email \
	    --metadata=$METADATA
	```
	
- AI Platform Notebook에 보면 생성된 인스턴스를 확인할 수 있음

<img src="https://www.dropbox.com/s/uyt009qll92pmk6/Screenshot%202020-05-10%2020.59.48.png?raw=1">


### 3) Github 연결하기
- JupyterLab에서 아래 버튼 클릭

<img src="https://www.dropbox.com/s/urb1x0ocwtqp574/Screenshot%202020-05-10%2020.12.53.png?raw=1">

- Github HTTPS 연결
- 암호와 비밀번호를 입력하면 폴더가 추가됨
- 파일을 수정한 후, Git 버튼 클릭하면 어떤 것이 Staged고 Changed고 Untracked인지 볼 수 있음(History도 포함)

<img src="https://www.dropbox.com/s/rzset2u99tchx56/Screenshot%202020-05-10%2020.18.07.png?raw=1">

- Changed에 있는 README.md에 마우스 커서를 올려두면 우측에 다양한 아이콘이 생김

<img src="https://www.dropbox.com/s/r5fwqsym9pz2xfq/Screenshot%202020-05-10%2020.20.30.png?raw=1" width="400" height="350">

- +-가 같이 있는 버튼은 Diff 체크해주고, +버튼은 Staged로 올려줌
- Staged에 올라간 후는 - 버튼이 생김(Unstaged)
- Staged에 파일을 올린 후, 커밋 메세지를 입력하고 커밋 버튼을 누르면 커밋이 됨
	- 간단히 이름과 이메일을 물어봄

	<img src="https://www.dropbox.com/s/12yitscj7onw5d2/Screenshot%202020-05-10%2020.22.19.png?raw=1" width="200" height="500">

- 이제 Push를 위해 클라우드 모양을 클릭함
	- 아이디, 비밀번호를 입력하면 Push가 됨

	<img src="https://www.dropbox.com/s/0dlrr4xdpd2pvpp/Screenshot%202020-05-10%2020.24.47.png?raw=1" width="400" height="400">
	
	
	
### 정리
- 클릭으로 JupyterLab 환경을 간단하게 만들 수 있음
- Docker 이미지로 생성할 수도 있음
- Github 연동도 가능
- 가격은 리소스 사용 비용만 나감
- 항상 사용하고 중지 또는 삭제 클릭!	
- 작성한 내용 외에도 [SSH를 사용해 액세스](https://cloud.google.com/ai-platform/notebooks/docs/ssh-access?hl=ko), [AI Platform Notebooks Cloud IAM 역할](https://cloud.google.com/ai-platform/notebooks/docs/iam?hl=ko) 등도 있음


### Reference
- [AI Platform Notebooks 공식 문서](https://cloud.google.com/ai-platform/notebooks/docs?hl=ko)
- [jupyterlab_cookiecutter](https://github.com/gclouduniverse/jupyterlab_cookiecutter)
- [proj_custom_ai_platform_notebook](https://github.com/tarrade/proj_custom_ai_platform_notebook)
- [julia on ai platform](https://github.com/mmm/julia-on-ai-platform)
- [Jupyter Notebook-Centric Development (Cloud Next '19)](https://youtu.be/xU_xdogXFeE)
- [How to use Jupyter on a Google Cloud VM](https://towardsdatascience.com/how-to-use-jupyter-on-a-google-cloud-vm-5ba1b473f4c2)