# DLMI-SAM Auto Labeler

Meta AI의 [SAM 3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3)를 기반으로 한 반자동 비디오/이미지 세그멘테이션 및 어노테이션 GUI 도구입니다. 포인트/박스 프롬프트, 폴리곤 입력, 텍스트 기반 탐지(PCS), 프레임 단위 마스크 전파를 지원하며 LabelMe 호환 어노테이션을 생성합니다.


## 설치

```bash
git clone https://github.com/chickencoin2/DLMI_SAM.git
cd DLMI_SAM

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

> **참고:** `requirements.txt`는 기본적으로 CUDA 12.8용 PyTorch를 설치합니다. 다른 CUDA 버전을 사용하는 경우 [PyTorch - Get Started](https://pytorch.org/get-started/locally/)를 참조하여 `--extra-index-url` 줄을 수정하세요.

> **참고:** 이 프로젝트는 SAM3 모델을 지원하는 `transformers` 라이브러리가 필요합니다. `requirements.txt`에는 정상 동작이 검증된 특정 커밋(`3a8d291`)이 지정되어 있습니다. 다른 버전의 `transformers`는 오탈자나 불완전한 SAM3 지원으로 인해 동작하지 않을 수 있습니다. 별도 버전을 설치하는 경우, `Sam3VideoModel`, `Sam3TrackerVideoModel`, `Sam3Model` 클래스가 정상적으로 포함되어 있는지 확인하세요.

## 사용법

```bash
python app.py
```

> **참고:** 첫 실행 시 SAM3 모델 로딩에 수십초 이상 소요될 수 있습니다.

실행하면 아래와 같은 초기 화면이 나타납니다. 좌측은 영상 표시 영역, 우측은 컨트롤 패널입니다.

![초기 화면](images/main.png)

1. **Select Source** 버튼으로 비디오 파일을 선택합니다
2. Object control 탭에서 프롬프트 방식, 라벨링될 객체 명칭을 선택합니다
3. 영상 위에서 프롬프트를 입력하여 세그멘테이션을 수행합니다
4. **Propagate/Review** 탭에서 전파하여 전체 프레임에 마스크를 확장합니다
5. 결과를 검토한 뒤 LabelMe JSON 또는 YOLO 형식으로 저장합니다

아래는 실제 세그멘테이션 및 전파 과정 예시입니다:

![세그멘테이션 예시](images/demo.gif)


## 라이선스

이 프로젝트는 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)에 따라 배포됩니다.

## 감사의 말

이 프로젝트는 Meta AI의 **SAM 3 (Segment Anything Model 3)**를 기반으로 구축되었으며, [Hugging Face Transformers](https://github.com/huggingface/transformers) 라이브러리를 통해 사용합니다.

SAM 3를 사용하시는 경우 원본 논문을 인용해 주세요:

```bibtex
@misc{carion2025sam3segmentconcepts,
  title={SAM 3: Segment Anything with Concepts},
  author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
  year={2025},
  eprint={2511.16719},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.16719},
}
```

## 인용

이 라벨링 도구를 연구에 사용하시는 경우 다음과 같이 인용해 주세요:

```bibtex
@software{dlmi_sam_autolabeler,
  title={DLMI-SAM Auto Labeler},
  author={TODO},
  year={2025},
  url={https://github.com/chickencoin2/DLMI_SAM}
}
```
