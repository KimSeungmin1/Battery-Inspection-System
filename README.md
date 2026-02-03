# Battery Inspection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI 기반 배터리 외관 검사 시스템입니다.  
AI-based automated inspection solution for battery exterior quality control.

---

## 개요 / Overview

본 프로젝트는 배터리 외관 검사를 위한 실시간 AI 검사 솔루션을 제공합니다. Basler 산업용 카메라와 연동하여 영상을 수집하고, MobileNetV3 기반 분류 모델과 OpenCV 기반 영상처리를 결합한 하이브리드 방식으로 불량 여부를 판정합니다.

This project provides a real-time AI inspection solution for battery exterior quality control. It integrates with Basler industrial cameras to capture images and performs defect detection through a hybrid approach combining MobileNetV3-based classification and OpenCV-based image processing.

---

## 주요 기능 / Features

| 기능 | 설명 |
|------|------|
| **실시간 검사** | Basler 카메라 영상 스트리밍 및 실시간 배터리 감지 |
| **AI 분류** | MobileNetV3 기반 3-class 분류 (Normal / Damaged / Pollution) |
| **하이브리드 검사** | AI 분류 + OpenCV 크랙/스크래치/이물질 탐지 |
| **결과 안정화** | 다수결 및 시간 기반 판정 안정화 (깜빡임 방지) |
| **모델 학습** | 자체 데이터셋으로 분류기 학습 및 정확도 평가 지원 |

---

## 프로젝트 구조 / Project Structure

```
Battery-Inspection-System/
├── inspection_app/          # 검사 앱 (카메라 연동 실시간 검사)
│   └── main.py
├── MobileNetV3/             # 모델 학습 및 설정
│   ├── config.yaml          # 학습/추론 설정
│   ├── train_autoencoder_mixed.py   # 분류기 학습
│   ├── train_cooldown.py    # Cool Down 학습 (Fine-tuning)
│   ├── eval_classifier_accuracy.py  # 정확도 평가
│   └── requirements.txt
├── .env.example             # 환경변수 설정 예시
├── LICENSE                  # MIT License
├── requirements.txt         # 프로젝트 의존성 (pip freeze)
└── README.md
```

---

## 요구사항 / Requirements

- Python 3.11.7 (권장, 라이브러리 호환성 검증됨 / recommended, library compatibility verified)
  - Python 3.14.x 등 최신 버전은 일부 라이브러리와 호환 문제가 있을 수 있습니다.
- NVIDIA GPU (CUDA 12.x 권장, CPU 전용 실행 가능)
- Basler 산업용 카메라 (pypylon 지원)
- Windows (pypylon, PyQt6 지원 환경)

### 테스트된 환경 / Tested Environment

| 항목 | 버전 |
|------|------|
| Python | 3.11.7 |
| GPU | NVIDIA GeForce RTX 5050 (8GB) |
| NVIDIA 드라이버 | 591.74 |
| CUDA (드라이버 지원) | 13.1 |
| PyTorch (빌드) | 2.11.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| OpenCV | 4.12.0 |
| PyQt6 | 6.10.1 |
| pypylon | 4.2.0 |

---

## 설치 / Installation

### 1. 저장소 클론

```bash
git clone https://github.com/<username>/Battery-Inspection-System.git
cd Battery-Inspection-System
```

### 2. 가상환경 생성 및 활성화 (권장)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 의존성 설치

**검사 앱 실행만 필요한 경우** (MobileNetV3 폴더 기준):

```bash
cd MobileNetV3
pip install -r requirements.txt
```

**전체 환경 동기화가 필요한 경우** (프로젝트 루트 기준):

```bash
pip install -r requirements.txt
```

> PyTorch CUDA 버전은 `MobileNetV3/requirements.txt` 내 주석을 참고하여 환경에 맞게 변경할 수 있습니다.

---

## 실행 / Usage

### 검사 앱 실행 (Inspection App)

```bash
# 프로젝트 루트에서
python inspection_app/main.py

# 앱 실행 이미지
<img width="1908" height="1044" alt="image" src="https://github.com/user-attachments/assets/f50f1a82-ef3c-4233-bd87-d6f517ebe461" />

```

또는

```bash
cd inspection_app
python main.py
```

### 모델 학습 (Training)

```bash
cd MobileNetV3
python train_autoencoder_mixed.py
```

학습 데이터 경로는 `config.yaml` 또는 `.env`의 `DATA_DIR`로 설정합니다. 상세 설정은 `MobileNetV3/config.yaml`을 참고하시기 바랍니다.

### 정확도 평가

```bash
cd MobileNetV3
python eval_classifier_accuracy.py --model runs/model_classifier_best.pth
```

---

## 설정 / Configuration

### config.yaml

`MobileNetV3/config.yaml`에서 학습 파라미터, 데이터 경로, UI 설정 등을 수정할 수 있습니다.

### 환경변수 (.env)

데이터 경로가 기본값과 다른 경우 `.env.example`을 복사하여 `.env`로 저장한 뒤, 다음 변수를 설정합니다.

| 변수 | 설명 |
|------|------|
| `DATA_DIR` | 학습 데이터 폴더 전체 경로 (data_resized 디렉터리까지) |
| `DATA_BASE_DIR` | config.yaml 상대경로의 기준 디렉터리 |

---

## 학습 데이터 구조 / Data Structure

기본 경로: `MobileNetV3/data/data_resized/`

```
data_resized/
├── Training/
│   ├── Sources/          # 학습 이미지
│   └── label/            # 라벨 (JSON 등)
└── Validation/
    ├── Sources/
    └── label/
```

---

## 라이선스 / License

본 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.  
This project is distributed under the [MIT License](LICENSE).

저작권 (c) 2026 KimSeungmin1  
Copyright (c) 2026 KimSeungmin1

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하시기 바랍니다.

---

## 참고 / Notes

- 검사 앱 실행 시 `model_classifier_best.pth`가 없으면 AI 분류 기능이 비활성화되며, OpenCV 기반 검사만 수행됩니다.
- 모델 파일 위치: `inspection_app/` 또는 `MobileNetV3/runs/` (자동 탐색)
