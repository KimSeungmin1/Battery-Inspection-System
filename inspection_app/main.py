# -*- coding: utf-8 -*-
"""
AI Inspection System - Battery Detection and Classification
AI 배터리 검사 시스템 - 배터리 탐지 및 분류

목적 (Purpose):
    배터리 외관 검사를 위한 AI 기반 자동화 검사 솔루션입니다.
    AI-based automated inspection solution for battery exterior quality control.

주요 기능 (Features):
    - 실시간 카메라 영상 처리 및 배터리 감지: Basler 카메라 연동
      Real-time camera feed processing and battery detection (Basler camera)
    - AI 기반 배터리 불량 분류: MobileNetV3 (정상/Normal, 크랙/Damaged, 오염/Pollution)
      AI-based defect classification (MobileNetV3)
    - OpenCV 기반 크랙 검출: 하이브리드 검사 (AI + 전통 영상처리)
      OpenCV-based crack detection (hybrid AI + traditional image processing)
    - 안정화된 판정 결과 표시: 깜빡임 방지, 판정 유지
      Stable result display (no flickering, result persistence)

실행 방법 (Run):
    # inspection_app 폴더에서 / From inspection_app folder:
    python main.py
    
    # 프로젝트 루트에서 / From project root:
    python inspection_app/main.py
    
    # 또는 PyInstaller로 빌드된 exe 실행 / Or run built exe
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from pypylon import pylon

# 로그 출력 즉시 반영 (UTF-8 한글 깨짐 방지) / Ensure UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGridLayout,
                             QSizePolicy, QSlider, QSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
import pyqtgraph as pg
import time

# PyTorch 및 MobileNetV3 분류기 (선택 의존성: 없어도 앱은 동작)
# PyTorch and MobileNetV3 classifier (optional: app works without it)
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("[WARNING] PyTorch를 사용할 수 없습니다.")


class MobileNetV3Classifier:
    """
    MobileNetV3 기반 배터리 불량 분류기
    MobileNetV3-based Battery Defect Classifier
    
    사전 학습된 MobileNetV3 모델을 사용하여 배터리 이미지를 분류합니다.
    Uses pre-trained MobileNetV3 to classify battery images as normal or defect.
    
    지원 클래스 구성 (Supported class configurations):
    - 2클래스: normal(0), defect(1)
    - 3클래스: Normal(0), Damaged(1), Pollution(2)
    
    Attributes:
        model: PyTorch 모델 인스턴스 / PyTorch model instance
        device: 계산 디바이스 (CPU/GPU) / Compute device
        class_names: 클래스 이름 리스트 / Class name list
        transform: 이미지 전처리 파이프라인 / Image preprocessing pipeline
    """
    def __init__(self, model_path=None, device=None):
        if not CLASSIFIER_AVAILABLE:
            raise RuntimeError("PyTorch가 설치되지 않았습니다.")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 모델 생성 (학습 코드와 일치: mobilenet_v3_large + Dropout 0.3)
        # 학습 모델 구조: classifier[0], [1], [2], Dropout(0.3), Linear(3)
        # 따라서 마지막 레이어는 classifier.4가 됨
        try:
            self.model = models.mobilenet_v3_large(weights=None)
            in_features = self.model.classifier[3].in_features
            dropout_rate = 0.3  # 학습 시 사용한 Dropout 비율 (config.yaml에서 확인)
            
            # Dropout을 포함한 분류기 레이어 구성 (학습 모델과 동일)
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],  # 기존 첫 번째 레이어
                self.model.classifier[1],  # 기존 두 번째 레이어
                self.model.classifier[2],  # 기존 세 번째 레이어
                nn.Dropout(p=dropout_rate),  # Dropout 추가
                nn.Linear(in_features, 3)  # 최종 분류 레이어 (3개 클래스)
            )
            print(f"[INFO] 모델 구조: mobilenet_v3_large + Dropout(0.3), 클래스 수: 3")
        except Exception as e:
            print(f"[WARNING] mobilenet_v3_large 생성 실패: {e}, small로 시도")
            self.model = models.mobilenet_v3_small(weights=None)
            in_features = self.model.classifier[3].in_features
            dropout_rate = 0.3
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],
                self.model.classifier[1],
                self.model.classifier[2],
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 3)
            )
            print(f"[INFO] 모델 구조: mobilenet_v3_small + Dropout(0.3), 클래스 수: 3")
        
        # 모델 로드
        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                print(f"[INFO] state_dict 키 개수: {len(state_dict.keys())}")
                print(f"[INFO] 첫 5개 키: {list(state_dict.keys())[:5]}")
                
                # 모델 구조 확인: state_dict의 classifier 마지막 레이어 확인
                # 학습 모델은 Dropout이 포함되어 있으므로 classifier.4가 마지막 레이어
                classifier_key = None
                has_dropout = False
                # 먼저 classifier.4 확인 (Dropout 포함 모델)
                for key in state_dict.keys():
                    if 'classifier.4.weight' in key:
                        classifier_key = key
                        has_dropout = True
                        break
                
                # classifier.4가 없으면 classifier.3 확인 (Dropout 없는 모델)
                if classifier_key is None:
                    for key in state_dict.keys():
                        if 'classifier.3.weight' in key:
                            classifier_key = key
                            has_dropout = False
                            break
                
                if classifier_key:
                    classifier_weight = state_dict[classifier_key]
                    num_classes_in_model = classifier_weight.shape[0]
                    print(f"[INFO] 학습된 모델 정보:")
                    print(f"   - 클래스 수: {num_classes_in_model}")
                    print(f"   - 마지막 레이어 키: {classifier_key}")
                    print(f"   - Dropout 포함: {has_dropout}")
                    
                    # 모델 구조 재구성 (state_dict와 일치하도록)
                    if has_dropout:
                        # Dropout 포함 모델: 이미 Sequential로 구성되어 있음
                        if not isinstance(self.model.classifier, nn.Sequential) or len(self.model.classifier) != 5:
                            print(f"[INFO] 모델 구조 재구성: Dropout 포함 모델로 변경")
                            in_features = self.model.classifier[3].in_features if not isinstance(self.model.classifier, nn.Sequential) else self.model.classifier[-1].in_features
                            dropout_rate = 0.3
                            self.model.classifier = nn.Sequential(
                                self.model.classifier[0] if isinstance(self.model.classifier, nn.Sequential) else self.model.classifier[0],
                                self.model.classifier[1] if isinstance(self.model.classifier, nn.Sequential) else self.model.classifier[1],
                                self.model.classifier[2] if isinstance(self.model.classifier, nn.Sequential) else self.model.classifier[2],
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(in_features, num_classes_in_model)
                            )
                    else:
                        # Dropout 없는 모델: Sequential이 아닌 경우만 재구성
                        if isinstance(self.model.classifier, nn.Sequential):
                            print(f"[INFO] 모델 구조 재구성: Dropout 없는 모델로 변경")
                            in_features = self.model.classifier[-1].in_features
                            self.model.classifier = nn.Sequential(
                                self.model.classifier[0],
                                self.model.classifier[1],
                                self.model.classifier[2],
                                nn.Linear(in_features, num_classes_in_model)
                            )
                        else:
                            # 이미 올바른 구조
                            in_features = self.model.classifier[3].in_features
                            self.model.classifier[3] = nn.Linear(in_features, num_classes_in_model)
                    
                    # 클래스 매핑 조정
                    if num_classes_in_model == 2:
                        self.class_names = ['defect', 'normal']
                        print(f"[INFO] 클래스 매핑: {self.class_names}")
                    elif num_classes_in_model == 3:
                        self.class_names = ['Normal', 'Damaged', 'Pollution']
                        print(f"[INFO] 클래스 매핑: {self.class_names}")
                
                # 모델 가중치 로드
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"[SUCCESS] 모델 로드 완료 (strict=True): {model_path}")
                except Exception as e:
                    print(f"[WARNING] strict=True로 로드 실패, strict=False로 재시도: {e}")
                    # strict=False로 재시도 (일부 키 불일치 허용)
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"[WARNING] 누락된 키: {missing_keys[:5]}... (총 {len(missing_keys)}개)")
                    if unexpected_keys:
                        print(f"[WARNING] 예상치 못한 키: {unexpected_keys[:5]}... (총 {len(unexpected_keys)}개)")
                    print(f"[SUCCESS] 모델 로드 완료 (strict=False): {model_path}")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"모델 로드 실패: {e}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 전처리 파이프라인 (Letterbox Resize는 별도 함수로 처리)
        # transforms.Resize는 사용하지 않음 (Letterbox 함수 사용)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 클래스 매핑 (기본값: 3개 클래스 - Normal, Damaged, Pollution)
        # 학습 코드와 일치: 첫 글자 대문자
        # 모델 로드 시 실제 클래스 수에 따라 자동 조정됨
        self.class_names = ['Normal', 'Damaged', 'Pollution']
    
    def preprocess_image(self, cv2_image):
        """
        검은색 여백 제거: 강제 스트레치 리사이즈
        1. Bilateral Filter (노이즈 제거 및 경계 보존)
        2. 강제 스트레치 리사이즈 to 320x320 (비율 무시, 검은색 여백 없음)
        3. Tensor 변환 (ToTensor + Normalize)
        
        Args:
            cv2_image: numpy.ndarray (BGR 형식)
        
        Returns:
            torch.Tensor: 전처리된 텐서 (1, 3, 320, 320)
            numpy.ndarray: 전처리된 이미지 (320x320, BGR) - 디버깅용
        """
        # 1. Bilateral Filter 적용 (가장 먼저)
        # 학습 데이터와 동일한 파라미터: d=9, sigmaColor=75, sigmaSpace=75
        bilateral_filtered = cv2.bilateralFilter(
            cv2_image, 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        )
        
        # 2. 강제 스트레치 리사이즈 to 320x320 (비율 무시, 검은색 여백 없음)
        # Letterbox 제거: 검은색 여백이 오염으로 오인식되는 문제 해결
        stretched = cv2.resize(bilateral_filtered, (320, 320), interpolation=cv2.INTER_LINEAR)
        
        # 3. BGR -> RGB 변환
        rgb_image = cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(rgb_image)
        
        # 4. Tensor 변환 (ToTensor + Normalize)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return input_tensor, stretched
    
    def predict(self, cv2_image, return_preprocessed=False):
        """
        OpenCV BGR 이미지를 입력받아 분류 예측
        학습 데이터와 동일한 전처리 과정 적용
        
        Args:
            cv2_image: numpy.ndarray (BGR 형식)
            return_preprocessed: True이면 전처리된 이미지도 반환
        
        Returns:
            tuple: (class_name, confidence, prob_normal, prob_defect) 또는 
                   (class_name, confidence, prob_normal, prob_defect, preprocessed_image)
        """
        try:
            # 전처리 (Bilateral + Letterbox + Tensor)
            input_tensor, preprocessed_img = self.preprocess_image(cv2_image)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                class_idx = predicted.item()
                confidence_value = confidence.item()
                
                # 실제 모델의 클래스 수 확인
                num_classes = probs.shape[1]
                
                # 클래스 이름 가져오기 (인덱스 범위 체크)
                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    # 인덱스가 범위를 벗어나면 첫 번째 클래스로 설정
                    class_name = self.class_names[0] if len(self.class_names) > 0 else "unknown"
                
                # 전체 확률도 반환 (실제 모델 클래스 수에 맞게)
                prob_normal = 0.0
                prob_defect = 0.0
                
                if num_classes == 2:
                    # 2개 클래스 모델: probs[0] = [prob_class0, prob_class1]
                    # 실제 모델은 2개 클래스만 있으므로, probs[0]의 인덱스는 0과 1만 존재
                    # class_names가 3개로 설정되어 있어도, 실제 확률 배열은 2개만 있음
                    
                    # class_idx를 확인하여 실제 모델의 클래스 순서 파악
                    # class_idx=0이고 class_name='defect'이면, probs[0][0]이 defect 확률
                    # class_idx=0이고 class_name='normal'이면, probs[0][0]이 normal 확률
                    # class_idx와 class_name을 함께 확인하여 확률 할당
                    # class_idx=0이고 class_name='defect'이면 probs[0][0]이 defect 확률
                    # class_idx=0이고 class_name='normal'이면 probs[0][0]이 normal 확률
                    # class_idx=1이고 class_name='defect'이면 probs[0][1]이 defect 확률
                    # class_idx=1이고 class_name='normal'이면 probs[0][1]이 normal 확률
                    
                    if class_idx == 0:
                        # 첫 번째 클래스가 선택됨
                        if class_name == 'defect':
                            # probs[0][0]이 defect 확률, probs[0][1]이 normal 확률
                            prob_defect = probs[0][0].item()
                            prob_normal = probs[0][1].item()
                        elif class_name == 'normal':
                            # probs[0][0]이 normal 확률, probs[0][1]이 defect 확률
                            prob_normal = probs[0][0].item()
                            prob_defect = probs[0][1].item()
                        else:
                            # class_name으로 판단 불가능하면 알파벳 순서 가정: ['defect', 'normal']
                            prob_defect = probs[0][0].item()
                            prob_normal = probs[0][1].item()
                    elif class_idx == 1:
                        # 두 번째 클래스가 선택됨
                        if class_name == 'defect':
                            # probs[0][1]이 defect 확률, probs[0][0]이 normal 확률
                            prob_defect = probs[0][1].item()
                            prob_normal = probs[0][0].item()
                        elif class_name == 'normal':
                            # probs[0][1]이 normal 확률, probs[0][0]이 defect 확률
                            prob_normal = probs[0][1].item()
                            prob_defect = probs[0][0].item()
                        else:
                            # class_name으로 판단 불가능하면 알파벳 순서 가정: ['defect', 'normal']
                            prob_defect = probs[0][1].item()
                            prob_normal = probs[0][0].item()
                    else:
                        # 알파벳 순서 가정 (fallback): ['defect', 'normal']
                        prob_defect = probs[0][0].item()
                        prob_normal = probs[0][1].item()
                    
                    # 디버깅: 확률 할당 확인 (5초마다)
                    if not hasattr(self, '_last_prob_assignment_log_time') or time.time() - self._last_prob_assignment_log_time > 5.0:
                        print(f"[DEBUG] 확률 할당 확인: class_idx={class_idx}, class_name={class_name}")
                        print(f"   probs[0][0]={probs[0][0].item():.3f}, probs[0][1]={probs[0][1].item():.3f}")
                        print(f"   할당 결과: prob_defect={prob_defect:.3f}, prob_normal={prob_normal:.3f}")
                        self._last_prob_assignment_log_time = time.time()
                elif num_classes == 3:
                    # 3개 클래스 모델: 두 가지 가능성
                    # 1. 기존: ['background', 'defect', 'normal']
                    # 2. 새로운: ['normal', 'Damaged', 'Pollution']
                    
                    if self.class_names == ['Normal', 'Damaged', 'Pollution']:
                        # 새로운 3개 클래스 모델: Normal(0), Damaged(1), Pollution(2)
                        prob_normal = probs[0][0].item()  # Normal
                        prob_damaged = probs[0][1].item()  # Damaged
                        prob_pollution = probs[0][2].item()  # Pollution
                        # defect 확률은 Damaged + Pollution의 합
                        prob_defect = prob_damaged + prob_pollution
                        
                        # 디버깅: 확률 확인 (5초마다)
                        if not hasattr(self, '_last_prob_log_time') or time.time() - self._last_prob_log_time > 5.0:
                            print(f"[DEBUG] 확률 정보 (3개 클래스): 클래스={class_name}, num_classes={num_classes}")
                            print(f"   prob_normal={prob_normal:.3f}, prob_damaged={prob_damaged:.3f}, prob_pollution={prob_pollution:.3f}")
                            print(f"   prob_defect={prob_defect:.3f} (Damaged+Pollution), confidence={confidence_value:.3f}")
                            self._last_prob_log_time = time.time()
                    else:
                        # 기존 3개 클래스 모델: ['background', 'defect', 'normal']
                        # 또는 새로운 모델: ['Normal', 'Damaged', 'Pollution']
                        if 'Normal' in self.class_names:
                            normal_idx = self.class_names.index('Normal')
                        elif 'normal' in self.class_names:
                            normal_idx = self.class_names.index('normal')
                            if normal_idx < num_classes:
                                prob_normal = probs[0][normal_idx].item()
                        if 'defect' in self.class_names:
                            defect_idx = self.class_names.index('defect')
                            if defect_idx < num_classes:
                                prob_defect = probs[0][defect_idx].item()
                
                # 디버깅: 확률 확인 (5초마다)
                if not hasattr(self, '_last_prob_log_time') or time.time() - self._last_prob_log_time > 5.0:
                    prob_sum = sum([probs[0][i].item() for i in range(num_classes)])
                    if num_classes == 2:
                        print(f"[DEBUG] 확률 정보: 클래스={class_name}, num_classes={num_classes}, class_names={self.class_names}")
                        print(f"   probs[0][0]={probs[0][0].item():.3f} (defect로 할당), probs[0][1]={probs[0][1].item():.3f} (normal로 할당)")
                        print(f"   prob_normal={prob_normal:.3f}, prob_defect={prob_defect:.3f}, confidence={confidence_value:.3f}")
                        print(f"   확률 합계: {prob_sum:.3f} (정상: 1.0), normal+defect={prob_normal + prob_defect:.3f}")
                    else:
                        print(f"[DEBUG] 확률 정보: 클래스={class_name}, num_classes={num_classes}, class_names={self.class_names}")
                        print(f"   전체 확률: {[probs[0][i].item() for i in range(num_classes)]}")
                        print(f"   prob_normal={prob_normal:.3f}, prob_defect={prob_defect:.3f}, confidence={confidence_value:.3f}")
                        print(f"   확률 합계: {prob_sum:.3f} (정상: 1.0)")
                    self._last_prob_log_time = time.time()
                
                # 모든 클래스의 확률 리스트 추출 (진단 모드용)
                all_probs = [probs[0][i].item() for i in range(num_classes)]
                
                if return_preprocessed:
                    return (class_name, confidence_value, prob_normal, prob_defect, preprocessed_img, all_probs)
                else:
                    return (class_name, confidence_value, prob_normal, prob_defect, all_probs)
        
        except Exception as e:
            # 로그 출력 제한 (5초마다만 출력)
            if not hasattr(self, '_last_predict_error_time') or time.time() - self._last_predict_error_time > 5.0:
                print(f"[ERROR] 분류기 예측 중 오류: {e}")
                import traceback
                traceback.print_exc()
                self._last_predict_error_time = time.time()
            
            if return_preprocessed:
                return ('background', 0.0, 0.0, 0.0, None)
            else:
                return ('background', 0.0, 0.0, 0.0)


class InspectionThread(QThread):
    """
    카메라 제어 및 배터리 검사 로직 실행 스레드
    
    이 클래스는 Basler 카메라로부터 영상을 받아 AI 기반 배터리 불량 검사를 수행합니다.
    주요 기능:
    - 실시간 카메라 영상 처리
    - 배터리 감지 및 ROI 추출
    - AI 모델을 통한 불량 분류
    - OpenCV 기반 크랙 검출 (하이브리드 검사)
    - 판정 결과 안정화 (깜빡임 방지)
    
    Signals:
        change_pixmap_signal: 카메라 영상 전송 신호
        result_signal: 검사 결과 전송 신호
        defect_detail_signal: 불량 상세 정보 전송 신호
        camera_connected_signal: 카메라 연결 상태 신호
        preprocessed_image_signal: AI 전처리 이미지 전송 신호
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, dict, object)  # (res, errors, battery_rect)
    defect_detail_signal = pyqtSignal(list)
    camera_connected_signal = pyqtSignal(bool)
    preprocessed_image_signal = pyqtSignal(np.ndarray)  # AI가 보고 있는 전처리된 이미지
    
    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.camera = None
        self._camera_connected = False
        self.classifier = None
        self.confidence_threshold = 0.5  # 50% 이상 (배경 필터링 강화: 너무 낮으면 배경도 인식)
        self._confidence_history = []  # confidence 히스토리 (안정화용)
        self._stable_confidence = 0.0  # 안정화된 confidence 값
        self._battery_detection_history = []  # 배터리 인식 히스토리 (안정화용)
        
        # 판정 깜빡임 방지: 최근 프레임들의 판정 결과 버퍼링 (다수결 방식)
        self._result_buffer = []  # 최근 판정 결과 저장 (OK, NG, NO_BATTERY)
        self._defect_type_buffer = []  # NG일 때의 불량 타입 저장 (crack, pollution 등)
        self._errors_buffer = []  # 최근 errors 딕셔너리 저장
        self._defects_buffer = []  # 최근 defects 리스트 저장
        self._battery_rect_buffer = []  # 최근 battery_rect 저장
        self._buffer_size = 10  # 10프레임 (약 0.3초) 동안 모아서 판단 (안정화 강화)
        
        # 오염(Pollution) 깜빡임 방지: 연속 검출 카운터
        self._pollution_trigger_count = 0  # 오염 연속 검출 횟수 카운터
        self._pollution_trigger_threshold = 5  # 5프레임 연속이어야 인정
        
        # 카메라 설정 변수 (기본값)
        self.camera_exposure = 10000  # 마이크로초 단위 (10ms)
        self.camera_width = 4096  # 기본 너비 (4K)
        self.camera_height = 2160  # 기본 높이 (4K)
    
    def _classify_defect_type(self, frame, x, y, w, h, prob_defect, prob_normal, classifier=None, roi_image=None):
        """
        하자 타입을 분류 (크랙, 스크래치, 오염)
        
        Args:
            frame: 전체 프레임
            x, y, w, h: 하자 영역 바운딩 박스
            prob_defect: defect 확률
            prob_normal: normal 확률
            classifier: MobileNetV3Classifier 객체 (3개 클래스 모델인 경우 사용)
            roi_image: ROI 이미지 (이미 추출된 경우)
            
        Returns:
            str: 'crack', 'scratch', 'color', 'damaged', 'pollution', 또는 'defect' (구분 불가능할 때)
        """
        # 3개 클래스 모델인 경우 모델 출력 직접 사용
        if classifier is not None and hasattr(classifier, 'class_names'):
            if classifier.class_names == ['Normal', 'Damaged', 'Pollution']:
                # 3개 클래스 모델: 모델 출력 직접 사용
                if roi_image is not None:
                    try:
                        result = classifier.predict(roi_image)
                        if len(result) >= 2:
                            class_name, confidence = result[0], result[1]
                            # 모델이 직접 구분하므로 그대로 반환 (매핑 필요)
                            if class_name == 'Damaged':
                                # Damaged는 크랙/스크래치로 매핑 가능
                                # 추가 이미지 분석으로 구분 시도
                                try:
                                    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                                    edges = cv2.Canny(gray, 50, 150)
                                    edge_ratio = np.sum(edges > 0) / (w * h)
                                    if edge_ratio > 0.20 and w * h < 300:
                                        return "crack"
                                    else:
                                        return "scratch"
                                except:
                                    return "damaged"
                            elif class_name == 'Pollution':
                                return "color"  # Pollution = 오염
                            else:
                                return "defect"
                    except Exception as e:
                        pass  # 모델 예측 실패 시 이미지 분석으로 fallback
        
        # 2개 클래스 모델이거나 모델 예측 실패 시 이미지 분석 기반 추정
        try:
            # ROI 추출
            if roi_image is not None:
                roi = roi_image
            else:
                roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return "defect"
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 엣지 검출 (크랙/스크래치 특징)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / (w * h)
            
            # HSV 색상 분석
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_std = np.std(hsv[:, :, 0])  # Hue 채널 분산
            s_std = np.std(hsv[:, :, 1])  # Saturation 채널 분산
            v_mean = np.mean(hsv[:, :, 2])  # Value 채널 평균 (밝기)
            
            # 밝기 분석 (커버가 까인 경우 밝기가 변함)
            gray_mean = np.mean(gray)
            gray_std = np.std(gray)
            
            # 커버가 까인 것(스크래치) vs 오염 구분
            # 스크래치: 엣지가 많고, 밝기 변화가 크고, 색상 분산은 작음
            # 오염: 엣지는 적고, 색상 분산이 크고, 밝기 변화는 작음
            
            # 크랙 판정: 엣지 비율이 매우 높고, 면적이 작음
            if edge_ratio > 0.20 and w * h < 300:
                return "crack"
            # 스크래치 판정: 엣지 비율이 중간이고, 밝기 변화가 크고, 색상 분산이 작음
            elif edge_ratio > 0.08 and gray_std > 25 and h_std < 20:
                return "scratch"
            # 오염 판정: 엣지 비율이 낮고, 색상 분산이 크고, 밝기 변화가 작음
            elif edge_ratio < 0.10 and h_std > 25 and gray_std < 20:
                return "color"
            # 구분 불가능하면 일반 defect
            else:
                return "defect"
        except Exception as e:
            # 오류 발생 시 일반 defect 반환
            return "defect"
        
    def run(self):
        """
        메인 실행 루프 (QThread 오버라이드)
        
        이 메서드는 카메라로부터 영상을 받아 실시간 배터리 검사를 수행합니다.
        처리 흐름:
        1. 분류기 모델 로드
        2. 카메라 연결
        3. 카메라 설정 적용
        4. 실시간 프레임 처리 루프
            - 프레임 캡처
            - 배터리 검사 수행
            - 결과 버퍼링 및 안정화
            - UI 신호 전송
        5. 스레드 종료 시 정리
        """
        try:
            print("=" * 60)
            print("[INFO] InspectionThread.run() 시작")
            print("=" * 60)
            self._run_flag = True
            
            # 모델 로드
            print("[INFO] 모델 로드 호출 전")
            print(f"   CLASSIFIER_AVAILABLE: {CLASSIFIER_AVAILABLE}")
            self._load_classifier()
            print(f"[INFO] 모델 로드 호출 후: classifier={self.classifier is not None}")
            if self.classifier is None:
                print("[WARNING] classifier가 None입니다!")
            
            # 카메라 연결
            print("[INFO] 카메라 연결 시도")
            if not self._connect_camera():
                print("[ERROR] 카메라 연결 실패")
                return
            print("[SUCCESS] 카메라 연결 성공")
        except Exception as e:
            print(f"[ERROR] run() 메서드 초기화 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 이미지 변환기
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # 메인 루프
        while self._run_flag and self.camera and self.camera.IsGrabbing():
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grabResult.GrabSucceeded():
                    try:
                        image = converter.Convert(grabResult)
                        frame = image.GetArray()
                        
                        if frame is None or frame.size == 0:
                            grabResult.Release()
                            continue
                        
                        # 프레임 크기 조정 (카메라 해상도에 맞춤, 표시용으로 640x480으로 조정)
                        # 카메라 해상도로 캡처된 프레임을 표시용 크기로 조정
                        target_display_width = 640
                        target_display_height = 480
                        frame = cv2.resize(frame, (target_display_width, target_display_height))
                        display_frame = frame.copy()
                        
                        # classifier 로드 확인 및 예외 처리 강화
                        if self.classifier is None:
                            # classifier가 로드되지 않았으면 에러 메시지 표시
                            cv2.putText(display_frame, "CLASSIFIER NOT LOADED", (50, 240), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if self._camera_connected:
                                self.change_pixmap_signal.emit(display_frame)
                                self.result_signal.emit("ERROR", {"defect": False}, None)
                            grabResult.Release()
                            self.msleep(30)
                            continue
                        
                        # 배터리 검사
                        result, errors, battery_rect, defects = self._inspect_frame(frame, display_frame)
                        
                        # 불량 타입 추출 (NG인 경우)
                        defect_type = None
                        if result == "NG" and len(defects) > 0:
                            defect_type = defects[0].get('type', None)
                        elif result == "NG" and errors:
                            # errors에서 불량 타입 추출
                            if errors.get('crack', False):
                                defect_type = 'crack'
                            elif errors.get('color', False):
                                defect_type = 'pollution'
                            elif errors.get('scratch', False):
                                defect_type = 'scratch'
                        
                        # 판정 결과 버퍼에 추가 (깜빡임 방지)
                        self._result_buffer.append(result)
                        self._defect_type_buffer.append(defect_type)
                        self._errors_buffer.append(errors)
                        self._defects_buffer.append(defects)
                        self._battery_rect_buffer.append(battery_rect)
                        
                        # 버퍼 크기 제한
                        if len(self._result_buffer) > self._buffer_size:
                            self._result_buffer.pop(0)
                            self._defect_type_buffer.pop(0)
                            self._errors_buffer.pop(0)
                            self._defects_buffer.pop(0)
                            self._battery_rect_buffer.pop(0)
                        
                        # 다수결로 최종 결과 결정
                        stable_result, stable_errors, stable_battery_rect, stable_defects = self._get_majority_result()
                        
                        # UI 업데이트 (다수결로 결정된 안정화된 결과만 전달)
                        if self._camera_connected:
                            self.change_pixmap_signal.emit(display_frame)
                            self.result_signal.emit(stable_result, stable_errors, stable_battery_rect)
                            if stable_result in ["OK", "NG"]:
                                self.defect_detail_signal.emit(stable_defects if stable_defects else [])
                            else:
                                self.defect_detail_signal.emit([])
                            
                            # 전처리된 이미지도 매 프레임마다 업데이트 (안정적으로)
                            if hasattr(self, '_current_frame_preprocessed_img') and self._current_frame_preprocessed_img is not None:
                                self.preprocessed_image_signal.emit(self._current_frame_preprocessed_img)
                    
                    except Exception as e:
                        print(f"[WARNING] 이미지 처리 중 오류: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        grabResult.Release()
                
                self.msleep(30)
            
            except pylon.TimeoutException:
                print("[WARNING] 카메라 타임아웃")
                continue
            except Exception as e:
                print(f"[WARNING] 프레임 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 정리
        self._disconnect_camera()
    
    def _load_classifier(self):
        """
        AI 분류기 모델 로드 / Load AI classifier model
        
        사전 학습된 MobileNetV3 모델을 로드합니다.
        Loads pre-trained MobileNetV3 classifier.
        
        모델 파일 검색 순서 (Model path search order):
        1. 실행 파일(exe) 경로 / Executable directory
        2. 스크립트 파일 경로 / Script directory (inspection_app 폴더)
        3. MobileNetV3/runs 디렉토리 / MobileNetV3/runs directory
        
        모델 로드 실패 시 classifier=None, 검사 중단 / On failure: classifier=None, inspection disabled
        """
        try:
            print("[INFO] 분류기 모델 로드 시작...", flush=True)
            sys.stdout.flush()
            print(f"   CLASSIFIER_AVAILABLE: {CLASSIFIER_AVAILABLE}", flush=True)
            sys.stdout.flush()
            if not CLASSIFIER_AVAILABLE:
                print("[WARNING] PyTorch를 사용할 수 없습니다.")
                return
            
            model_path = None
            
            # 1순위: 실행 파일 디렉토리
            try:
                exe_dir = Path(sys.executable).resolve().parent
                exe_model_path = exe_dir / "model_classifier_best.pth"
                print(f"[INFO] 모델 경로 확인 1: {exe_model_path} (존재: {exe_model_path.exists()})")
                if exe_model_path.exists():
                    model_path = exe_model_path
                    print(f"[SUCCESS] 모델 파일 발견: {model_path}")
            except Exception as e:
                print(f"[WARNING] 경로 확인 실패 1: {e}")
            
            # 2순위: 스크립트 경로
            if model_path is None or not model_path.exists():
                try:
                    script_model_path = Path(__file__).resolve().parent / "model_classifier_best.pth"
                    print(f"[INFO] 모델 경로 확인 2: {script_model_path} (존재: {script_model_path.exists()})")
                    if script_model_path.exists():
                        model_path = script_model_path
                        print(f"[SUCCESS] 모델 파일 발견: {model_path}")
                except Exception as e:
                    print(f"[WARNING] 경로 확인 실패 2: {e}")
            
            # 3순위: MobileNetV3/runs
            if model_path is None or not model_path.exists():
                try:
                    mobile_net_path = Path(__file__).resolve().parent.parent / "MobileNetV3" / "runs"
                    runs_model_path = mobile_net_path / "model_classifier_best.pth"
                    print(f"[INFO] 모델 경로 확인 3: {runs_model_path} (존재: {runs_model_path.exists()})", flush=True)
                    sys.stdout.flush()
                    if runs_model_path.exists():
                        model_path = runs_model_path
                        print(f"[SUCCESS] 모델 파일 발견: {model_path}", flush=True)
                        sys.stdout.flush()
                except Exception as e:
                    print(f"[WARNING] 경로 확인 실패 3: {e}")
            
            # 모델 로드
            if model_path and model_path.exists():
                try:
                    print(f"[INFO] 모델 로드 시도: {model_path}", flush=True)
                    sys.stdout.flush()
                    self.classifier = MobileNetV3Classifier(model_path=str(model_path))
                    print(f"[SUCCESS] 분류기 모델 로드 완료: {model_path}", flush=True)
                    sys.stdout.flush()
                    print(f"[SUCCESS] classifier 객체 생성 확인: {self.classifier is not None}", flush=True)
                    sys.stdout.flush()
                    if self.classifier is not None:
                        print(f"[SUCCESS] 모델 정보:")
                        print(f"   - 모델 타입: {type(self.classifier.model).__name__}")
                        print(f"   - 실제 클래스 수: {len(self.classifier.class_names)}")
                        print(f"   - 클래스 목록: {self.classifier.class_names}")
                        print(f"   - 디바이스: {self.classifier.device}")
                        print(f"   - 모델 파라미터 수: {sum(p.numel() for p in self.classifier.model.parameters()):,}")
                        print(f"   [SUCCESS] 학습된 모델({model_path.name})이 성공적으로 로드되어 사용 중입니다!")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[WARNING] 모델 로드 실패: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("[ERROR] 분류기 모델 파일을 찾을 수 없습니다.")
                print(f"   현재 스크립트 경로: {Path(__file__).resolve()}")
                print(f"   스크립트 부모 경로: {Path(__file__).resolve().parent}")
                print(f"   MobileNetV3 예상 경로: {Path(__file__).resolve().parent.parent / 'MobileNetV3' / 'runs'}")
        except Exception as e:
            print(f"[ERROR] _load_classifier() 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _connect_camera(self):
        """
        Basler 카메라 연결
        
        이 메서드는 Basler Pylon SDK를 사용하여 카메라에 연결합니다.
        
        Returns:
            bool: 연결 성공 시 True, 실패 시 False
        """
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                print("🚨 연결된 Basler 카메라가 없습니다!")
                self._camera_connected = False
                self.camera_connected_signal.emit(False)
                return False
            
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            
            # 카메라 설정 적용
            self._apply_camera_settings()
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            self._camera_connected = True
            self.camera_connected_signal.emit(True)
            print("[SUCCESS] 카메라 연결 성공")
            return True
        
        except Exception as e:
            print(f"🚨 카메라 연결 실패: {e}")
            self._camera_connected = False
            self.camera_connected_signal.emit(False)
            return False
    
    def _apply_camera_settings(self):
        """
        카메라 설정 적용 (Exposure)
        
        이 메서드는 카메라 노출 시간(Exposure)을 설정합니다.
        자동 노출을 비활성화하고 수동 모드로 전환합니다.
        """
        if self.camera is None or not self.camera.IsOpen():
            return
        
        try:
            # Grabbing 중인지 확인
            was_grabbing = False
            if self.camera.IsGrabbing():
                was_grabbing = True
                self.camera.StopGrabbing()
            
            # ExposureAuto를 Off로 설정 (수동 제어를 위해)
            # IEnumeration 타입은 IsWritable()가 없으므로 직접 try-except로 처리
            try:
                if hasattr(self.camera, 'ExposureAuto'):
                    self.camera.ExposureAuto.SetValue('Off')
                    print("[INFO] ExposureAuto: Off (수동 모드)")
            except Exception as e:
                print(f"[WARNING] ExposureAuto 설정 실패: {e}")
            
            # Exposure (노출 시간) 설정
            try:
                if hasattr(self.camera, 'ExposureTime') and self.camera.ExposureTime.IsWritable():
                    # 카메라가 지원하는 최대/최소 값 확인
                    exp_max = self.camera.ExposureTime.GetMax()
                    exp_min = self.camera.ExposureTime.GetMin()
                    exp_value = max(exp_min, min(exp_max, self.camera_exposure))
                    self.camera.ExposureTime.SetValue(exp_value)
                    self.camera_exposure = exp_value
                    print(f"[INFO] Exposure 설정: {exp_value} μs ({exp_value/1000:.2f} ms)")
                elif hasattr(self.camera, 'ExposureTimeRaw') and self.camera.ExposureTimeRaw.IsWritable():
                    # ExposureTimeRaw 사용 (일부 카메라 모델)
                    self.camera.ExposureTimeRaw.SetValue(self.camera_exposure)
                    print(f"[INFO] ExposureRaw 설정: {self.camera_exposure} μs")
            except Exception as e:
                print(f"[WARNING] Exposure 설정 실패: {e}")
            
            # Width, Height (해상도) 설정 (카메라 연결 시 반드시 적용)
            try:
                if hasattr(self.camera, 'Width') and self.camera.Width.IsWritable():
                    width_max = self.camera.Width.GetMax()
                    width_min = self.camera.Width.GetMin()
                    width_value = max(width_min, min(width_max, self.camera_width))
                    self.camera.Width.SetValue(width_value)
                    self.camera_width = width_value
                    print(f"[INFO] Width 설정: {width_value} (범위: {width_min}~{width_max})")
                if hasattr(self.camera, 'Height') and self.camera.Height.IsWritable():
                    height_max = self.camera.Height.GetMax()
                    height_min = self.camera.Height.GetMin()
                    height_value = max(height_min, min(height_max, self.camera_height))
                    self.camera.Height.SetValue(height_value)
                    self.camera_height = height_value
                    print(f"[INFO] Height 설정: {height_value} (범위: {height_min}~{height_max})")
                
                # 해상도 변경 후 안정화를 위한 짧은 대기
                self.msleep(100)
            except Exception as e:
                print(f"[WARNING] 해상도 설정 실패: {e}")
                import traceback
                traceback.print_exc()
            
            # Grabbing 재시작
            if was_grabbing and self.camera and self.camera.IsOpen():
                try:
                    if not self.camera.IsGrabbing():
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        print("[SUCCESS] 카메라 Grabbing 재시작 완료")
                except Exception as e:
                    print(f"[WARNING] Grabbing 재시작 실패: {e}")
                    self._camera_connected = False
                    self.camera_connected_signal.emit(False)
        
        except Exception as e:
            print(f"[WARNING] 카메라 설정 적용 실패: {e}")
            import traceback
            traceback.print_exc()
            # 예외 발생 시에도 Grabbing 재시작 시도
            if was_grabbing and self.camera and self.camera.IsOpen():
                try:
                    if not self.camera.IsGrabbing():
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                except:
                    pass
    
    def update_camera_exposure(self, exposure):
        """
        카메라 노출 시간 업데이트 (마이크로초 단위)
        
        이 메서드는 카메라 노출 시간을 동적으로 변경합니다.
        먼저 "on-the-fly" 방식(그래빙 중지 없이)으로 변경을 시도하고,
        실패 시 Stop/Start 방식으로 재시도합니다.
        
        Args:
            exposure: 노출 시간 (마이크로초 단위)
        """
        print(f"[DEBUG] update_camera_exposure 호출: {exposure} μs")
        self.camera_exposure = exposure
        
        if self.camera is None:
            print("[WARNING] 카메라가 None입니다")
            return
        
        if not self.camera.IsOpen():
            print("[WARNING] 카메라가 열려있지 않습니다")
            return
        
        try:
            # ExposureAuto를 Off로 설정 (매번 확인)
            try:
                if hasattr(self.camera, 'ExposureAuto'):
                    self.camera.ExposureAuto.SetValue('Off')
                    print("[INFO] ExposureAuto: Off 설정 완료")
            except Exception as e:
                print(f"[WARNING] ExposureAuto 설정 실패: {e}")
            
            # On-the-fly 변경 시도: Grabbing 중지 없이 바로 적용
            try:
                # ExposureTime 속성 확인 및 설정
                if hasattr(self.camera, 'ExposureTime'):
                    try:
                        # IsWritable 확인 없이 직접 시도 (일부 카메라는 IsWritable이 없을 수 있음)
                        exp_max = self.camera.ExposureTime.GetMax()
                        exp_min = self.camera.ExposureTime.GetMin()
                        exp_value = max(exp_min, min(exp_max, float(exposure)))
                        self.camera.ExposureTime.SetValue(exp_value)
                        self.camera_exposure = exp_value
                        print(f"[SUCCESS] Exposure 업데이트 (On-the-fly): {exp_value} μs ({exp_value/1000:.2f} ms)")
                        return  # 성공하면 여기서 종료
                    except Exception as e1:
                        print(f"[WARNING] ExposureTime 직접 설정 실패: {e1}")
                        # IsWritable 확인 후 재시도
                        if hasattr(self.camera.ExposureTime, 'IsWritable') and self.camera.ExposureTime.IsWritable():
                            exp_max = self.camera.ExposureTime.GetMax()
                            exp_min = self.camera.ExposureTime.GetMin()
                            exp_value = max(exp_min, min(exp_max, float(exposure)))
                            self.camera.ExposureTime.SetValue(exp_value)
                            self.camera_exposure = exp_value
                            print(f"[SUCCESS] Exposure 업데이트 (On-the-fly, IsWritable 확인 후): {exp_value} μs")
                            return
                elif hasattr(self.camera, 'ExposureTimeRaw'):
                    try:
                        self.camera.ExposureTimeRaw.SetValue(float(exposure))
                        print(f"[SUCCESS] ExposureRaw 업데이트 (On-the-fly): {exposure} μs")
                        return
                    except Exception as e2:
                        print(f"[WARNING] ExposureTimeRaw 설정 실패: {e2}")
            except Exception as e:
                # On-the-fly 변경 실패 시 기존 방식(Stop -> Set -> Start)으로 재시도
                print(f"[WARNING] On-the-fly Exposure 변경 실패, Stop/Start 방식으로 재시도: {e}")
                import traceback
                traceback.print_exc()
            
            # 기존 방식: Grabbing 중지 후 설정 적용
            was_grabbing = False
            if self.camera.IsGrabbing():
                was_grabbing = True
                print("[INFO] Grabbing 중지 중...")
                self.camera.StopGrabbing()
                self.msleep(50)
            
            # Exposure 설정 (Stop/Start 방식)
            try:
                if hasattr(self.camera, 'ExposureTime'):
                    exp_max = self.camera.ExposureTime.GetMax()
                    exp_min = self.camera.ExposureTime.GetMin()
                    exp_value = max(exp_min, min(exp_max, float(exposure)))
                    self.camera.ExposureTime.SetValue(exp_value)
                    self.camera_exposure = exp_value
                    print(f"[SUCCESS] Exposure 업데이트 (Stop/Start): {exp_value} μs ({exp_value/1000:.2f} ms)")
                elif hasattr(self.camera, 'ExposureTimeRaw'):
                    self.camera.ExposureTimeRaw.SetValue(float(exposure))
                    print(f"[SUCCESS] ExposureRaw 업데이트 (Stop/Start): {exposure} μs")
                else:
                    print("[WARNING] ExposureTime 또는 ExposureTimeRaw 속성을 찾을 수 없습니다")
            except Exception as e:
                print(f"[ERROR] Exposure 설정 실패 (Stop/Start): {e}")
                import traceback
                traceback.print_exc()
            
            # Grabbing 재시작
            if was_grabbing and self.camera and self.camera.IsOpen():
                try:
                    if not self.camera.IsGrabbing():
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        print("[SUCCESS] 카메라 Grabbing 재시작 완료")
                except Exception as e:
                    print(f"[WARNING] Grabbing 재시작 실패: {e}")
                    self._camera_connected = False
                    self.camera_connected_signal.emit(False)
                
        except Exception as e:
            print(f"[ERROR] Exposure 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _disconnect_camera(self):
        """
        카메라 연결 해제
        
        이 메서드는 카메라 그래빙을 중지하고 연결을 해제합니다.
        스레드 종료 시 자동으로 호출됩니다.
        """
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
                self.camera = None
            self._camera_connected = False
            self.camera_connected_signal.emit(False)
        except Exception:
            pass
    
    def _is_battery_present(self, roi):
        """
        배터리 존재 여부 확인 (HSV 색상 공간 기반 Rule-based Pre-filtering)
        
        ROI 이미지의 중앙 50% 영역을 HSV 색상 공간으로 변환하여
        채도(Saturation)와 명도(Value)의 평균값을 계산합니다.
        배경(무채색, 어두움)을 필터링하여 배터리 감지를 안정화합니다.
        
        기준:
        - 채도 평균 < 30 또는 명도 평균 < 60: 배터리 없음(배경)으로 판단
        
        Args:
            roi: numpy.ndarray (BGR 형식의 ROI 이미지)
        
        Returns:
            bool: True면 배터리 있음, False면 배터리 없음(배경)
        """
        try:
            h, w = roi.shape[:2]
            
            # 중앙 50% 영역만 추출 (가장자리 배경 배제)
            cy, cx = h // 2, w // 2
            h_crop, w_crop = h // 2, w // 2
            
            # 중앙 영역 좌표 계산
            y1 = max(0, cy - h_crop // 2)
            y2 = min(h, cy + h_crop // 2)
            x1 = max(0, cx - w_crop // 2)
            x2 = min(w, cx + w_crop // 2)
            
            # 중앙 50% 영역만 추출
            center_roi = roi[y1:y2, x1:x2]
            
            # BGR -> HSV 변환 (색상 공간 변환)
            hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
            
            # 채도(Saturation) 채널의 평균값 계산 (유채색 여부)
            mean_s = np.mean(hsv[:, :, 1])  # S 채널 평균
            
            # 명도(Value) 채널의 평균값 계산 (밝기 여부)
            mean_v = np.mean(hsv[:, :, 2])  # V 채널 평균
            
            # 배터리 감지 기준 강화 (배경 완벽 차단):
            # - 채도 평균이 30 미만: 무채색(배경)으로 판단
            # - 명도 평균이 60 미만: 너무 어두워서 배터리 아님 (50 -> 60으로 강화)
            # 이 조건을 만족하면 AI 추론을 아예 생략하고 즉시 NO_BATTERY 상태로 처리
            if mean_s < 30 or mean_v < 60:
                return False  # 배터리 없음 (배경)
            
            return True  # 배터리 있음
        
        except Exception as e:
            # 오류 발생 시 안전하게 False 반환 (배터리 없음으로 처리하여 NO_BATTERY 반환)
            print(f"[WARNING] _is_battery_present 오류: {e}")
            return False
    
    def _detect_crack_opencv(self, roi):
        """
        OpenCV를 사용한 크랙(구멍) 검출 (Hybrid Inspection)
        AI 모델이 놓치는 작은 구멍을 OpenCV로 보완
        노이즈 과검(Overkill) 문제 해결: 강력한 필터링 적용
        
        Args:
            roi: ROI 이미지 (BGR)
        
        Returns:
            tuple: (is_crack, valid_cracks) 
                - is_crack: 크랙이 감지되었으면 True
                - valid_cracks: 감지된 크랙의 윤곽선 리스트 [contour, ...]
        """
        try:
            # 1. 그레이스케일 변환
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 2. 강력한 노이즈 제거 (Blur 커널 키움)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # 7x7 블러 (노이즈 제거)
            
            # 3. 이진화 (Adaptive Threshold) - 파라미터 튜닝
            thresh = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV,  # 반전: 어두운 영역을 흰색으로
                29,  # Block Size: 29 (더 큰 영역)
                6    # C: 6 (균형잡힌 기준)
            )
            
            # 4. 형태학적 연산 - 자잘한 점 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 5. 윤곽선 검출
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_cracks = []
            h, w = roi.shape[:2]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # [필터 1] 면적 제한: 50-1500 (균형잡힌 기준)
                if area < 50 or area > 1500:
                    continue
                
                # [필터 2] 둥근 정도(Circularity) 체크
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 원형도가 0.3 미만이면(길쭉하면) 스크래치나 글자로 간주
                if circularity < 0.3:
                    continue
                
                # [필터 3] ROI 가장자리 제외 (빛 반사 오인식 방지)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                margin = 20  # 가장자리 20픽셀
                if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
                    continue
                
                valid_cracks.append(cnt)
            
            is_crack = len(valid_cracks) > 0
            
            if is_crack:
                areas = [cv2.contourArea(cnt) for cnt in valid_cracks]
                # 로그 빈도 제한 (1초마다 한 번만 출력)
                if not hasattr(self, '_last_precise_log_time') or time.time() - self._last_precise_log_time > 1.0:
                    print(f"[INFO] 정밀 크랙 감지: {len(valid_cracks)}개 (필터링 통과, 면적 범위: {min(areas):.1f}-{max(areas):.1f} 픽셀)")
                    self._last_precise_log_time = time.time()
            
            return (is_crack, valid_cracks)
            
        except Exception as e:
            print(f"[ERROR] _detect_crack_opencv 오류: {e}")
            import traceback
            traceback.print_exc()
            return (False, [])
    
    def _get_majority_result(self):
        """
        버퍼에서 다수결로 최종 결과 결정 (깜빡임 방지)
        
        Returns:
            tuple: (result, errors, battery_rect, defects) - 다수결로 결정된 결과
        """
        # 버퍼가 비어있으면 기본값 반환
        if len(self._result_buffer) == 0:
            return ("NO_BATTERY", {"defect": False}, None, [])
        
        # 버퍼가 충분히 채워지지 않았으면 최신 결과 반환
        if len(self._result_buffer) < self._buffer_size:
            last_idx = len(self._result_buffer) - 1
            return (self._result_buffer[last_idx], 
                   self._errors_buffer[last_idx] if last_idx < len(self._errors_buffer) else {"defect": False},
                   self._battery_rect_buffer[last_idx] if last_idx < len(self._battery_rect_buffer) else None,
                   self._defects_buffer[last_idx] if last_idx < len(self._defects_buffer) else [])
        
        # 최근 5프레임의 결과만 사용
        recent_results = self._result_buffer[-self._buffer_size:]
        recent_defect_types = self._defect_type_buffer[-self._buffer_size:] if len(self._defect_type_buffer) >= self._buffer_size else []
        
        # 다수결 계산
        from collections import Counter
        result_counts = Counter(recent_results)
        majority_result = result_counts.most_common(1)[0][0]  # 가장 많이 나온 결과
        
        # NG인 경우, 불량 타입도 다수결로 결정
        majority_defect_type = None
        if majority_result == "NG" and len(recent_defect_types) >= self._buffer_size:
            # NG인 프레임들만 필터링
            ng_defect_types = [dt for i, dt in enumerate(recent_defect_types[-self._buffer_size:]) 
                             if recent_results[i] == "NG"]
            if len(ng_defect_types) > 0:
                defect_type_counts = Counter(ng_defect_types)
                majority_defect_type = defect_type_counts.most_common(1)[0][0]
        
        # 다수결 결과에 해당하는 최신 데이터 찾기
        for i in range(len(self._result_buffer) - 1, -1, -1):
            if self._result_buffer[i] == majority_result:
                # 불량 타입도 일치하는지 확인 (NG인 경우)
                if majority_result == "NG":
                    if majority_defect_type and i < len(self._defect_type_buffer):
                        # 해당 인덱스의 불량 타입 확인
                        defect_type = self._defect_type_buffer[i] if i < len(self._defect_type_buffer) else None
                        if defect_type == majority_defect_type:
                            return (majority_result, 
                                   self._errors_buffer[i] if i < len(self._errors_buffer) else {"defect": False},
                                   self._battery_rect_buffer[i] if i < len(self._battery_rect_buffer) else None,
                                   self._defects_buffer[i] if i < len(self._defects_buffer) else [])
                    else:
                        # 불량 타입이 없거나 일치하면 반환
                        return (majority_result, 
                               self._errors_buffer[i] if i < len(self._errors_buffer) else {"defect": False},
                               self._battery_rect_buffer[i] if i < len(self._battery_rect_buffer) else None,
                               self._defects_buffer[i] if i < len(self._defects_buffer) else [])
                else:
                    # OK나 NO_BATTERY인 경우 바로 반환
                    return (majority_result, 
                           self._errors_buffer[i] if i < len(self._errors_buffer) else {"defect": False},
                           self._battery_rect_buffer[i] if i < len(self._battery_rect_buffer) else None,
                           self._defects_buffer[i] if i < len(self._defects_buffer) else [])
        
        # 찾지 못한 경우 (예외 상황) 최신 결과 반환
        last_idx = len(self._result_buffer) - 1
        return (self._result_buffer[last_idx], 
               self._errors_buffer[last_idx] if last_idx < len(self._errors_buffer) else {"defect": False},
               self._battery_rect_buffer[last_idx] if last_idx < len(self._battery_rect_buffer) else None,
               self._defects_buffer[last_idx] if last_idx < len(self._defects_buffer) else [])
    
    def _inspect_frame(self, frame, display_frame):
        """
        프레임 단위 배터리 불량 검사 메서드 (Hybrid 방식)
        
        이 메서드는 한 프레임에 대해 다음 단계를 수행합니다:
        1. 배터리 후보 감지 (고정 ROI 사용)
        2. 배터리 존재 확인 (HSV 기반 Rule-based Filtering)
        3. OpenCV 기반 크랙 검출 (하이브리드 검사)
        4. AI 모델을 통한 불량 분류 (Normal, Damaged, Pollution)
        5. 판정 로직 적용 (임계값 기반 OK/NG 판정)
        6. 오염(Pollution) 깜빡임 방지 (5프레임 연속 검출 필요)
        
        Args:
            frame: numpy.ndarray (원본 카메라 프레임, BGR 형식)
            display_frame: numpy.ndarray (UI 표시용 프레임, BGR 형식)
        
        Returns:
            tuple: (result, errors, battery_rect, defects)
                - result (str): 검사 결과 ("OK", "NG", "NO_BATTERY", "ERROR")
                - errors (dict): 불량 타입 플래그 딕셔너리
                - battery_rect (tuple): 배터리 ROI 좌표 (x, y, w, h) 또는 None
                - defects (list): 불량 상세 정보 리스트
        """
        # Step 1: 고정 영역 스캔 모드 - 화면 중앙 고정 ROI만 사용
        candidates = self._detect_battery_candidates(frame)
        
        # 고정 스캔 모드에서는 항상 후보가 1개 있음
        if len(candidates) == 0:
            # 예외 상황 (프레임 크기가 너무 작은 경우)
            cv2.putText(display_frame, "FRAME TOO SMALL", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return ("ERROR", {"defect": False}, None, [])
        
        # Step 2: AI 판독 (고정 ROI만 사용)
        best_candidate = None
        best_confidence = 0.0
        best_class = None
        
        if self.classifier is None:
            # 분류기가 없으면 검사 불가
            if not hasattr(self, '_last_no_classifier_warning') or time.time() - self._last_no_classifier_warning > 10.0:
                print("[WARNING] 분류기 모델이 없습니다. 검사를 수행할 수 없습니다.")
                print("   모델 파일 위치 확인:")
                print(f"   1. {Path(sys.executable).resolve().parent / 'model_classifier_best.pth'}")
                print(f"   2. {Path(__file__).resolve().parent / 'model_classifier_best.pth'}")
                print(f"   3. {Path(__file__).resolve().parent.parent / 'MobileNetV3' / 'runs' / 'model_classifier_best.pth'}")
                # 실제 파일 존재 여부 확인
                path1 = Path(sys.executable).resolve().parent / "model_classifier_best.pth"
                path2 = Path(__file__).resolve().parent / "model_classifier_best.pth"
                path3 = Path(__file__).resolve().parent.parent / "MobileNetV3" / "runs" / "model_classifier_best.pth"
                print(f"   실제 파일 존재 여부:")
                print(f"     1. 존재: {path1.exists()}")
                print(f"     2. 존재: {path2.exists()}")
                print(f"     3. 존재: {path3.exists()}")
                print(f"   현재 classifier 상태: {self.classifier}")
                self._last_no_classifier_warning = time.time()
            cv2.putText(display_frame, "CLASSIFIER NOT LOADED", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return ("ERROR", {"defect": False}, None, [])
        else:
            # 고정 ROI에 대해 AI 판독 (단일 후보만 처리)
            x, y, w, h, score = candidates[0]
            try:
                # ROI 추출 (패딩 없이 정확히 고정 영역만)
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0 or len(roi.shape) != 3:
                    cv2.putText(display_frame, "ROI EXTRACTION FAILED", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return ("ERROR", {"defect": False}, None, [])
                
                # Step 1: 배터리 존재 확인 (Rule-based Pre-filtering, 강화됨)
                # 논문 기반: Two-stage Detection Strategy
                # HSV 분석을 통해 배경 완벽 차단
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_saturation = np.mean(hsv_roi[:, :, 1])  # 채도(S)
                mean_value = np.mean(hsv_roi[:, :, 2])  # 밝기(V)
                
                # HSV 값 저장 (UI 디버깅 정보 표시용)
                self._last_hsv_s = mean_saturation
                self._last_hsv_v = mean_value
                
                is_battery = self._is_battery_present(roi)
                if not is_battery:
                    # 배터리가 없으면 AI 추론을 건너뛰고 즉시 NO_BATTERY 상태로 처리
                    # 가이드 박스 그리기 (회색)
                    x1 = max(0, min(int(x), display_frame.shape[1] - 1))
                    y1 = max(0, min(int(y), display_frame.shape[0] - 1))
                    x2 = max(0, min(int(x + w), display_frame.shape[1]))
                    y2 = max(0, min(int(y + h), display_frame.shape[0]))
                    if x2 > x1 and y2 > y1:
                        guide_color = (200, 200, 200)  # 회색
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), guide_color, 2)
                        # UI 피드백 강화: 상세 사유 표시
                        status_text = f"NO_BATTERY (Low Saturation: S={mean_saturation:.1f}, V={mean_value:.1f})"
                        cv2.putText(display_frame, status_text, 
                                    (x1, max(10, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
                    
                    # 화면 좌측 상단에 HSV 디버깅 정보 표시
                    debug_y = 30
                    hsv_debug_text = f"[Light: {mean_value:.0f} | Color: {mean_saturation:.0f}]"
                    (hsv_text_width, hsv_text_height), baseline = cv2.getTextSize(hsv_debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (10, debug_y - hsv_text_height - 3), (10 + hsv_text_width + 6, debug_y + 3), (0, 0, 0), -1)
                    cv2.putText(display_frame, hsv_debug_text, (13, debug_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)  # 회색 텍스트
                    
                    return ("NO_BATTERY", {"defect": False}, None, [])
                
                # ========================================================================
                # Step: OpenCV 크랙 검사 (Hybrid Inspection)
                # AI 판정 전에 OpenCV로 구멍(Crack) 검출 수행
                # AI가 놓치는 작은 구멍을 OpenCV로 보완
                # 하이브리드 불량 검출: OpenCV가 '명확한 구멍'을 찾으면 무조건 NG
                # ========================================================================
                is_crack_opencv, valid_cracks = self._detect_crack_opencv(roi)
                
                if is_crack_opencv:
                    # OpenCV가 크랙을 발견하면 AI 결과 무시하고 즉시 NG 처리
                    # 로그 빈도 제한 (1초마다 한 번만 출력)
                    if not hasattr(self, '_last_opencv_log_time') or time.time() - self._last_opencv_log_time > 1.0:
                        print(f"[WARNING] OpenCV 크랙 감지: AI 판정 무시하고 즉시 NG 처리")
                        self._last_opencv_log_time = time.time()
                    
                    # 전처리된 이미지 저장 (UI 업데이트용)
                    try:
                        if self.classifier is not None:
                            _, preprocessed_img = self.classifier.preprocess_image(roi)
                            if preprocessed_img is not None:
                                self._current_frame_preprocessed_img = preprocessed_img.copy()
                    except Exception as e:
                        print(f"[WARNING] OpenCV 크랙 감지 시 전처리 이미지 생성 실패: {e}")
                    
                    # 감지된 크랙 위치에 빨간 원 그리기 (시각화)
                    for contour in valid_cracks:
                        # 윤곽선의 중심 좌표 계산
                        M = cv2.moments(contour)
                        if M["m00"] == 0:
                            continue
                        
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(contour)
                        
                        # ROI 좌표를 전체 프레임 좌표로 변환
                        global_x = int(x + cx)
                        global_y = int(y + cy)
                        radius = max(5, int(np.sqrt(area / np.pi) * 0.8))  # 면적 기반 반지름
                        
                        # 화면 범위 확인
                        global_x = max(0, min(global_x, display_frame.shape[1] - 1))
                        global_y = max(0, min(global_y, display_frame.shape[0] - 1))
                        
                        # 빨간 원 그리기
                        cv2.circle(display_frame, (global_x, global_y), radius, (0, 0, 255), 2)  # 빨간색 원
                        cv2.circle(display_frame, (global_x, global_y), 3, (0, 0, 255), -1)  # 빨간색 중심점
                        
                        # 대미지 정보 텍스트 표시
                        text = f"Damage ({int(area)})"
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(display_frame, 
                                    (global_x - text_width // 2 - 2, global_y - radius - text_height - 5),
                                    (global_x + text_width // 2 + 2, global_y - radius - 1),
                                    (0, 0, 0), -1)
                        cv2.putText(display_frame, text, 
                                    (global_x - text_width // 2, global_y - radius - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # 가이드 박스 그리기 (빨간색)
                    x1 = max(0, min(int(x), display_frame.shape[1] - 1))
                    y1 = max(0, min(int(y), display_frame.shape[0] - 1))
                    x2 = max(0, min(int(x + w), display_frame.shape[1]))
                    y2 = max(0, min(int(y + h), display_frame.shape[0]))
                    if x2 > x1 and y2 > y1:
                        guide_color = (0, 0, 255)  # 빨간색
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), guide_color, 2)
                        
                        # 상태 텍스트 표시
                        status_text = f"NG (OpenCV Detected Damage: {len(valid_cracks)}개)"
                        cv2.putText(display_frame, status_text, 
                                    (x1, max(10, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
                    
                    # NG 결과 반환 (AI 판정 건너뛰기)
                    errors = {"crack": True, "scratch": False, "color": False, "defect": False}
                    defects = [{
                        'type': 'crack',
                        'bbox': (x, y, w, h),
                        'area': w * h,
                        'prob': 1.0,  # OpenCV 감지는 100% 확신
                        'method': 'opencv'  # OpenCV로 감지됨을 표시
                    }]
                    
                    return ("NG", errors, (x, y, w, h), defects)
                
                # AI 판독 (학습된 모델 사용) - 전처리된 이미지도 받기
                if self.classifier is None:
                    print("[ERROR] classifier가 None인데 predict 호출 시도!")
                    cv2.putText(display_frame, "CLASSIFIER NOT LOADED", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return ("ERROR", {"defect": False}, None, [])
                
                # 전처리된 이미지도 받기 (진단 모드: 모든 확률 포함)
                result = self.classifier.predict(roi, return_preprocessed=True)
                if len(result) >= 6:
                    class_name, confidence, prob_normal, prob_defect, preprocessed_img, all_probs = result[0], result[1], result[2], result[3], result[4], result[5]
                elif len(result) >= 5:
                    class_name, confidence, prob_normal, prob_defect, preprocessed_img = result[0], result[1], result[2], result[3], result[4]
                    all_probs = [prob_normal, prob_defect]  # 기본값
                else:
                    # 이전 버전 호환
                    class_name, confidence, prob_normal, prob_defect = result[:4]
                    preprocessed_img = None
                    all_probs = [prob_normal, prob_defect]  # 기본값
                
                # 전처리된 이미지 저장 (UI 업데이트용)
                if preprocessed_img is not None:
                    self._current_frame_preprocessed_img = preprocessed_img
                
                # 최선 후보 저장
                best_confidence = confidence
                best_candidate = (x, y, w, h)
                best_class = class_name
                # 모든 확률 저장 (진단 모드용)
                self._last_all_probs = all_probs
                self._last_prob_defect = prob_defect
                self._last_prob_normal = prob_normal
                
            except Exception as e:
                print(f"[ERROR] 고정 영역 판독 중 오류: {e}")
                cv2.putText(display_frame, f"ERROR: {str(e)[:30]}", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return ("ERROR", {"defect": False}, None, [])
            
            # Step 4: 결과 확정
            if best_candidate is None:
                # 후보가 없으면 NO_BATTERY 반환 (강제 중앙 ROI도 실패한 경우)
                cv2.putText(display_frame, "BATTERY NOT DETECTED", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                return ("NO_BATTERY", {"defect": False}, None, [])
        
        # 고정 영역 스캔 모드: 항상 고정 ROI 좌표 사용 (스무딩 불필요)
        if best_candidate is None:
            # 예외 상황 (이미 위에서 처리했지만 안전장치)
            return ("NO_BATTERY", {"defect": False}, None, [])
        
        x, y, w, h = best_candidate
        battery_rect = best_candidate  # 반환값용
        
        # 단순 판정 로직: 텍스트 오인식 방지 (confidence 90% 미만이면 OK)
        all_probs = getattr(self, '_last_all_probs', [])
        if self.classifier is not None:
            class_names = getattr(self.classifier, 'class_names', ['Normal', 'Damaged', 'Pollution'])
        else:
            class_names = ['Normal', 'Damaged', 'Pollution']
        
        # Normal 클래스 인덱스 찾기
        normal_idx = 0  # 기본값
        if 'Normal' in class_names:
            normal_idx = class_names.index('Normal')
        elif 'normal' in class_names:
            normal_idx = class_names.index('normal')
        
        # Normal 확률과 불량 확률 계산
        if len(all_probs) > normal_idx:
            p_good = all_probs[normal_idx]  # Normal 확률
            p_bad = 1.0 - p_good  # 불량 확률 (Damaged + Pollution 합계)
        else:
            # all_probs가 없으면 prob_normal 사용
            p_good = getattr(self, '_last_prob_normal', 0.0)
            p_bad = getattr(self, '_last_prob_defect', 0.0)
        
        # ========================================================================
        # 민감한 불량 검출 로직 (Sensitive Mode - Safety First)
        # 기조: "가짜 불량(과검)이 나오더라도, 진짜 불량은 절대 놓치지 않는다"
        # 비즈니스 요구사항: 
        # - Damaged(물리적 파손): 50%만 넘어도 즉시 NG (치명적)
        # - Pollution(오염): 60%만 넘어도 즉시 NG (AI가 크랙을 오염으로 착각할 때 70~80% 나오므로)
        # - Normal(정상): 70% 이상이어야만 OK (불량 끼가 조금이라도 보이면 NG)
        # - 애매한 경우: 안전하게 NG 처리 (Fail Safety)
        # ========================================================================
        
        # 확률 분해: all_probs에서 각 클래스별 확률 추출
        # class_names 순서가 ['Normal', 'Damaged', 'Pollution'] 이라고 가정
        idx_normal = 0
        idx_damaged = 1
        idx_pollution = 2
        
        if len(all_probs) >= 3:
            # 3개 클래스 모델: 각 확률을 직접 사용
            p_normal = all_probs[idx_normal]
            p_damaged = all_probs[idx_damaged]
            p_pollution = all_probs[idx_pollution]
        else:
            # fallback: all_probs가 없거나 3개 미만인 경우
            # best_class와 best_confidence로부터 추정
            if best_class == 'Normal':
                p_normal = best_confidence
                p_damaged = 0.0
                p_pollution = 0.0
            elif best_class == 'Damaged':
                p_normal = 0.0
                p_damaged = best_confidence
                p_pollution = 0.0
            elif best_class == 'Pollution':
                p_normal = 0.0
                p_damaged = 0.0
                p_pollution = best_confidence
            else:
                # 기타 경우: p_good, p_bad 사용
                p_normal = p_good
                p_damaged = p_bad * 0.5  # 추정값
                p_pollution = p_bad * 0.5  # 추정값
        
        # ========================================================================
        # 최종 로직: 배경 오인식 및 텍스트 과검(False Positive) 방지
        # 기조: "배경과 텍스트는 무시하고, 진짜 불량만 정확하게 잡는다"
        # ========================================================================
        
        # === 최종 판정 트리 (Robust Logic) ===
        
        # Case A: Normal(정상) 우선권 - Normal이 가장 높으면 무조건 OK
        # 확률값 상관없이 Normal이 Damaged나 Pollution보다 높으면 OK
        if p_normal > p_damaged and p_normal > p_pollution:
            # 정상일 때 오염 카운터 리셋
            self._pollution_trigger_count = 0
            result = "OK"
            errors = {"crack": False, "scratch": False, "color": False, "defect": False}
            defects = []
            status_text = f"OK (Normal: {p_normal:.1%})"
        
        # Case B: Damaged(크랙/구멍) 체크 - 50% 이상이면 즉시 NG (치명적, 민감하게)
        # 크랙은 치명적이므로 즉시 표시 (오염과 달리 깜빡임 방지 없음)
        elif p_damaged >= 0.50:
            # 크랙 검출 시 오염 카운터 리셋 (크랙이 우선)
            self._pollution_trigger_count = 0
            result = "NG"
            defect_type = 'crack'
            errors = {"crack": True, "scratch": False, "color": False, "defect": False}
            status_text = f"NG (Crack: {p_damaged:.1%})"
            defects = [{
                'type': defect_type,
                'bbox': (x, y, w, h),
                'area': w * h,
                'prob': p_damaged
            }]
        
        # Case C: Pollution(오염) 체크 - 95% 이상이고 5프레임 연속일 때만 NG
        # 배경 노이즈나 글자는 보통 80%대이므로, 95% 이상일 때만 진짜 오염으로 판정
        # 깜빡임 방지: 5프레임 연속 검출될 때만 NG로 판정
        elif p_pollution >= 0.95:
            # 오염 연속 검출 카운터 증가
            self._pollution_trigger_count += 1
            
            # 5프레임 연속 검출되었을 때만 NG로 판정
            if self._pollution_trigger_count >= self._pollution_trigger_threshold:
                result = "NG"
                defect_type = 'pollution'
                errors = {"crack": False, "scratch": False, "color": True, "defect": False}
                status_text = f"NG (Pollution: {p_pollution:.1%}, {self._pollution_trigger_count}연속)"
                defects = [{
                    'type': defect_type,
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'prob': p_pollution
                }]
            else:
                # 아직 연속 횟수가 부족하면 OK로 처리 (깜빡임 방지)
                result = "OK"
                errors = {"crack": False, "scratch": False, "color": False, "defect": False}
                defects = []
                status_text = f"OK (Pollution 감시 중: {p_pollution:.1%}, {self._pollution_trigger_count}/{self._pollution_trigger_threshold}연속)"
        else:
            # Pollution이 95% 미만이면 카운터 리셋하고 OK 처리
            # (Normal도 가장 높지 않고, Damaged도 50% 미만, Pollution도 95% 미만인 경우)
            self._pollution_trigger_count = 0
            result = "OK"
            errors = {"crack": False, "scratch": False, "color": False, "defect": False}
            defects = []
            if p_pollution > p_damaged:
                status_text = f"OK (Ignored Pollution/Text: {p_pollution:.1%})"
            elif p_damaged > p_pollution:
                status_text = f"OK (Weak Damage: {p_damaged:.1%})"
            else:
                status_text = f"OK (Normal Dominant: {p_normal:.1%})"
        
        # 가이드 박스는 항상 파란색으로 고정
        guide_color = (255, 0, 0)  # 파란색 (BGR)
        line_thickness = 3
        
        # 라벨 색상은 결과에 따라 변경 (가이드 박스는 파란색 고정)
        if result == "OK":
            label_color = (0, 255, 0)  # 초록색
        elif result == "NG":
            label_color = (0, 0, 255)  # 빨간색
        else:
            label_color = (200, 200, 200)  # 회색
        
        # status_text를 label 변수에 할당 (기존 코드 호환성)
        label = status_text
        
        # 가이드 박스 그리기 (항상 표시) - 좌표 범위 체크 추가
        x1 = max(0, min(int(x), display_frame.shape[1] - 1))
        y1 = max(0, min(int(y), display_frame.shape[0] - 1))
        x2 = max(0, min(int(x + w), display_frame.shape[1]))
        y2 = max(0, min(int(y + h), display_frame.shape[0]))
        
        # 유효한 좌표인지 확인
        if x2 > x1 and y2 > y1:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), guide_color, line_thickness)
            cv2.putText(display_frame, label, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        
        # 화면 좌측 상단에 확률 텍스트 및 HSV 디버깅 정보 표시
        debug_y = 30
        
        # HSV 디버깅 정보 표시 (밝기와 채도)
        hsv_s = getattr(self, '_last_hsv_s', 0.0)
        hsv_v = getattr(self, '_last_hsv_v', 0.0)
        hsv_debug_text = f"[Light: {hsv_v:.0f} | Color: {hsv_s:.0f}]"
        (hsv_text_width, hsv_text_height), baseline = cv2.getTextSize(hsv_debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display_frame, (10, debug_y - hsv_text_height - 3), (10 + hsv_text_width + 6, debug_y + 3), (0, 0, 0), -1)
        cv2.putText(display_frame, hsv_debug_text, (13, debug_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)  # 회색 텍스트
        
        # 확률 텍스트 표시 (HSV 정보 아래)
        prob_y = debug_y + hsv_text_height + 8
        
        # all_probs에서 각 클래스별 확률 추출
        if len(all_probs) >= 3:
            # 3개 클래스 모델: Normal(0), Damaged(1), Pollution(2)
            prob_normal = all_probs[0] if len(all_probs) > 0 else 0.0
            prob_damaged = all_probs[1] if len(all_probs) > 1 else 0.0
            prob_pollution = all_probs[2] if len(all_probs) > 2 else 0.0
            debug_text = f"Normal: {prob_normal:.1%} | Crack: {prob_damaged:.1%} | Pollution: {prob_pollution:.1%}"
        elif len(all_probs) == 2:
            # 2개 클래스 모델: normal, defect
            prob_normal = all_probs[0] if len(all_probs) > 0 else 0.0
            prob_defect = all_probs[1] if len(all_probs) > 1 else 0.0
            debug_text = f"Normal: {prob_normal:.1%} | Defect: {prob_defect:.1%}"
        else:
            # 기본값 (all_probs가 없는 경우)
            debug_text = f"Normal: {p_good:.1%} | Defect: {p_bad:.1%}"
        
        # 배경 박스 그리기 (텍스트 가독성 향상)
        (text_width, text_height), baseline = cv2.getTextSize(debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (10, prob_y - text_height - 5), (10 + text_width + 10, prob_y + 5), (0, 0, 0), -1)
        cv2.putText(display_frame, debug_text, (15, prob_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 노란색 텍스트
        
        # 진단 모드: 디버깅 출력 제거 (화면에 이미 표시됨)
        # 결과 반환 (result는 best_class 그대로)
        return (result, errors, battery_rect, defects)
    
    def _detect_battery_candidates(self, frame):
        """
        배터리 후보 영역 감지 (고정 ROI 모드)
        
        이 메서드는 화면 중앙에 고정된 ROI 영역을 반환합니다.
        배터리 탐지 불안정 문제를 해결하기 위해 고정 영역을 사용합니다.
        
        Args:
            frame: numpy.ndarray (원본 카메라 프레임, BGR 형식)
        
        Returns:
            list: 배터리 후보 리스트 [(x, y, w, h, score), ...]
                  고정 모드에서는 항상 1개 요소를 포함
        """
        frame_h, frame_w = frame.shape[:2]
        
        # 고정 ROI 크기: 가로 320, 세로 480 (화면 높이 꽉 차게)
        roi_w = 320
        roi_h = 480
        
        # 화면 중앙 좌표 계산
        x = (frame_w - roi_w) // 2  # (640 - 320) // 2 = 160
        y = 0  # 화면 상단부터 시작
        
        # 경계 체크 (프레임 크기가 작은 경우 대비)
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        roi_w = min(roi_w, frame_w - x)
        roi_h = min(roi_h, frame_h - y)
        
        # 점수는 무조건 1.0 (고정 영역이므로)
        return [(x, y, roi_w, roi_h, 1.0)]
    
    def stop(self):
        """
        스레드 중지
        
        이 메서드는 검사 스레드를 안전하게 중지합니다.
        카메라 연결을 해제하고 실행 플래그를 False로 설정합니다.
        """
        self._run_flag = False
        self._disconnect_camera()


class BatteryInspector(QWidget):
    """
    배터리 검사 시스템 메인 클래스 (Inspection App)
    Main class for Battery Inspection System (inspection_app)
    
    PyQt6를 사용하여 사용자 인터페이스를 제공합니다.
    주요 기능:
    - 실시간 카메라 영상 표시
    - 검사 결과 표시 (OK/NG/NO_BATTERY)
    - 불량 카운트 통계 표시 (TOTAL, OK, CRACK, POLLUTION)
    - AI 전처리 이미지 및 불량 영역 시각화
    - 카메라 설정 조절 (Exposure)
    """
    def __init__(self):
        super().__init__()
        self.thread = InspectionThread()
        self.stats = {"TOTAL": 0, "OK": 0, "CRACK": 0, "POLLUTION": 0}
        self.previous_result = None
        self.result_persist_count = 0
        self.result_persist_threshold = 10  # 5 -> 10: 더 안정적으로 (10프레임 지속)
        self.last_counted_battery_id = None
        self._last_counted_battery_id = None
        # 시간 기반 카운트 로직
        self._result_start_time = None  # 현재 결과가 시작된 시간
        self._result_duration_threshold = 2.5  # 2.5초 (2~3초 중간값)
        self._last_counted_result = None  # 마지막으로 카운트된 결과
        self._last_counted_time = None  # 마지막 카운트 시간
        self._last_battery_rect = None  # IoU 계산을 위한 이전 배터리 위치
        self._result_history = []  # 판정 히스토리 (안정화용)
        self._history_size = 15  # 최근 15개 프레임의 판정 저장 (7 -> 15)
        self._stable_result = None  # 안정화된 최종 판정
        self._stable_count = 0  # 안정화된 결과가 지속된 프레임 수
        self._confidence_history = []  # confidence 히스토리 (안정화용)
        self._stable_confidence = 0.0  # 안정화된 confidence 값
        self._current_preprocessed_img = None  # 현재 전처리된 이미지 (하자 영역 시각화용)
        self._last_defects = []  # 마지막 defects 정보 (카운트에 사용)
        
        # 카메라 설정 Debouncing을 위한 타이머
        self.exposure_timer = QTimer()
        self.exposure_timer.setSingleShot(True)  # 한 번만 실행
        self.exposure_timer.timeout.connect(self._apply_exposure_setting)
        self._pending_exposure = None  # 대기 중인 Exposure 값
        self._exposure_auto_off_done = False  # ExposureAuto Off 설정 완료 플래그
        
        # 해상도 설정 Debouncing을 위한 타이머
        self.resolution_timer = QTimer()
        self.resolution_timer.setSingleShot(True)
        self.resolution_timer.timeout.connect(self._apply_resolution_setting)
        self._pending_width = None
        self._pending_height = None
        
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("AI Inspection System - Basler Cam Mode")
        self.setGeometry(100, 100, 1280, 720)  # 1280x720 비율로 설정
        
        # 전체 위젯 스타일 설정 (현대적인 다크 테마)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #1e1e2e, stop:1 #121212);
                color: #e0e0e0;
                font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            }
        """)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 왼쪽: 카메라 뷰, 검사 카운트, 검사 결과
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        
        # 카메라 뷰 (카드 스타일)
        camera_label = QLabel("NO_CAMERA")
        camera_label.setMinimumSize(640, 480)
        camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0a0a0a, stop:1 #000000);
                border: 2px solid #3a3a4a;
                border-radius: 16px;
                color: #ffb84d;
                font-size: 28px;
                font-weight: 600;
                padding: 20px;
            }
        """)
        self.camera_label = camera_label
        left_layout.addWidget(camera_label)
        
        # 검사 카운트 (카드 스타일)
        stats_title = QLabel("검사 카운트")
        stats_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        stats_title.setStyleSheet("""
            color: #e0e0e0;
            padding: 4px 0px;
            font-weight: 600;
        """)
        stats_title.setContentsMargins(0, 8, 0, 4)
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(8)  # 간격 조정: 카메라 뷰 너비(640px)에 맞춤
        
        # 카메라 뷰 너비(640px)에 맞춰 4개 카드를 동일한 비율로 배치
        # 계산: (640 - 3*8) / 4 = 616 / 4 = 154px
        card_width = 154
        card_height = 70
        
        total_label = QLabel("TOTAL\n0")
        ok_label = QLabel("OK\n0")
        crack_label = QLabel("Damage\n0")
        pollution_label = QLabel("Pollution\n0")
        
        # TOTAL 카드 스타일 (동일한 크기)
        total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        total_label.setMinimumSize(card_width, card_height)
        total_label.setMaximumSize(card_width, card_height)
        total_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #2d2d3d, stop:1 #1f1f2f);
                border: 2px solid #4a4a5a;
                border-radius: 12px;
                color: #e0e0e0;
                font-size: 15px;
                font-weight: 600;
                padding: 8px;
            }
        """)
        
        # OK 카드 스타일 (동일한 크기)
        ok_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ok_label.setMinimumSize(card_width, card_height)
        ok_label.setMaximumSize(card_width, card_height)
        ok_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #2d7a2d, stop:1 #1f5a1f);
                border: 2px solid #4a9a4a;
                border-radius: 12px;
                color: #ffffff;
                font-size: 15px;
                font-weight: 700;
                padding: 8px;
            }
        """)
        
        # 크랙 카드 스타일 (동일한 크기)
        crack_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crack_label.setMinimumSize(card_width, card_height)
        crack_label.setMaximumSize(card_width, card_height)
        crack_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #8a3d2d, stop:1 #6a2d1f);
                border: 2px solid #aa5d4a;
                border-radius: 12px;
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                padding: 8px;
            }
        """)
        
        # 오염 카드 스타일 (동일한 크기)
        pollution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pollution_label.setMinimumSize(card_width, card_height)
        pollution_label.setMaximumSize(card_width, card_height)
        pollution_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #7a4d2d, stop:1 #5a3d1f);
                border: 2px solid #9a6d4a;
                border-radius: 12px;
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                padding: 8px;
            }
        """)
        
        self.lbl_total = total_label
        self.lbl_ok = ok_label
        self.lbl_crack = crack_label
        self.lbl_pollution = pollution_label
        
        stats_layout.addWidget(total_label)
        stats_layout.addWidget(ok_label)
        stats_layout.addWidget(crack_label)
        stats_layout.addWidget(pollution_label)
        
        left_layout.addWidget(stats_title)
        left_layout.addLayout(stats_layout)
        
        # 검사 결과 (카드 스타일)
        res_display = QLabel("NO_BATTERY")
        res_display.setMinimumHeight(60)
        res_display.setMaximumHeight(70)
        res_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_display.setContentsMargins(0, 4, 0, 4)
        res_display.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3a3a2a, stop:1 #2a2a1a);
                border: 2px solid #6a6a4a;
                border-radius: 16px;
                color: #ffb84d;
                font-size: 26px;
                font-weight: 700;
                padding: 8px 16px;
            }
        """)
        self.res_display = res_display
        left_layout.addWidget(res_display)
        
        # 오른쪽: AI 이미지 및 하자 정보
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        # AI가 보고 있는 전처리된 이미지 영역
        preprocessed_label = QLabel("AI가 보고 있는 이미지 (320x320)")
        preprocessed_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        preprocessed_label.setStyleSheet("""
            color: #e0e0e0; 
            padding: 4px 0px;
            font-weight: 600;
        """)
        preprocessed_label.setContentsMargins(0, 0, 0, 4)
        preprocessed_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # 가로 중앙 정렬
        preprocessed_area = QLabel("전처리 이미지\n(Bilateral + Letterbox)")
        preprocessed_area.setMinimumSize(320, 320)
        preprocessed_area.setMaximumSize(320, 320)
        preprocessed_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preprocessed_area.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0a0a1a, stop:1 #000000);
                border: 2px solid #00d4ff;
                border-radius: 16px;
                color: #00d4ff;
                font-size: 14px;
                font-weight: 600;
                padding: 10px;
            }
        """)
        self.preprocessed_area = preprocessed_area
        right_layout.addWidget(preprocessed_label)
        right_layout.addWidget(preprocessed_area)
        
        # 하자 영역 시각화 화면
        defect_visualization_label = QLabel("Defect Visualization")
        defect_visualization_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        defect_visualization_label.setStyleSheet("""
            color: #e0e0e0; 
            padding: 4px 0px;
            font-weight: 600;
        """)
        defect_visualization_label.setContentsMargins(0, 8, 0, 4)
        defect_visualization_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # 가로 중앙 정렬
        defect_visualization_area = QLabel("Defect area will be displayed here")
        defect_visualization_area.setMinimumSize(320, 150)
        defect_visualization_area.setMaximumSize(320, 200)
        defect_visualization_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        defect_visualization_area.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #2a2a3a, stop:1 #1a1a2a);
                border: 2px solid #5a5a6a;
                border-radius: 16px;
                color: #b0b0b0;
                font-size: 13px;
                padding: 10px;
            }
        """)
        self.defect_visualization_area = defect_visualization_area
        right_layout.addWidget(defect_visualization_label)
        right_layout.addWidget(defect_visualization_area)
        
        # 하자 판정 결과
        defect_label = QLabel("Detected Defects")
        defect_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        defect_label.setStyleSheet("""
            color: #e0e0e0; 
            padding: 4px 0px;
            font-weight: 600;
        """)
        defect_label.setContentsMargins(0, 8, 0, 4)
        defect_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # 가로 중앙 정렬
        defect_area = QLabel("No Defect")
        # defect_area의 너비를 기준으로 다른 요소들을 정렬
        defect_area.setMinimumSize(320, 55)  # 너비를 320px로 고정 (AI 이미지와 동일)
        defect_area.setMaximumSize(320, 65)  # 너비를 320px로 고정
        defect_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        defect_area.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f5f5f5, stop:1 #e5e5e5);
                border: 2px solid #d0d0d0;
                border-radius: 12px;
                color: #555555;
                font-size: 15px;
                font-weight: 600;
                padding: 8px;
            }
        """)
        self.defect_area = defect_area
        right_layout.addWidget(defect_label)
        right_layout.addWidget(defect_area)
        
        # 오른쪽 레이아웃의 모든 요소를 가로 중앙 정렬
        right_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        # 카메라 설정 패널
        camera_settings_group = QGroupBox("카메라 설정")
        camera_settings_group.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        camera_settings_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #5a5a6a;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e0e0e0;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #2a2a3a, stop:1 #1a1a2a);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        camera_settings_layout = QVBoxLayout()
        camera_settings_layout.setSpacing(8)
        
        # 노출 시간 (Exposure) 설정
        exposure_layout = QHBoxLayout()
        exposure_label = QLabel("밝기 (Exposure):")
        exposure_label.setStyleSheet("color: #e0e0e0; font-size: 12px; min-width: 120px;")
        exposure_slider = QSlider(Qt.Orientation.Horizontal)
        exposure_slider.setMinimum(1000)  # 1ms
        exposure_slider.setMaximum(100000)  # 100ms
        exposure_slider.setValue(10000)  # 10ms 기본값
        exposure_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        exposure_slider.setTickInterval(10000)
        exposure_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #5a5a6a;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2a, stop:1 #3a3a4a);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a9eff, stop:1 #2a7eff);
                border: 1px solid #3a8eff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5aaeff, stop:1 #3a8eff);
            }
        """)
        exposure_value_label = QLabel("10.0 ms")
        exposure_value_label.setStyleSheet("color: #00d4ff; font-size: 11px; font-weight: 600; min-width: 60px;")
        # 슬라이더 값 변경 시 즉시 UI 업데이트
        exposure_slider.valueChanged.connect(lambda v: exposure_value_label.setText(f"{v/1000:.1f} ms"))
        # Debouncing: 슬라이더 값 변경 시 타이머 시작 (100ms 후 적용)
        exposure_slider.valueChanged.connect(lambda v: self._schedule_exposure_update(v))
        exposure_layout.addWidget(exposure_label)
        exposure_layout.addWidget(exposure_slider)
        exposure_layout.addWidget(exposure_value_label)
        camera_settings_layout.addLayout(exposure_layout)
        
        # 해상도 (Width, Height) 설정
        resolution_layout = QVBoxLayout()
        resolution_layout.setSpacing(4)
        
        # Width 설정
        width_layout = QHBoxLayout()
        width_label = QLabel("너비 (Width):")
        width_label.setStyleSheet("color: #e0e0e0; font-size: 12px; min-width: 120px;")
        width_spinbox = QSpinBox()
        width_spinbox.setMinimum(320)
        width_spinbox.setMaximum(8192)  # 더 큰 해상도 지원 (일부 카메라는 8K까지 지원)
        width_spinbox.setValue(4096)  # 기본값: 4K
        width_spinbox.setSingleStep(16)  # 카메라가 일반적으로 16픽셀 단위로 조정
        width_spinbox.setSuffix(" px")
        width_spinbox.setStyleSheet("""
            QSpinBox {
                background: #2a2a3a;
                border: 1px solid #4a4a5a;
                border-radius: 6px;
                color: #e0e0e0;
                font-size: 11px;
                padding: 4px;
                min-width: 80px;
            }
            QSpinBox:hover {
                border: 1px solid #5a5a6a;
            }
            QSpinBox:focus {
                border: 1px solid #4a9eff;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #3a3a4a;
                border: 1px solid #5a5a6a;
                border-radius: 3px;
                width: 20px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #4a4a5a;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background: #5a5a6a;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 8px;
                height: 8px;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #e0e0e0;
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #e0e0e0;
            }
        """)
        width_spinbox.valueChanged.connect(lambda v: self._schedule_resolution_update(v, None))
        width_layout.addWidget(width_label)
        width_layout.addWidget(width_spinbox)
        resolution_layout.addLayout(width_layout)
        
        # Height 설정
        height_layout = QHBoxLayout()
        height_label = QLabel("높이 (Height):")
        height_label.setStyleSheet("color: #e0e0e0; font-size: 12px; min-width: 120px;")
        height_spinbox = QSpinBox()
        height_spinbox.setMinimum(240)
        height_spinbox.setMaximum(6144)  # 더 큰 해상도 지원 (일부 카메라는 6K까지 지원)
        height_spinbox.setValue(2160)  # 기본값: 4K
        height_spinbox.setSingleStep(16)
        height_spinbox.setSuffix(" px")
        height_spinbox.setStyleSheet("""
            QSpinBox {
                background: #2a2a3a;
                border: 1px solid #4a4a5a;
                border-radius: 6px;
                color: #e0e0e0;
                font-size: 11px;
                padding: 4px;
                min-width: 80px;
            }
            QSpinBox:hover {
                border: 1px solid #5a5a6a;
            }
            QSpinBox:focus {
                border: 1px solid #4a9eff;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #3a3a4a;
                border: 1px solid #5a5a6a;
                border-radius: 3px;
                width: 20px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #4a4a5a;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background: #5a5a6a;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 8px;
                height: 8px;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #e0e0e0;
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #e0e0e0;
            }
        """)
        height_spinbox.valueChanged.connect(lambda v: self._schedule_resolution_update(None, v))
        height_layout.addWidget(height_label)
        height_layout.addWidget(height_spinbox)
        resolution_layout.addLayout(height_layout)
        
        camera_settings_layout.addLayout(resolution_layout)
        
        camera_settings_group.setLayout(camera_settings_layout)
        right_layout.addWidget(camera_settings_group)
        
        # 카메라 설정 UI 요소 저장
        self.exposure_slider = exposure_slider
        self.width_spinbox = width_spinbox
        self.height_spinbox = height_spinbox
        
        # 버튼 (현대적인 스타일)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        btn_start = QPushButton("Start Camera")
        btn_stop = QPushButton("Stop")
        btn_stop.setEnabled(False)
        
        # Start 버튼 스타일
        btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4a9eff, stop:1 #2a7eff);
                border: none;
                border-radius: 12px;
                color: #ffffff;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #5aaeff, stop:1 #3a8eff);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3a8eff, stop:1 #1a6eff);
            }
        """)
        
        # Stop 버튼 스타일
        btn_stop.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #6a6a6a, stop:1 #4a4a4a);
                border: none;
                border-radius: 12px;
                color: #b0b0b0;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 20px;
            }
            QPushButton:enabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff5a5a, stop:1 #ff3a3a);
                color: #ffffff;
            }
            QPushButton:enabled:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6a6a, stop:1 #ff4a4a);
            }
            QPushButton:enabled:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff4a4a, stop:1 #ff2a2a);
            }
        """)
        
        btn_start.clicked.connect(self.start_sys)
        btn_stop.clicked.connect(self.stop_sys)
        
        self.btn_start = btn_start
        self.btn_stop = btn_stop
        
        button_layout.addWidget(btn_start)
        button_layout.addWidget(btn_stop)
        left_layout.addLayout(button_layout)
        
        # 레이아웃 결합
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)  # 레이아웃 직접 추가
        self.setLayout(main_layout)
    
    def connect_signals(self):
        """신호 연결"""
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.result_signal.connect(self.update_result)
        self.thread.defect_detail_signal.connect(self.update_defects)
        self.thread.camera_connected_signal.connect(self.on_camera_status_changed)
        self.thread.preprocessed_image_signal.connect(self.update_preprocessed_image)
    
    def start_sys(self):
        """시스템 시작"""
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
    
    def stop_sys(self):
        """시스템 중지"""
        self.thread.stop()
        self.thread.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_label.setText("NO_CAMERA")
        self.camera_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0a0a0a, stop:1 #000000);
                border: 2px solid #3a3a4a;
                border-radius: 16px;
                color: #ffb84d;
                font-size: 28px;
                font-weight: 600;
                padding: 20px;
            }
        """)
    
    def update_image(self, img):
        """이미지 업데이트"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, w*c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        self.camera_label.setText("")
    
    def update_result(self, res, errors, battery_rect):
        """검사 결과 업데이트"""
        if res is None or res == "":
            return
        
        # UI 표시 (현대적인 그라데이션 스타일)
        if res == "OK":
            self.res_display.setText("OK")
            self.res_display.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #2d7a2d, stop:1 #1f5a1f);
                    border: 2px solid #4a9a4a;
                    border-radius: 16px;
                    color: #ffffff;
                    font-size: 28px;
                    font-weight: 700;
                    padding: 8px 16px;
                }
            """)
        elif res == "NG":
            self.res_display.setText("NG")
            self.res_display.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #7a2d2d, stop:1 #5a1f1f);
                    border: 2px solid #9a4a4a;
                    border-radius: 16px;
                    color: #ffffff;
                    font-size: 28px;
                    font-weight: 700;
                    padding: 8px 16px;
                }
            """)
        elif res == "NO_BATTERY":
            # NO_BATTERY일 때 회색 그라데이션
            self.res_display.setText("NO_BATTERY")
            self.res_display.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #5a5a5a, stop:1 #3a3a3a);
                    border: 2px solid #7a7a7a;
                    border-radius: 16px;
                    color: #ffffff;
                    font-size: 26px;
                    font-weight: 700;
                    padding: 8px 16px;
                }
            """)
        else:
            self.res_display.setText(str(res))
            self.res_display.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #3a3a2a, stop:1 #2a2a1a);
                    border: 2px solid #6a6a4a;
                    border-radius: 16px;
                    color: #ffb84d;
                    font-size: 26px;
                    font-weight: 700;
                    padding: 8px 16px;
                }
            """)
        
        # 판정 히스토리에 추가 (안정화용) - NO_BATTERY도 포함
        if res in ["OK", "NG", "NO_BATTERY"]:
            self._result_history.append(res)
            if len(self._result_history) > self._history_size:
                self._result_history.pop(0)
        
        # 카운트 (OK/NG만, ERROR는 제외)
        if res in ["OK", "NG"]:
            
            # 강력한 히스테리시스 적용: OK/NO_BATTERY 상태 유지 강화
            from collections import Counter
            if len(self._result_history) >= 10:
                # 최근 10개 프레임 분석
                recent_10 = self._result_history[-10:]
                result_counts = Counter(recent_10)
                ok_count = result_counts.get("OK", 0)
                ng_count = result_counts.get("NG", 0)
                no_battery_count = result_counts.get("NO_BATTERY", 0)
                
                # 현재 안정화된 결과가 있으면
                if self._stable_result is not None:
                    # OK 상태 유지: 최근 10개 중 8개 이상이 OK면 OK 유지 (6 -> 8로 강화)
                    if self._stable_result == "OK":
                        if ok_count >= 8:  # 6 -> 8로 강화 (더 안정적으로 유지)
                            stable_result = "OK"
                            self._stable_count += 1
                        elif ng_count >= 8:  # 최근 10개 중 8개 이상이 NG면 NG로 변경
                            stable_result = "NG"
                            self._stable_count = 0
                        elif no_battery_count >= 8:  # 최근 10개 중 8개 이상이 NO_BATTERY면 NO_BATTERY로 변경 (7 -> 8로 강화)
                            stable_result = "NO_BATTERY"
                            self._stable_count = 0
                        else:
                            # 애매한 경우: 현재 상태 유지 (번쩍거림 방지)
                            stable_result = "OK"
                            self._stable_count += 1
                    # NO_BATTERY 상태 유지: 최근 10개 중 7개 이상이 NO_BATTERY면 NO_BATTERY 유지
                    elif self._stable_result == "NO_BATTERY":
                        if no_battery_count >= 7:
                            stable_result = "NO_BATTERY"
                            self._stable_count += 1
                        elif ok_count >= 6:  # 최근 10개 중 6개 이상이 OK면 OK로 변경
                            stable_result = "OK"
                            self._stable_count = 0
                        elif ng_count >= 8:  # 최근 10개 중 8개 이상이 NG면 NG로 변경
                            stable_result = "NG"
                            self._stable_count = 0
                        else:
                            # 애매한 경우: 현재 상태 유지
                            stable_result = "NO_BATTERY"
                            self._stable_count += 1
                    # NG 상태 유지: 최근 10개 중 8개 이상이 NG면 NG 유지 (6 -> 8로 강화)
                    elif self._stable_result == "NG":
                        if ng_count >= 8:  # 6 -> 8로 강화 (더 안정적으로 유지)
                            stable_result = "NG"
                            self._stable_count += 1
                        elif ok_count >= 8:  # 최근 10개 중 8개 이상이 OK면 OK로 변경 (7 -> 8로 강화)
                            stable_result = "OK"
                            self._stable_count = 0
                        elif no_battery_count >= 8:  # 최근 10개 중 8개 이상이 NO_BATTERY면 NO_BATTERY로 변경 (7 -> 8로 강화)
                            stable_result = "NO_BATTERY"
                            self._stable_count = 0
                        else:
                            # 애매한 경우: 현재 상태 유지 (번쩍거림 방지)
                            stable_result = "NG"
                            self._stable_count += 1
                    else:
                        stable_result = self._stable_result
                else:
                    # 처음 시작할 때는 최근 10개 중 다수결 원칙
                    if ok_count >= 6:  # 7개 -> 6개로 완화
                        stable_result = "OK"
                        self._stable_count = 1
                    elif ng_count >= 8:
                        stable_result = "NG"
                        self._stable_count = 1
                    elif no_battery_count >= 7:
                        stable_result = "NO_BATTERY"
                        self._stable_count = 1
                    else:
                        stable_result = res
                        self._stable_count = 1
            elif len(self._result_history) >= 5:
                # 히스토리가 5개 이상이면 최근 5개로 판단
                recent_5 = self._result_history[-5:]
                result_counts = Counter(recent_5)
                ok_count = result_counts.get("OK", 0)
                ng_count = result_counts.get("NG", 0)
                
                if self._stable_result is not None:
                    if self._stable_result == "OK":
                        if ok_count >= 4:  # 최근 5개 중 4개 이상이 OK면 OK 유지
                            stable_result = "OK"
                            self._stable_count += 1
                        elif ng_count >= 4:  # 최근 5개 중 4개 이상이 NG면 NG로 변경
                            stable_result = "NG"
                            self._stable_count = 0
                        else:
                            stable_result = "OK"
                            self._stable_count += 1
                    else:
                        if ng_count >= 3:
                            stable_result = "NG"
                            self._stable_count += 1
                        elif ok_count >= 4:
                            stable_result = "OK"
                            self._stable_count = 0
                        else:
                            stable_result = "NG"
                            self._stable_count += 1
                else:
                    if ok_count >= 4:
                        stable_result = "OK"
                        self._stable_count = 1
                    elif ng_count >= 4:
                        stable_result = "NG"
                        self._stable_count = 1
                    else:
                        stable_result = res
                        self._stable_count = 1
            else:
                # 히스토리가 충분하지 않으면 현재 결과 사용
                stable_result = res
                self._stable_count = 1
            
            # 안정화된 결과 저장
            previous_stable_result = self._stable_result
            self._stable_result = stable_result
            
            # 시간 기반 지속 시간 확인
            current_time = time.time()
            # stable_result가 변경되었을 때만 타이머 리셋
            if previous_stable_result != stable_result:
                self._result_start_time = current_time
                print(f"[DEBUG] 결과 변경: {previous_stable_result} -> {stable_result}, 타이머 리셋")
            elif self._result_start_time is None:
                self._result_start_time = current_time
            
            result_duration = current_time - self._result_start_time
            
            # 같은 결과가 지속되는지 확인 (프레임 기반)
            if self.previous_result == stable_result:
                self.result_persist_count += 1
            else:
                self.previous_result = stable_result
                self.result_persist_count = 1
            
            # 안정화된 결과 사용
            res = stable_result
            
            # 배터리가 인식되지 않았을 때 히스토리 초기화 방지
            # (배터리가 계속 인식되고 있으면 히스토리 유지)
            if res in ["OK", "NG"]:
                # 배터리 인식 성공 - 히스토리 유지
                pass
            else:
                # 배터리 인식 실패 - 히스토리 일부 초기화 (완전 초기화는 하지 않음)
                if len(self._result_history) > 5:
                    # 최근 5개만 유지
                    self._result_history = self._result_history[-5:]
            
            # 배터리 ID 생성 및 추적 (같은 배터리 중복 카운트 방지)
            if battery_rect:
                x, y, w, h = battery_rect
                # 중심점과 크기를 기반으로 ID 생성 (더 안정적)
                center_x = x + w // 2
                center_y = y + h // 2
                # 50픽셀 단위로 그리드화하여 같은 그리드면 같은 배터리로 인식 (100 -> 50으로 세밀화)
                grid_x = center_x // 50
                grid_y = center_y // 50
                grid_w = w // 25  # 50 -> 25로 세밀화
                grid_h = h // 25
                battery_id = f"{grid_x}_{grid_y}_{grid_w}_{grid_h}"
                
                # IoU 기반으로 같은 배터리인지 확인 (추가 안정성)
                if hasattr(self, '_last_battery_rect') and self._last_battery_rect is not None:
                    last_x, last_y, last_w, last_h = self._last_battery_rect
                    # IoU 계산
                    inter_x = max(x, last_x)
                    inter_y = max(y, last_y)
                    inter_w = min(x + w, last_x + last_w) - inter_x
                    inter_h = min(y + h, last_y + last_h) - inter_y
                    if inter_w > 0 and inter_h > 0:
                        inter_area = inter_w * inter_h
                        union_area = w * h + last_w * last_h - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        # IoU가 0.3 이상이면 같은 배터리로 간주 (0.5 -> 0.3으로 완화하여 더 정확하게 추적)
                        if iou > 0.3:
                            battery_id = self._last_counted_battery_id
                
                self._last_battery_rect = battery_rect
            else:
                battery_id = None
                self._last_battery_rect = None
            
            # 카운트된 배터리 ID 히스토리 관리 (같은 배터리 중복 카운트 방지)
            if not hasattr(self, '_counted_battery_ids'):
                self._counted_battery_ids = {}  # {battery_id: count_time}
            
            # 오래된 카운트 기록 제거 (30초 이상 지난 기록은 삭제)
            expired_ids = [bid for bid, t in self._counted_battery_ids.items() if current_time - t > 30.0]
            for bid in expired_ids:
                del self._counted_battery_ids[bid]
            
            # 카운트 로직: 상태 표시(OK/NG)에 따라 단순하게 카운트
            # 결과가 변경될 때마다 카운트 (NO_BATTERY → OK/NG, OK → NG, NG → OK 등)
            # 같은 결과가 연속으로 나오면 한 번만 카운트 (중복 방지)
            should_count = False
            if res in ["OK", "NG"]:
                # 이전 결과 확인 (안정화된 결과 사용)
                previous_stable_result = getattr(self, '_last_counted_result', None)
                
                # 결과가 변경되었거나, 처음 카운트하는 경우
                if previous_stable_result != res:
                    should_count = True
                    # 결과가 변경되었으므로 이전 결과 기록 삭제 (다시 돌아오면 카운트 가능)
                    if previous_stable_result is not None and battery_id is not None:
                        previous_key = f"{battery_id}_{previous_stable_result}"
                        if previous_key in self._counted_battery_ids:
                            del self._counted_battery_ids[previous_key]
                elif battery_id is not None:
                    # 같은 결과가 연속으로 나오는 경우, 아직 카운트되지 않았으면 카운트
                    current_key = f"{battery_id}_{res}"
                    if current_key not in self._counted_battery_ids:
                        should_count = True
            
            # 임계값 도달 및 새로운 배터리인 경우만 카운트 (기존 로직은 주석 처리, 시간 기반 로직 사용)
            if should_count:
                
                self.stats["TOTAL"] += 1
                if res == "OK":
                    self.stats["OK"] += 1
                elif res == "NG":
                    # defects 정보를 확인하여 크랙 또는 오염으로 분류
                    # update_result에서 받은 defects 정보 사용
                    defect_type = None
                    if hasattr(self, '_last_defects') and len(self._last_defects) > 0:
                        # 첫 번째 defect의 type 사용
                        defect_type = self._last_defects[0].get('type', 'defect')
                    
                    # defect_type에 따라 카운트
                    if defect_type in ['crack', 'damaged']:
                        self.stats["CRACK"] += 1
                    elif defect_type in ['pollution', 'color']:
                        self.stats["POLLUTION"] += 1
                    else:
                        # 타입을 알 수 없으면 크랙으로 카운트 (기본값)
                        self.stats["CRACK"] += 1
                
                # 카운트된 결과 기록 (시간과 함께 저장)
                if battery_id is not None:
                    count_key = f"{battery_id}_{res}"
                    self._counted_battery_ids[count_key] = current_time
                    self._last_counted_battery_id = battery_id
                    self.last_counted_battery_id = battery_id
                self._last_counted_result = res
                self._last_counted_time = current_time
                
                self.lbl_total.setText(f"TOTAL\n{self.stats['TOTAL']}")
                self.lbl_ok.setText(f"OK\n{self.stats['OK']}")
                self.lbl_crack.setText(f"Damage\n{self.stats['CRACK']}")
                self.lbl_pollution.setText(f"Pollution\n{self.stats['POLLUTION']}")
                
                # 디버깅 출력
                print(f"[INFO] 검사 카운트 (2.5초 지속): {res} ({defect_type if res == 'NG' else 'OK'}) - 지속 시간: {result_duration:.2f}초 (TOTAL={self.stats['TOTAL']}, OK={self.stats['OK']}, CRACK={self.stats['CRACK']}, POLLUTION={self.stats['POLLUTION']}, ID={battery_id})")
        else:
            # NO_BATTERY, ERROR 등은 카운트하지 않음
            # NO_BATTERY 상태일 때 마지막 카운트된 배터리 ID를 초기화하여
            # 다음에 배터리를 잡았을 때 새로운 배터리로 인식되도록 함
            self.previous_result = None
            self.result_persist_count = 0
            self._last_battery_rect = None
            if res == "NO_BATTERY":
                # NO_BATTERY 상태에서 다음 배터리를 잡았을 때 카운트되도록 초기화
                self._last_counted_battery_id = None
                self._last_counted_result = None
    
    def update_defects(self, defects):
        """하자 정보 업데이트 (크랙, 스크래치, 오염 구분)"""
        # defects 정보 저장 (카운트에 사용)
        self._last_defects = defects
        
        if len(defects) > 0:
            # 하자 타입별 개수 집계
            defect_types = {}
            for defect in defects:
                defect_type = defect.get('type', 'defect')
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
            
            # 하자 타입별 텍스트 생성
            type_texts = []
            if defect_types.get('crack', 0) > 0:
                type_texts.append(f"Damage {defect_types['crack']}개")
            if defect_types.get('scratch', 0) > 0:
                type_texts.append(f"Scratch {defect_types['scratch']}개")
            if defect_types.get('color', 0) > 0:
                type_texts.append(f"Pollution {defect_types['color']}개")
            if defect_types.get('defect', 0) > 0:
                type_texts.append(f"Defect {defect_types['defect']}개")
            
            # 텍스트 조합
            if type_texts:
                defect_text = "\n".join(type_texts)
            else:
                defect_text = f"Defect {len(defects)} detected"
            
            self.defect_area.setText(defect_text)
            self.defect_area.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffe5e5, stop:1 #ffcccc);
                    border: 2px solid #ff6a6a;
                    border-radius: 12px;
                    color: #cc0000;
                    font-size: 14px;
                    font-weight: 700;
                    padding: 8px;
                }
            """)
            
            # 하자 영역 시각화 화면 업데이트
            self._update_defect_visualization(defects)
        else:
            self.defect_area.setText("No Defect")
            self.defect_area.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #f5f5f5, stop:1 #e5e5e5);
                    border: 2px solid #d0d0d0;
                    border-radius: 12px;
                    color: #555555;
                    font-size: 15px;
                    font-weight: 600;
                    padding: 8px;
                }
            """)
            
            # 하자 영역 시각화 화면 초기화
            self._last_defects = []  # 빈 리스트로 초기화
            self._update_defect_visualization([])
    
    def _update_defect_visualization(self, defects):
        """하자 영역 시각화 화면 업데이트 (개선된 버전)"""
        try:
            # 전처리된 이미지가 없으면 빈 화면 표시
            if not hasattr(self, '_current_preprocessed_img') or self._current_preprocessed_img is None:
                # 빈 검은 화면 표시
                empty_img = np.zeros((150, 320, 3), dtype=np.uint8)
                img_rgb = cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, 320, 150, 320*3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.defect_visualization_area.setPixmap(pixmap)
                return
            
            # 전처리된 이미지 복사 (원본 보존)
            vis_img = self._current_preprocessed_img.copy()
            
            # 하자 영역이 있으면 그리기
            if len(defects) > 0:
                # ROI 크기: 320x480 (원본 프레임 기준)
                # 전처리된 이미지: 320x320
                # 따라서 y축만 스케일링 필요 (480 -> 320)
                roi_h_original = 480
                roi_w_original = 320
                vis_h, vis_w = vis_img.shape[:2]  # 320x320
                
                scale_y = vis_h / roi_h_original  # 320 / 480 = 0.666...
                scale_x = vis_w / roi_w_original  # 320 / 320 = 1.0
                
                for defect in defects:
                    bbox = defect.get('bbox', None)
                    if bbox is None:
                        continue
                    
                    x, y, w, h = bbox
                    # ROI 좌표를 시각화 이미지 좌표로 변환
                    # 고정 ROI: x = (640-320)//2 = 160, y = 0
                    roi_x_offset = 160  # 원본 프레임에서 ROI 시작 x 좌표
                    roi_y_offset = 0    # 원본 프레임에서 ROI 시작 y 좌표
                    
                    # ROI 내부 좌표로 변환
                    x_in_roi = x - roi_x_offset
                    y_in_roi = y - roi_y_offset
                    
                    # 시각화 이미지 좌표로 스케일링
                    x_vis = int(x_in_roi * scale_x)
                    y_vis = int(y_in_roi * scale_y)
                    w_vis = int(w * scale_x)
                    h_vis = int(h * scale_y)
                    
                    # 좌표 범위 체크
                    if x_vis < 0 or y_vis < 0 or x_vis + w_vis > vis_w or y_vis + h_vis > vis_h:
                        continue
                    
                    # 하자 타입에 따라 색상 결정 (더 명확한 색상)
                    defect_type = defect.get('type', 'defect')
                    if defect_type in ['crack', 'damaged']:
                        color = (0, 0, 255)  # 빨간색 (BGR) - 크랙
                        label_text = "크랙"
                    elif defect_type == 'scratch':
                        color = (0, 165, 255)  # 주황색 (BGR) - 스크래치
                        label_text = "스크래치"
                    elif defect_type in ['pollution', 'color']:
                        color = (255, 0, 255)  # 자홍색 (BGR) - 오염
                        label_text = "오염"
                    else:
                        color = (0, 0, 255)  # 빨간색 (BGR) - 기타
                        label_text = "하자"
                    
                    # 하자 영역을 더 명확하게 표시
                    # 1. 반투명 채우기 (overlay)
                    overlay = vis_img.copy()
                    cv2.rectangle(overlay, (x_vis, y_vis), (x_vis + w_vis, y_vis + h_vis), color, -1)
                    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                    
                    # 2. 두꺼운 테두리
                    cv2.rectangle(vis_img, (x_vis, y_vis), (x_vis + w_vis, y_vis + h_vis), color, 3)
                    
                    # 3. 하자 타입 텍스트 표시 (더 크고 명확하게)
                    font_scale = 0.6
                    font_thickness = 2
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = x_vis
                    text_y = max(y_vis - 10, text_size[1] + 10)
                    
                    # 텍스트 배경 (가독성 향상)
                    cv2.rectangle(vis_img, 
                                (text_x - 2, text_y - text_size[1] - 5),
                                (text_x + text_size[0] + 2, text_y + 5),
                                (0, 0, 0), -1)
                    
                    # 텍스트 표시
                    cv2.putText(vis_img, label_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            
            # 이미지 크기 조정 (시각화 영역 크기에 맞춤, 비율 유지)
            # 320x320 이미지를 320x150 영역에 맞추되, 비율을 유지하기 위해 letterbox 방식 사용
            target_w, target_h = 320, 150
            vis_h, vis_w = vis_img.shape[:2]
            
            # 비율 계산
            scale = min(target_w / vis_w, target_h / vis_h)
            new_w = int(vis_w * scale)
            new_h = int(vis_h * scale)
            
            # 리사이즈
            vis_img_resized = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Letterbox: 중앙에 배치하고 나머지는 검은색으로 채우기
            vis_img_final = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            vis_img_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = vis_img_resized
            
            # BGR -> RGB 변환
            vis_img_rgb = cv2.cvtColor(vis_img_final, cv2.COLOR_BGR2RGB)
            
            # QImage 생성
            qimg = QImage(vis_img_rgb.data, target_w, target_h, target_w*3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # 시각화 영역에 표시
            self.defect_visualization_area.setPixmap(pixmap)
            
        except Exception as e:
            print(f"[WARNING] 하자 영역 시각화 업데이트 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def update_preprocessed_image(self, img):
        """AI가 보고 있는 전처리된 이미지 업데이트"""
        try:
            # None 체크 강화
            if img is None:
                print("[WARNING] 전처리된 이미지가 None입니다.")
                # None일 때 빈 이미지 표시
                if hasattr(self, 'preprocessed_label'):
                    self.preprocessed_label.clear()
                return
            
            # numpy 배열인지 확인
            if not isinstance(img, np.ndarray):
                print(f"[WARNING] 전처리된 이미지가 numpy 배열이 아닙니다: type={type(img)}")
                return
            
            # 이미지가 올바른 형식인지 확인
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"[WARNING] 전처리된 이미지 형식 오류: shape={img.shape}")
                return
            
            # 이미지 크기 확인 (320x320이어야 함)
            h, w = img.shape[:2]
            if h != 320 or w != 320:
                print(f"[WARNING] 전처리된 이미지 크기 오류: {w}x{h} (예상: 320x320)")
                # 크기가 다르면 리사이즈
                img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
            
            # 하자 영역 시각화를 위해 이미지 저장 (BGR 형식으로 저장)
            self._current_preprocessed_img = img.copy()
            
            # BGR -> RGB 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # QImage 생성 (데이터 복사 필요)
            img_bytes = img_rgb.tobytes()
            qimg = QImage(img_bytes, w, h, w*3, QImage.Format.Format_RGB888)
            
            if qimg.isNull():
                print("[WARNING] QImage 생성 실패")
                return
            
            # QPixmap으로 변환
            pixmap = QPixmap.fromImage(qimg)
            
            if pixmap.isNull():
                print("[WARNING] QPixmap 생성 실패")
                return
            
            # 320x320 크기로 표시 (이미 320x320이지만 크기 조정)
            scaled_pixmap = pixmap.scaled(
                self.preprocessed_area.width(),
                self.preprocessed_area.height(),
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.preprocessed_area.setPixmap(scaled_pixmap)
            self.preprocessed_area.setText("")
            
            # 디버깅 출력 (5초마다)
            if not hasattr(self, '_last_preprocessed_log_time') or time.time() - self._last_preprocessed_log_time > 5.0:
                print(f"[SUCCESS] AI 전처리 이미지 업데이트: {w}x{h}")
                self._last_preprocessed_log_time = time.time()
                
        except Exception as e:
            print(f"[WARNING] 전처리 이미지 업데이트 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _schedule_exposure_update(self, exposure):
        """Exposure 업데이트 스케줄링 (Debouncing)"""
        self._pending_exposure = exposure
        self.exposure_timer.stop()  # 기존 타이머 취소
        self.exposure_timer.start(100)  # 100ms 후 적용
        print(f"[DEBUG] Exposure 업데이트 스케줄링: {exposure} μs (100ms 후 적용)")
    
    def _apply_exposure_setting(self):
        """타이머 timeout 시 실제 Exposure 설정 적용"""
        if self._pending_exposure is not None:
            exposure_value = self._pending_exposure
            self._pending_exposure = None  # 먼저 None으로 설정하여 중복 호출 방지
            print(f"[DEBUG] Exposure 설정 적용 시작: {exposure_value} μs")
            if self.thread and self.thread.camera and self.thread.camera.IsOpen():
                self.thread.update_camera_exposure(exposure_value)
            else:
                print("[WARNING] 카메라가 연결되지 않았습니다")
        else:
            print("[WARNING] _apply_exposure_setting: _pending_exposure가 None입니다")
    
    def _schedule_resolution_update(self, width, height):
        """해상도 업데이트 스케줄링 (Debouncing)"""
        if width is not None:
            self._pending_width = width
        if height is not None:
            self._pending_height = height
        self.resolution_timer.stop()
        self.resolution_timer.start(300)  # 300ms 후 적용 (해상도 변경은 더 긴 지연)
        print(f"[DEBUG] 해상도 업데이트 스케줄링: Width={self._pending_width}, Height={self._pending_height} (300ms 후 적용)")
    
    def _apply_resolution_setting(self):
        """타이머 timeout 시 실제 해상도 설정 적용"""
        width_value = self._pending_width
        height_value = self._pending_height
        self._pending_width = None
        self._pending_height = None
        
        if width_value is not None or height_value is not None:
            print(f"[DEBUG] 해상도 설정 적용 시작: Width={width_value}, Height={height_value}")
            if self.thread and self.thread.camera and self.thread.camera.IsOpen():
                if width_value is not None:
                    self.thread.update_camera_width(width_value)
                if height_value is not None:
                    self.thread.update_camera_height(height_value)
            else:
                print("[WARNING] 카메라가 연결되지 않았습니다")
    
    def on_camera_status_changed(self, connected):
        """카메라 연결 상태 변경"""
        if not connected:
            self.camera_label.setText("NO_CAMERA")
            self.camera_label.setStyleSheet("""
                QLabel {
                    background-color: #000000;
                    color: #ffa500;
                    font-size: 24px;
                    font-weight: bold;
                }
            """)
        else:
            # 카메라 연결 성공 시 UI SpinBox 값 동기화 (카메라에 실제 적용된 값으로)
            if self.thread and self.thread.camera and self.thread.camera.IsOpen():
                try:
                    if hasattr(self.thread, 'camera_width'):
                        self.width_spinbox.setValue(self.thread.camera_width)
                    if hasattr(self.thread, 'camera_height'):
                        self.height_spinbox.setValue(self.thread.camera_height)
                except Exception as e:
                    print(f"[WARNING] 카메라 연결 시 UI 동기화 실패: {e}")


def main():
    app = QApplication(sys.argv)
    window = BatteryInspector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

