"""
AutoEncoder 및 MobileNetV3 분류기 학습 스크립트
AutoEncoder and MobileNetV3 Classifier Training Script

목적 (Purpose):
    정상 및 불량 이미지가 동일 폴더에 혼합된 데이터셋을 대상으로 딥러닝 모델 학습을 수행합니다.
    Trains deep learning models on datasets where normal and defect images are mixed in the same folders.

주요 기능 (Features):
    - 파일명 패턴 기반 자동 분류: 파일명에 "defect", "normal", "NG", "OK" 등 키워드로 자동 라벨링
      Filename pattern-based auto labeling using keywords in filenames
    - CSV/JSON 라벨 파일 기반 분류: 외부 라벨 파일에서 정상/불량 정보 로드
      Label file support (CSV, JSON) for external labeling
    - 사용자 정의 정규식: custom_pattern으로 유연한 파일명 매칭
      Custom regex pattern for flexible filename matching
    - 다중 클래스 지원: Normal(0), Damaged(1), Pollution(2) 또는 2클래스
      Multi-class support: 3-class (Normal/Damaged/Pollution) or 2-class mode

사용 방법 (Usage):
    1. config.yaml에서 image_dir, label_file 경로 설정 (또는 .env의 DATA_DIR 사용)
    2. python train_autoencoder_mixed.py 실행
"""

from pathlib import Path
import logging
from datetime import datetime
import yaml
import os
import csv
import json
import re
import glob
from dotenv import load_dotenv

load_dotenv()  # .env에서 DATA_BASE_DIR, DATA_DIR 등 경로 오버라이드 로드
               # Load path overrides (DATA_BASE_DIR, DATA_DIR) from .env

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm


def load_config(config_path=None):
    """
    YAML 설정 파일을 로드합니다. / Load YAML configuration file.
    
    Args:
        config_path: 설정 파일 경로. None이면 스크립트와 같은 폴더의 config.yaml 사용
                     Config file path. None uses config.yaml in script directory.
        
    Returns:
        dict: 로드된 설정 딕셔너리 / Loaded config dictionary
        
    Raises:
        FileNotFoundError: 설정 파일이 없을 때 / When config file does not exist
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


class Config:
    """
    학습에 필요한 모든 설정값을 관리하는 클래스.
    Central configuration class for training parameters, data paths, and model settings.
    
    config.yaml에서 로드하며, .env의 DATA_DIR, DATA_BASE_DIR로 경로를 오버라이드할 수 있습니다.
    Loads from config.yaml; paths can be overridden via DATA_DIR, DATA_BASE_DIR in .env.
    
    Attributes:
        BASE_DIR: 경로 기준점 (config base_dir 또는 환경변수) / Path base
        MIXED_IMAGE_DIRS: 이미지 폴더 경로 리스트 / Image folder paths
        LABEL_FILES: 라벨 파일/폴더 경로 리스트 / Label file paths
        RUNS_DIR: 체크포인트/로그 저장 폴더 / Output directory for checkpoints, logs
    """
    def __init__(self, config_path=None):
        config = load_config(config_path)
        common = config.get('common', {})
        training = config.get('training', {})
        
        # 경로 설정 우선순위: 1) DATA_BASE_DIR(환경변수), 2) config base_dir, 3) 스크립트 폴더
        # Path priority: 1) DATA_BASE_DIR env, 2) config base_dir, 3) script directory
        self.BASE_DIR = Path(__file__).resolve().parent
        env_base = os.environ.get('DATA_BASE_DIR')
        if env_base:
            self.BASE_DIR = Path(env_base).resolve()
        elif common.get('base_dir') != ".":
            self.BASE_DIR = Path(common['base_dir']).resolve()
        
        # 혼합 데이터셋 경로 (여러 폴더 지원)
        # Mixed dataset paths (supports multiple folders)
        # DATA_DIR 환경변수로 전체 경로 오버라이드 가능 / Override with DATA_DIR env var
        mixed_dataset = training.get('mixed_dataset', {})
        image_dir_config = os.environ.get('DATA_DIR') or mixed_dataset.get('image_dir', 'datasets/mixed')
        
        # 리스트로 여러 폴더 지정 가능
        if isinstance(image_dir_config, list):
            self.MIXED_IMAGE_DIRS = []
            for dir_path in image_dir_config:
                img_dir = Path(dir_path)
                if not img_dir.is_absolute():
                    img_dir = self.BASE_DIR / img_dir
                self.MIXED_IMAGE_DIRS.append(img_dir)
        else:
            # 단일 폴더 또는 상위 폴더 + 패턴
            img_dir = Path(image_dir_config)
            if not img_dir.is_absolute():
                img_dir = self.BASE_DIR / img_dir
            
            # 폴더 패턴 지원 (예: "TS_Exterior_Img_Datasets_images_*" 또는 리스트)
            folder_pattern = mixed_dataset.get('folder_pattern', None)
            if folder_pattern and img_dir.exists() and img_dir.is_dir():
                # 여러 패턴 지원 (리스트 또는 단일 문자열)
                if isinstance(folder_pattern, list):
                    patterns = folder_pattern
                else:
                    patterns = [folder_pattern]
                
                matched_dirs = []
                for pattern in patterns:
                    pattern_path = img_dir / pattern
                    # Windows 경로 호환성: 슬래시로 통일하고 정규화
                    pattern_str = str(pattern_path).replace('\\', '/')
                    found_dirs = [Path(p) for p in glob.glob(pattern_str) if Path(p).is_dir()]
                    matched_dirs.extend(found_dirs)
                
                if matched_dirs:
                    self.MIXED_IMAGE_DIRS = matched_dirs
                    # logger는 나중에 초기화되므로 print 사용
                    print(f"Found {len(matched_dirs)} folders matching patterns: {patterns}")
                else:
                    self.MIXED_IMAGE_DIRS = [img_dir]
            else:
                self.MIXED_IMAGE_DIRS = [img_dir]
        
        # 하위 호환성을 위해 단일 폴더도 지원
        self.MIXED_IMAGE_DIR = self.MIXED_IMAGE_DIRS[0] if self.MIXED_IMAGE_DIRS else None
        
        # 라벨 파일 경로 (선택사항, 여러 경로 지원)
        # Label file paths (optional, supports multiple paths)
        # DATA_DIR 사용 시 Training/label, Validation/label 하위로 자동 설정
        # When DATA_DIR is set, labels are auto-resolved under Training/label, Validation/label
        label_file_config = mixed_dataset.get('label_file', None)
        if os.environ.get('DATA_DIR') and label_file_config:
            data_dir = Path(os.environ['DATA_DIR'])
            label_file_config = [
                str(data_dir / "Training" / "label" / "TL_Exterior_Img_Datasets_label"),
                str(data_dir / "Validation" / "label" / "VL_Exterior_Img_Datasets_label"),
            ]
        if label_file_config:
            if isinstance(label_file_config, list):
                # 여러 라벨 폴더 지원
                self.LABEL_FILES = []
                for label_path in label_file_config:
                    label_file = Path(label_path)
                    if not label_file.is_absolute():
                        label_file = self.BASE_DIR / label_file
                    self.LABEL_FILES.append(label_file)
                self.LABEL_FILE = self.LABEL_FILES[0]  # 하위 호환성
            else:
                # 단일 라벨 파일/폴더
                self.LABEL_FILE = Path(label_file_config)
                if not self.LABEL_FILE.is_absolute():
                    self.LABEL_FILE = self.BASE_DIR / self.LABEL_FILE
                self.LABEL_FILES = [self.LABEL_FILE]
        else:
            self.LABEL_FILE = None
            self.LABEL_FILES = []
        
        # 파일명 패턴 설정
        self.FILENAME_PATTERN_MODE = mixed_dataset.get('filename_pattern_mode', 'auto')  # 'auto', 'normal_keywords', 'defect_keywords', 'custom'
        self.NORMAL_KEYWORDS = mixed_dataset.get('normal_keywords', ['normal', 'ok', 'good', 'pass', '정상', '양품'])
        self.DEFECT_KEYWORDS = mixed_dataset.get('defect_keywords', ['defect', 'ng', 'bad', 'fail', '불량', '불량품'])
        self.CUSTOM_PATTERN = mixed_dataset.get('custom_pattern', None)  # 정규식 패턴
        self.MULTI_CLASS = mixed_dataset.get('multi_class', False)  # 다중 클래스 모드 (3개 클래스: normal, Damaged, Pollution)
        
        self.RUNS_DIR = self.BASE_DIR / common.get('runs_dir', 'runs')
        self.RUNS_DIR.mkdir(exist_ok=True)
        self.LOG_DIR = self.RUNS_DIR / "logs"
        self.LOG_DIR.mkdir(exist_ok=True)
        
        # 이미지 설정
        self.IMAGE_SIZE = common.get('image_size', 320)  # 기본값 128 -> 320 (config.yaml과 일치)
        self.USE_BILATERAL = common.get('use_bilateral', True)
        bilateral = common.get('bilateral', {})
        self.BILATERAL_D = bilateral.get('d', 9)
        self.BILATERAL_SIGMA_COLOR = bilateral.get('sigma_color', 75)
        self.BILATERAL_SIGMA_SPACE = bilateral.get('sigma_space', 75)
        
        # 학습 설정
        self.BATCH_SIZE = training.get('batch_size', 16)
        self.EPOCHS = training.get('epochs', 15)
        self.LR = training.get('learning_rate', 1e-3)
        self.MODEL_NAME = training.get('model_name', 'autoencoder_mixed.pth')
        self.RESUME = training.get('resume', True)  # 학습 재개 여부
        self.CHECKPOINT_NAME = training.get('checkpoint_name', 'checkpoint.pth')  # 체크포인트 파일명
        self.SAVE_EPOCH_MODELS = training.get('save_epoch_models', True)  # 각 epoch마다 모델 저장 여부
        
        # 학습률 스케줄러 설정
        self.USE_LR_SCHEDULER = training.get('learning_rate_schedule', False)
        self.LR_SCHEDULER_TYPE = training.get('lr_scheduler_type', 'step')  # 'step' or 'cosine'
        self.LR_DECAY_FACTOR = training.get('lr_decay_factor', 0.5)
        self.LR_STEP_SIZE = training.get('lr_step_size', 20)
        self.MIN_LR = training.get('min_lr', 1e-5)
        self.WEIGHT_DECAY = training.get('weight_decay', 0.0)  # Weight Decay (L2 정규화)
        
        # 정규화 설정
        regularization = training.get('regularization', {})
        self.DROPOUT = regularization.get('dropout', 0.0)  # Dropout 비율
        self.LABEL_SMOOTHING = regularization.get('label_smoothing', 0.0)  # Label Smoothing
        
        # Focal Loss 설정 (불균형 데이터셋 최적화)
        self.USE_FOCAL_LOSS = regularization.get('use_focal_loss', True)  # Focal Loss 사용 여부
        self.FOCAL_LOSS_GAMMA = regularization.get('focal_loss_gamma', 2.0)  # Focal Loss gamma 값
        
        # WeightedRandomSampler 설정 (배치 내 클래스 균형)
        self.USE_WEIGHTED_SAMPLER = training.get('use_weighted_sampler', True)  # WeightedRandomSampler 사용 여부
        
        # 클래스 가중치 수동 오버라이드 (불균형 보정용)
        # 예: [1.0, 40.0, 0.5] -> normal, Damaged, Pollution
        self.CLASS_WEIGHT_OVERRIDE = training.get('class_weight_override', None)
        
        # Early Stopping 설정
        early_stopping = training.get('early_stopping', {})
        self.USE_EARLY_STOPPING = early_stopping.get('enabled', False)
        self.EARLY_STOPPING_PATIENCE = early_stopping.get('patience', 5)
        self.EARLY_STOPPING_MIN_DELTA = early_stopping.get('min_delta', 0.001)
        
        # 학습 방식 선택: normal_only, contrastive, classifier(라벨 supervised)
        self.TRAINING_MODE = training.get('training_mode', 'classifier')
        self.CONTRASTIVE_WEIGHT = training.get('contrastive_weight', 0.1)
        
        # 성능 최적화 설정
        optimization = training.get('optimization', {})
        self.NUM_WORKERS = optimization.get('num_workers', 2)  # 데이터 로딩 워커 수 (외장하드 사용 시 2-4 권장)
        self.PIN_MEMORY = optimization.get('pin_memory', True)  # GPU 사용 시 True로 설정 권장
        self.PREFETCH_FACTOR = optimization.get('prefetch_factor', 2)  # 데이터 프리페칭
        self.USE_MIXED_PRECISION = optimization.get('use_mixed_precision', True)  # FP16 학습 (GPU만)
        self.GRADIENT_ACCUMULATION_STEPS = optimization.get('gradient_accumulation_steps', 1)  # 그래디언트 누적
        
        # 데이터 증강 설정
        augmentation = training.get('augmentation', {})
        self.USE_AUGMENTATION = augmentation.get('use_augmentation', False)
        self.AUG_HORIZONTAL_FLIP = augmentation.get('horizontal_flip', 0.5)
        self.AUG_ROTATION = augmentation.get('rotation', 10)
        self.AUG_COLOR_JITTER = augmentation.get('color_jitter', 0.2)
        self.AUG_BRIGHTNESS = augmentation.get('brightness', 0.1)
        self.AUG_CONTRAST = augmentation.get('contrast', 0.1)
        self.AUG_RANDOM_ERASING = augmentation.get('random_erasing', 0.0)  # Random Erasing 확률
        
        # 디바이스 설정
        device_setting = common.get('device', 'auto')
        if device_setting == 'auto':
            # 사용 가능한 모든 GPU를 시도하고, 실패하면 CPU로 전환
            self.DEVICE = self._auto_select_device()
        else:
            self.DEVICE = torch.device(device_setting)
    
    def _auto_select_device(self):
        """
        사용 가능한 GPU를 자동으로 선택합니다.
        
        GPU가 없거나 오류가 발생할 경우 CPU로 자동 전환합니다.
        
        Returns:
            torch.device: 선택된 디바이스
        """
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device("cpu")
        
        # 모든 사용 가능한 GPU를 시도
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} CUDA GPU(s), testing compatibility...")
        
        for i in range(gpu_count):
            try:
                device = torch.device(f"cuda:{i}")
                gpu_name = torch.cuda.get_device_name(i)
                print(f"Testing GPU {i}: {gpu_name}...")
                
                # 간단한 테스트로 GPU가 작동하는지 확인
                test_tensor = torch.randn(1, 3, 32, 32).to(device)
                test_conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
                test_output = test_conv(test_tensor)
                del test_tensor, test_conv, test_output
                torch.cuda.empty_cache()
                
                print(f"Successfully selected GPU {i}: {gpu_name}")
                return device
            except RuntimeError as e:
                print(f"GPU {i} ({gpu_name if i < gpu_count else 'unknown'}) failed: {e}")
                print(f"  Trying next GPU or CPU...")
                continue
        
        # 모든 GPU가 실패하면 CPU 사용
        print("All GPUs failed, falling back to CPU")
        return torch.device("cpu")
    
    @classmethod
    def get_instance(cls, config_path=None):
        """
        Config 클래스의 싱글톤 인스턴스를 반환합니다.
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            Config: Config 인스턴스
        """
        if not hasattr(cls, '_instance'):
            cls._instance = cls(config_path)
        return cls._instance


_config_instance = None

def get_config(config_path=None):
    """
    전역 Config 인스턴스를 반환합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        Config: Config 인스턴스
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.get_instance(config_path)
    return _config_instance

Config = get_config()


def setup_logging():
    """
    로깅 시스템을 초기화합니다.
    
    파일 및 콘솔에 로그를 출력하도록 설정합니다.
    
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    log_file = Config.LOG_DIR / f"train_mixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


logger = setup_logging()


def load_labels_from_file(label_file, multi_class=False):
    """
    라벨 파일에서 이미지-라벨 매핑을 로드합니다.
    
    지원 형식:
        - CSV: filename,label 또는 filename,status (label: 0=normal, 1=defect)
        - JSON 단일 파일: {"filename1": 0, "filename2": 1} 또는 {"filename1": "normal", "filename2": "defect"}
        - JSON 폴더: 각 JSON 파일이 {"image_info": {"file_name": "...", "is_normal": true/false}, "defects": [...]} 형식
    
    Args:
        label_file: 라벨 파일 또는 폴더 경로
        multi_class: True인 경우 3개 클래스 (normal=0, Damaged=1, Pollution=2),
                     False인 경우 2개 클래스 (normal=0, defect=1)
    
    Returns:
        dict: 파일명을 키로, 라벨을 값으로 하는 딕셔너리
    """
    label_path = Path(label_file)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file/directory not found: {label_path}")
    
    labels = {}
    
    # 폴더인 경우 (JSON 파일들이 여러 개, 하위 폴더 포함 재귀 검색)
    if label_path.is_dir():
        json_files = list(label_path.rglob("*.json"))  # rglob으로 하위 폴더까지 재귀 검색
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON files found in {label_path}")
        
        logger.info(f"Loading labels from {len(json_files)} JSON files in {label_path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 배터리 검사 JSON 형식: image_info.is_normal 필드 사용
                    if 'image_info' in data:
                        image_info = data.get('image_info', {})
                        filename = image_info.get('file_name', '')
                        is_normal = image_info.get('is_normal', None)
                        defects = data.get('defects', [])
                        
                        if filename:
                            if is_normal == True:
                                # 정상 이미지
                                labels[filename] = 0
                            elif is_normal == False:
                                # 불량 이미지: multi_class 모드인 경우 defects 배열의 name 필드로 세부 분류
                                if multi_class and len(defects) > 0:
                                    # defects 배열에서 가장 많이 나타나는 타입 선택
                                    defect_types = [d.get('name', 'Damaged') for d in defects if isinstance(d, dict) and 'name' in d]
                                    if defect_types:
                                        # 가장 많이 나타나는 타입
                                        from collections import Counter
                                        most_common_type = Counter(defect_types).most_common(1)[0][0]
                                        
                                        # 클래스 매핑: Damaged=1, Pollution=2
                                        if most_common_type == 'Pollution':
                                            labels[filename] = 2
                                        elif most_common_type == 'Damaged':
                                            labels[filename] = 1
                                        else:
                                            # 알 수 없는 타입은 Damaged로 처리
                                            labels[filename] = 1
                                    else:
                                        # defects가 있지만 name이 없으면 Damaged로 처리
                                        labels[filename] = 1
                                else:
                                    # 2개 클래스 모드: defect=1
                                    labels[filename] = 1
                            elif 'defects' in data:
                                # is_normal이 없으면 defects 배열로 판단
                                if multi_class and len(defects) > 0:
                                    defect_types = [d.get('name', 'Damaged') for d in defects if isinstance(d, dict) and 'name' in d]
                                    if defect_types:
                                        from collections import Counter
                                        most_common_type = Counter(defect_types).most_common(1)[0][0]
                                        if most_common_type == 'Pollution':
                                            labels[filename] = 2
                                        elif most_common_type == 'Damaged':
                                            labels[filename] = 1
                                        else:
                                            labels[filename] = 1
                                    else:
                                        labels[filename] = 1
                                else:
                                    labels[filename] = 0 if len(defects) == 0 else 1
                            else:
                                logger.warning(f"No label info found in {json_file.name}, skipping")
                    # 단순 JSON 형식: {"filename": 0/1}
                    elif isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, bool)):
                                labels[key] = 0 if (value == 0 or value is False) else 1
                            elif isinstance(value, str):
                                if value.lower() in ['normal', 'ok', 'good', 'pass', '정상', '양품']:
                                    labels[key] = 0
                                elif value.lower() in ['defect', 'ng', 'bad', 'fail', '불량', '불량품']:
                                    labels[key] = 1
            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}, skipping")
                continue
        
        logger.info(f"Loaded {len(labels)} labels from {len(json_files)} JSON files")
        return labels
    
    # 단일 파일인 경우
    if label_path.suffix.lower() == '.csv':
        with open(label_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', row.get('file', ''))
                label_str = row.get('label', row.get('status', row.get('class', '')))
                
                # 라벨을 숫자로 변환
                if label_str.lower() in ['normal', 'ok', 'good', 'pass', '0', '정상', '양품']:
                    labels[filename] = 0
                elif label_str.lower() in ['defect', 'ng', 'bad', 'fail', '1', '불량', '불량품']:
                    labels[filename] = 1
                else:
                    try:
                        labels[filename] = int(label_str)
                    except ValueError:
                        logger.warning(f"Unknown label '{label_str}' for {filename}, skipping")
    
    elif label_path.suffix.lower() == '.json':
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 배터리 검사 JSON 형식
            if 'image_info' in data:
                image_info = data.get('image_info', {})
                filename = image_info.get('file_name', '')
                is_normal = image_info.get('is_normal', None)
                defects = data.get('defects', [])
                
                if filename:
                    if is_normal == True:
                        labels[filename] = 0
                    elif is_normal == False:
                        if multi_class and len(defects) > 0:
                            defect_types = [d.get('name', 'Damaged') for d in defects if isinstance(d, dict) and 'name' in d]
                            if defect_types:
                                from collections import Counter
                                most_common_type = Counter(defect_types).most_common(1)[0][0]
                                if most_common_type == 'Pollution':
                                    labels[filename] = 2
                                elif most_common_type == 'Damaged':
                                    labels[filename] = 1
                                else:
                                    labels[filename] = 1
                            else:
                                labels[filename] = 1
                        else:
                            labels[filename] = 1
                    elif 'defects' in data:
                        if multi_class and len(defects) > 0:
                            defect_types = [d.get('name', 'Damaged') for d in defects if isinstance(d, dict) and 'name' in d]
                            if defect_types:
                                from collections import Counter
                                most_common_type = Counter(defect_types).most_common(1)[0][0]
                                if most_common_type == 'Pollution':
                                    labels[filename] = 2
                                elif most_common_type == 'Damaged':
                                    labels[filename] = 1
                                else:
                                    labels[filename] = 1
                            else:
                                labels[filename] = 1
                        else:
                            labels[filename] = 0 if len(defects) == 0 else 1
            # 단순 JSON 형식
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, bool)):
                        labels[key] = 0 if (value == 0 or value is False) else 1
                    elif isinstance(value, str):
                        if value.lower() in ['normal', 'ok', 'good', 'pass', '정상', '양품']:
                            labels[key] = 0
                        elif value.lower() in ['defect', 'ng', 'bad', 'fail', '불량', '불량품']:
                            labels[key] = 1
                    else:
                        logger.warning(f"Unknown label '{value}' for {key}, skipping")
    
    logger.info(f"Loaded {len(labels)} labels from {label_path}")
    return labels


def classify_by_filename(filename, normal_keywords, defect_keywords, custom_pattern=None):
    """
    파일명을 기반으로 정상/불량을 분류합니다.
    
    Args:
        filename: 이미지 파일명
        normal_keywords: 정상 이미지를 나타내는 키워드 리스트
        defect_keywords: 불량 이미지를 나타내는 키워드 리스트
        custom_pattern: 사용자 정의 정규식 패턴
    
    Returns:
        int: 0 (정상), 1 (불량), None (분류 불가)
    """
    filename_lower = filename.lower()
    
    # 커스텀 패턴 사용
    if custom_pattern:
        match = re.search(custom_pattern, filename_lower)
        if match:
            # 패턴에서 그룹을 추출하여 분류 (예: (normal|defect) 그룹)
            if 'normal' in match.group().lower() or 'ok' in match.group().lower():
                return 0
            elif 'defect' in match.group().lower() or 'ng' in match.group().lower():
                return 1
    
    # 키워드로 분류
    for keyword in normal_keywords:
        if keyword.lower() in filename_lower:
            return 0
    
    for keyword in defect_keywords:
        if keyword.lower() in filename_lower:
            return 1
    
    return None  # 분류 불가


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    체크포인트에서 모델, optimizer, scaler 상태를 로드합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model: 로드할 모델 객체
        optimizer: 로드할 optimizer 객체 (선택)
        scaler: 로드할 scaler 객체 (선택)
    
    Returns:
        int: 시작할 에폭 번호 (체크포인트가 없으면 1)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found at {checkpoint_path}, starting from epoch 1")
        return 1
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state from checkpoint")
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Loaded scaler state from checkpoint")
        
        start_epoch = checkpoint.get('epoch', 1) + 1  # 다음 에폭부터 시작
        logger.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}, starting from epoch 1")
        return 1


def save_checkpoint(checkpoint_path, model, optimizer, epoch, scaler=None):
    """
    체크포인트를 저장합니다.
    
    모델 상태, optimizer 상태, 현재 에폭 번호를 저장합니다.
    
    Args:
        checkpoint_path: 저장할 체크포인트 파일 경로
        model: 저장할 모델 객체
        optimizer: 저장할 optimizer 객체
        epoch: 현재 에폭 번호
        scaler: 저장할 scaler 객체 (선택)
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved at epoch {epoch}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


class MixedDataset(Dataset):
    """
    정상 및 불량 이미지가 동일 폴더에 혼합된 데이터셋 클래스.
    
    이미지 분류는 다음 우선순위로 수행됩니다:
        1. 라벨 파일 (CSV/JSON)
        2. 파일명 패턴
        3. 사용자 정의 분류 함수
    """

    def __init__(self, image_dir, transform=None, use_bilateral=True, 
                 label_file=None, normal_keywords=None, defect_keywords=None, 
                 custom_pattern=None, training_mode='normal_only', multi_class=False, indices=None):
        """
        Args:
            image_dir: 모든 이미지가 있는 폴더 (단일 폴더 Path 또는 폴더 리스트)
            transform: 이미지 전처리 변환
            use_bilateral: Bilateral Filter 적용 여부
            label_file: 라벨 파일 경로 (CSV 또는 JSON)
            normal_keywords: 정상 이미지를 나타내는 파일명 키워드 리스트
            defect_keywords: 불량 이미지를 나타내는 파일명 키워드 리스트
            custom_pattern: 커스텀 정규식 패턴
            training_mode: 'normal_only'일 경우 불량 이미지 제외
        """
        # 여러 폴더 지원
        if isinstance(image_dir, list):
            self.image_dirs = [Path(d) for d in image_dir]
        else:
            self.image_dirs = [Path(image_dir)]
        
        # 모든 폴더 존재 확인
        for img_dir in self.image_dirs:
            if not img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        # 모든 폴더에서 이미지 파일 찾기
        self.files = []
        for img_dir in self.image_dirs:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                self.files.extend(img_dir.glob(ext))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in directories: {[str(d) for d in self.image_dirs]}")
        
        # 하위 호환성
        self.image_dir = self.image_dirs[0]
        
        # 라벨 로드 (여러 라벨 폴더 지원)
        self.multi_class = multi_class
        labels_dict = {}
        if label_file:
            # 여러 라벨 폴더 지원
            if isinstance(label_file, list):
                for label_path in label_file:
                    label_path_obj = Path(label_path)
                    if label_path_obj.exists():
                        labels = load_labels_from_file(label_path, multi_class=multi_class)
                        labels_dict.update(labels)  # 여러 폴더의 라벨 합치기
            else:
                label_path_obj = Path(label_file)
                if label_path_obj.exists():
                    labels_dict = load_labels_from_file(label_file, multi_class=multi_class)
        
        # 파일별로 라벨 분류
        self.files_labeled = []
        self.labels = []
        normal_keywords = normal_keywords or []
        defect_keywords = defect_keywords or []
        
        for img_path in self.files:
            filename = img_path.name
            
            # 1. 라벨 파일에서 찾기
            label = None
            if filename in labels_dict:
                label = labels_dict[filename]
            elif str(img_path) in labels_dict:
                label = labels_dict[str(img_path)]
            else:
                # 2. 파일명 패턴으로 분류
                label = classify_by_filename(filename, normal_keywords, defect_keywords, custom_pattern)
            
            # training_mode가 'normal_only'인 경우 불량 이미지는 제외
            if training_mode == 'normal_only' and label != 0:
                continue
            
            # 분류 불가능한 경우 제외
            if label is None:
                logger.warning(f"Could not classify {filename}, skipping")
                continue
            
            self.files_labeled.append(img_path)
            self.labels.append(label)
        
        if len(self.files_labeled) == 0:
            raise RuntimeError(f"No valid labeled images found in {self.image_dir}")

        self.transform = transform
        self.use_bilateral = use_bilateral

        if self.multi_class:
            normal_count = sum(1 for l in self.labels if l == 0)
            damaged_count = sum(1 for l in self.labels if l == 1)
            pollution_count = sum(1 for l in self.labels if l == 2)
            logger.info(f"Loaded images: Normal={normal_count}, Damaged={damaged_count}, Pollution={pollution_count}, Total={len(self.files_labeled)}")
        else:
            normal_count = sum(1 for l in self.labels if l == 0)
        defect_count = sum(1 for l in self.labels if l == 1)
        logger.info(f"Loaded images: Normal={normal_count}, Defect={defect_count}, Total={len(self.files_labeled)}")

    def __len__(self):
        return len(self.files_labeled)

    def __getitem__(self, idx):
        """
        idx번째 이미지와 라벨을 반환합니다.
        
        주의: 입력 이미지는 이미 320x320 크기로 리사이즈되고 Bilateral Filter가
        적용된 상태입니다. 중복 전처리는 수행하지 않으며, 이미지 로드 및 RGB 변환만 수행합니다.
        
        Args:
            idx: 데이터셋 인덱스
            
        Returns:
            tuple: (이미지 텐서, 라벨)
        """
        img_path = self.files_labeled[idx]
        label = self.labels[idx]
        
        try:
            # 한글 경로 완벽 대응: 처음부터 imdecode 사용 (안정적)
            img_path_str = str(img_path)
            img_array = np.fromfile(img_path_str, np.uint8)
            img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_np is None:
                raise RuntimeError(f"Failed to decode image {img_path}")
            
            # BGR -> RGB 변환만 수행 (이미 전처리된 이미지이므로 리사이즈/필터 적용 안 함)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환 (이미 320x320 + Bilateral Filter 적용됨)
            img = Image.fromarray(img_np)
            
            # 크기 검증 (디버깅용, 경고만 출력)
            h, w = img_np.shape[:2]
            if h != Config.IMAGE_SIZE or w != Config.IMAGE_SIZE:
                logger.warning(f"Image {img_path} size mismatch: expected {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}, got {w}x{h}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        # Transform 적용 (ToTensor + Normalize만, 리사이즈/필터 없음)
        try:
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Failed to apply transform to {img_path}: {e}")

        return img, label


class FocalLoss(nn.Module):
    """
    Focal Loss 구현 클래스.
    
    불균형 데이터셋에 효과적인 손실 함수로, 잘못 분류된 샘플에 더 높은 가중치를 부여합니다.
    """
    def __init__(self, alpha=None, gamma=2.0, weight=None, label_smoothing=0.0):
        """
        Args:
            alpha: 클래스별 가중치 (Tensor 또는 None)
            gamma: Focal Loss 강도 (기본값: 2.0, 높을수록 어려운 샘플에 더 집중)
            weight: 클래스 가중치 (deprecated, alpha 사용)
            label_smoothing: Label Smoothing 비율
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 모델 출력 (batch_size, num_classes)
            targets: 정답 라벨 (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, 
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)  # 정확도에 대한 확률
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 클래스별 가중치 적용
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class ConvAutoEncoder(nn.Module):
    """
    CNN 기반 AutoEncoder 모델.
    
    인코더-디코더 구조를 사용하여 이미지를 재구성합니다.
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_normal_only():
    """
    정상 데이터만을 사용하여 AutoEncoder를 학습합니다.
    
    불량 데이터는 제외하고 정상 이미지만으로 모델을 학습합니다.
    """
    logger.info("=" * 60)
    logger.info("Training Mode: Normal Only (from mixed folder)")
    logger.info("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # 전체 데이터셋 생성
    full_dataset = MixedDataset(
        image_dir=Config.MIXED_IMAGE_DIRS,  # 여러 폴더 지원
        transform=transform,
        use_bilateral=Config.USE_BILATERAL,
        label_file=Config.LABEL_FILES if hasattr(Config, 'LABEL_FILES') and Config.LABEL_FILES else Config.LABEL_FILE,
        normal_keywords=Config.NORMAL_KEYWORDS,
        defect_keywords=Config.DEFECT_KEYWORDS,
        custom_pattern=Config.CUSTOM_PATTERN,
        training_mode='normal_only'
    )
    
    # Train/Validation Split (80:20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드
    )
    
    logger.info(f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%), "
                f"Validation={len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    
    # Train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Validation은 shuffle 불필요
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    loader = train_loader  # 하위 호환성을 위해 유지

    logger.info(f"Device: {Config.DEVICE}")
    if Config.DEVICE.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(Config.DEVICE.index)}")
    logger.info(f"Total batches per epoch: {len(loader)}")
    logger.info(f"Batch size: {Config.BATCH_SIZE}, Gradient accumulation steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")

    model = ConvAutoEncoder().to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    # 학습률 스케줄러 설정 (실무 수준 성능 향상)
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=Config.LR_STEP_SIZE, 
            gamma=Config.LR_DECAY_FACTOR
        )
        logger.info(f"Learning Rate Scheduler enabled: step_size={Config.LR_STEP_SIZE}, decay={Config.LR_DECAY_FACTOR}")
    
    # Mixed Precision Training (FP16) - GPU만 지원
    scaler = None
    if Config.USE_MIXED_PRECISION and Config.DEVICE.type == 'cuda':
        try:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed Precision Training (FP16) enabled")
        except Exception as e:
            logger.warning(f"Mixed Precision not available: {e}, using FP32")
            scaler = None
    
    # 체크포인트 로드 (학습 재개)
    checkpoint_path = Config.RUNS_DIR / Config.CHECKPOINT_NAME
    start_epoch = 1
    if Config.RESUME:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    # Loss 기록용 CSV 파일 (Train Loss + Validation Loss)
    loss_csv_path = Config.RUNS_DIR / "training_loss.csv"
    loss_history = []
    if loss_csv_path.exists() and Config.RESUME:
        # 기존 기록 로드 (재개 시)
        with open(loss_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                loss_history.append({
                    'epoch': int(row['epoch']),
                    'train_loss': float(row.get('train_loss', row.get('loss', 0))),
                    'val_loss': float(row.get('val_loss', 0))
                })
        logger.info(f"Loaded loss history: {len(loss_history)} epochs")
    else:
        # 새 CSV 파일 생성 (Train Loss + Validation Loss)
        with open(loss_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

    # 전체 학습 진행률 바
    total_epochs = Config.EPOCHS - start_epoch + 1
    epoch_pbar = tqdm(
        range(start_epoch, Config.EPOCHS + 1),
        desc="Overall Progress",
        unit="epoch",
        ncols=100,
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]'
    )

    try:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Overall Progress (Epoch {epoch}/{Config.EPOCHS})")
            logger.info(f"Starting epoch {epoch}/{Config.EPOCHS}...")
            model.train()
            total_loss = 0.0
            total_batches = len(loader)

            # 배치 진행률 바 생성
            pbar = tqdm(
                enumerate(loader),
                total=total_batches,
                desc=f"Epoch {epoch}/{Config.EPOCHS}",
                unit="batch",
                ncols=100,
                position=1,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for batch_idx, (imgs, _) in pbar:  # normal_only 모드에서는 labels 사용 안 함
                try:
                    imgs = imgs.to(Config.DEVICE, non_blocking=True)
                    
                    # Mixed Precision Training
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(imgs)
                            loss = criterion(outputs, imgs)
                            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                        
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = model(imgs)
                        loss = criterion(outputs, imgs)
                        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                        
                        loss.backward()
                        
                        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    current_loss = loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
                    total_loss += current_loss
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    
                    # 진행률 바 업데이트 (손실 정보 포함)
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.6f}',
                        'Avg': f'{avg_loss_so_far:.6f}'
                    })
                    
                    if Config.DEVICE.type == 'cuda' and len(train_loader) > 100:
                        if (batch_idx + 1) % 50 == 0:
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"GPU 메모리 부족! 배치 {batch_idx} 건너뜀")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise RuntimeError(f"Training error at epoch {epoch}, batch {batch_idx}: {e}")
            
            pbar.close()

            if len(train_loader) > 0:
                avg_train_loss = total_loss / len(train_loader)
                
                # Validation Loss 계산
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_imgs, _ in val_loader:
                        val_imgs = val_imgs.to(Config.DEVICE, non_blocking=True)
                        val_outputs = model(val_imgs)
                        val_loss += criterion(val_outputs, val_imgs).item()
                
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                model.train()  # 다시 train 모드로
                
                logger.info(f"[{epoch}/{Config.EPOCHS}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                epoch_pbar.set_postfix({
                    'Train': f'{avg_train_loss:.6f}',
                    'Val': f'{avg_val_loss:.6f}'
                })
                
                # Loss 기록 (CSV) - Train Loss + Validation Loss
                loss_history.append({
                    'epoch': epoch, 
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                })
                with open(loss_csv_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}'])
                
                # 매 Epoch마다 모델 저장 (설정에 따라 선택적)
                if Config.SAVE_EPOCH_MODELS:
                    epoch_model_path = Config.RUNS_DIR / f"model_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), epoch_model_path)
                    logger.info(f"Model saved: {epoch_model_path}")
                    
                    # 참고: 학습 Loss만으로는 실제 이상 탐지 성능을 판단할 수 없습니다.
                    # 학습 완료 후 각 epoch 모델로 이상 점수를 계산하여 검증 데이터의 정확도를 확인하고
                    # 가장 높은 정확도를 보이는 모델을 선택하시기 바랍니다.

            # 재구성 이미지 저장 비활성화 (필요시 주석 해제)
            # if epoch == 1 or epoch % 5 == 0:
            #     try:
            #         save_path = Config.RUNS_DIR / f"recon_epoch_{epoch}.png"
            #         save_image(
            #             torch.cat([imgs[:4], outputs[:4]], dim=0),
            #             save_path,
            #             nrow=4
            #         )
            #     except Exception as e:
            #         logger.warning(f"Failed to save reconstruction image for epoch {epoch}: {e}")
            
            # 학습률 스케줄러 업데이트
            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
            
            # 체크포인트 저장 (에폭 완료 후에만 저장)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, scaler)
            epoch_pbar.update(1)
        
        epoch_pbar.close()
        
        # 학습 완료 안내
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"All models saved to: {Config.RUNS_DIR}")
        logger.info(f"Loss history saved to: {loss_csv_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Check training_loss.csv to see loss trends")
        logger.info("2. Use infer_autoencoder.py with each epoch model to calculate anomaly scores")
        logger.info("3. Evaluate accuracy on validation data to find the best model")
        logger.info("4. Select the epoch model with highest accuracy as your best model")
        logger.info("=" * 60)
            
    except KeyboardInterrupt:
        epoch_pbar.close()
        # 중단 시: 완료된 epoch만 저장 (현재 epoch이 완료되지 않았으므로 이전 epoch 저장)
        completed_epoch = epoch - 1
        if completed_epoch >= start_epoch:
            logger.info(f"\nTraining interrupted at epoch {epoch} (not completed). Saving checkpoint for completed epoch {completed_epoch}...")
            save_checkpoint(checkpoint_path, model, optimizer, completed_epoch, scaler)
            logger.info(f"Checkpoint saved. Training can be resumed from epoch {completed_epoch + 1}")
        else:
            logger.info(f"\nTraining interrupted at epoch {epoch} (before any epoch completed). No checkpoint saved.")
        return model
    
    return model


def train_contrastive():
    """
    Contrastive Learning 방식으로 모델을 학습합니다.
    
    정상과 불량 이미지 간의 대조를 통해 특징을 학습합니다.
    """
    logger.info("=" * 60)
    logger.info("Training Mode: Contrastive Learning (from mixed folder)")
    logger.info("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # 전체 데이터셋 생성
    full_dataset = MixedDataset(
        image_dir=Config.MIXED_IMAGE_DIRS,  # 여러 폴더 지원
        transform=transform,
        use_bilateral=Config.USE_BILATERAL,
        label_file=Config.LABEL_FILES if hasattr(Config, 'LABEL_FILES') and Config.LABEL_FILES else Config.LABEL_FILE,
        normal_keywords=Config.NORMAL_KEYWORDS,
        defect_keywords=Config.DEFECT_KEYWORDS,
        custom_pattern=Config.CUSTOM_PATTERN,
        training_mode='contrastive'
    )
    
    # Train/Validation Split (80:20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드
    )
    
    logger.info(f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%), "
                f"Validation={len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    
    # Train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Validation은 shuffle 불필요
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    loader = train_loader  # 하위 호환성을 위해 유지

    logger.info(f"Device: {Config.DEVICE}")
    if Config.DEVICE.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(Config.DEVICE.index)}")
    logger.info(f"Total batches per epoch: {len(loader)}")
    logger.info(f"Batch size: {Config.BATCH_SIZE}, Gradient accumulation steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")

    model = ConvAutoEncoder().to(Config.DEVICE)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    # 학습률 스케줄러 설정 (실무 수준 성능 향상)
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=Config.LR_STEP_SIZE, 
            gamma=Config.LR_DECAY_FACTOR
        )
        logger.info(f"Learning Rate Scheduler enabled: step_size={Config.LR_STEP_SIZE}, decay={Config.LR_DECAY_FACTOR}")
    
    # Mixed Precision Training (FP16) - GPU만 지원
    scaler = None
    if Config.USE_MIXED_PRECISION and Config.DEVICE.type == 'cuda':
        try:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed Precision Training (FP16) enabled")
        except Exception as e:
            logger.warning(f"Mixed Precision not available: {e}, using FP32")
            scaler = None
    
    # Loss 기록용 CSV 파일 (Train Loss + Validation Loss)
    loss_csv_path = Config.RUNS_DIR / "training_loss.csv"
    loss_history = []
    if loss_csv_path.exists() and Config.RESUME:
        # 기존 기록 로드 (재개 시)
        with open(loss_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                loss_history.append({
                    'epoch': int(row['epoch']),
                    'train_loss': float(row.get('train_loss', row.get('loss', 0))),
                    'val_loss': float(row.get('val_loss', 0))
                })
        logger.info(f"Loaded loss history: {len(loss_history)} epochs")
    else:
        # 새 CSV 파일 생성 (Train Loss + Validation Loss)
        with open(loss_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
    
    # 체크포인트 로드 (학습 재개)
    checkpoint_path = Config.RUNS_DIR / Config.CHECKPOINT_NAME
    start_epoch = 1
    if Config.RESUME:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    # 전체 학습 진행률 바
    total_epochs = Config.EPOCHS - start_epoch + 1
    epoch_pbar = tqdm(
        range(start_epoch, Config.EPOCHS + 1),
        desc="Overall Progress",
        unit="epoch",
        ncols=100,
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]'
    )

    try:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Overall Progress (Epoch {epoch}/{Config.EPOCHS})")
            logger.info(f"Starting epoch {epoch}/{Config.EPOCHS}...")
            model.train()
            total_loss = 0.0
            total_normal_loss = 0.0
            total_defect_loss = 0.0
            total_batches = len(train_loader)

            # 배치 진행률 바 생성
            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"Epoch {epoch}/{Config.EPOCHS}",
                unit="batch",
                ncols=100,
                position=1,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for batch_idx, (imgs, labels) in pbar:
                try:
                    imgs = imgs.to(Config.DEVICE, non_blocking=True)
                    labels = labels.to(Config.DEVICE)
                    
                    # Mixed Precision Training
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(imgs)
                            loss_per_sample = criterion(outputs, imgs).view(imgs.size(0), -1).mean(dim=1)
                            
                            normal_mask = (labels == 0).float()
                            normal_loss = (loss_per_sample * normal_mask).mean()
                            
                            defect_mask = (labels == 1).float()
                            defect_loss = (loss_per_sample * defect_mask).mean()
                            defect_loss = -Config.CONTRASTIVE_WEIGHT * defect_loss
                            
                            loss = normal_loss + defect_loss
                            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                        
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = model(imgs)
                        loss_per_sample = criterion(outputs, imgs).view(imgs.size(0), -1).mean(dim=1)
                        
                        normal_mask = (labels == 0).float()
                        normal_loss = (loss_per_sample * normal_mask).mean()
                        
                        defect_mask = (labels == 1).float()
                        defect_loss = (loss_per_sample * defect_mask).mean()
                        
                        # Margin Loss 도입 (Loss 발산 방지)
                        # loss = normal_loss + max(0, margin - defect_loss)
                        # 불량 이미지는 일정 수준(margin)까지만 오차가 커지면 충분함
                        margin = 0.1  # Margin 값 (조정 가능)
                        margin_loss = torch.clamp(margin - defect_loss, min=0.0)
                        loss = normal_loss + Config.CONTRASTIVE_WEIGHT * margin_loss
                        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                        
                        loss.backward()
                        
                        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    current_loss = loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
                    total_loss += current_loss
                    total_normal_loss += normal_loss.item()
                    total_defect_loss += defect_loss.item()
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    
                    # 진행률 바 업데이트 (손실 정보 포함)
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.6f}',
                        'Avg': f'{avg_loss_so_far:.6f}',
                        'N': f'{normal_loss.item():.4f}',
                        'D': f'{defect_loss.item():.4f}'
                    })
                    
                    if Config.DEVICE.type == 'cuda' and len(train_loader) > 100:
                        if (batch_idx + 1) % 50 == 0:
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"GPU 메모리 부족! 배치 {batch_idx} 건너뜀")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise RuntimeError(f"Training error at epoch {epoch}, batch {batch_idx}: {e}")
            
            pbar.close()

            if len(train_loader) > 0:
                avg_train_loss = total_loss / len(train_loader)
                avg_normal_loss = total_normal_loss / len(train_loader)
                avg_defect_loss = total_defect_loss / len(train_loader)
                
                # Validation Loss 계산
                model.eval()
                val_loss = 0.0
                val_normal_loss = 0.0
                val_defect_loss = 0.0
                with torch.no_grad():
                    for val_imgs, val_labels in val_loader:
                        val_imgs = val_imgs.to(Config.DEVICE, non_blocking=True)
                        val_labels = val_labels.to(Config.DEVICE)
                        val_outputs = model(val_imgs)
                        val_loss_per_sample = criterion(val_outputs, val_imgs).view(val_imgs.size(0), -1).mean(dim=1)
                        
                        val_normal_mask = (val_labels == 0).float()
                        val_normal_loss += (val_loss_per_sample * val_normal_mask).mean().item()
                        
                        val_defect_mask = (val_labels == 1).float()
                        val_defect_loss += (val_loss_per_sample * val_defect_mask).mean().item()
                        
                        val_loss += val_loss_per_sample.mean().item()
                
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                avg_val_normal_loss = val_normal_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                avg_val_defect_loss = val_defect_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                model.train()  # 다시 train 모드로
                
                logger.info(f"[{epoch}/{Config.EPOCHS}] Train Loss: {avg_train_loss:.6f} "
                           f"(N:{avg_normal_loss:.4f}, D:{avg_defect_loss:.4f}), "
                           f"Val Loss: {avg_val_loss:.6f} "
                           f"(N:{avg_val_normal_loss:.4f}, D:{avg_val_defect_loss:.4f})")
                epoch_pbar.set_postfix({
                    'Train': f'{avg_train_loss:.6f}',
                    'Val': f'{avg_val_loss:.6f}',
                    'N': f'{avg_normal_loss:.4f}',
                    'D': f'{avg_defect_loss:.4f}'
                })
                
                # Loss 기록 (CSV) - Train Loss + Validation Loss
                loss_history.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                })
                with open(loss_csv_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}'])

            # 재구성 이미지 저장 비활성화 (필요시 주석 해제)
            # if epoch == 1 or epoch % 5 == 0:
            #     try:
            #         save_path = Config.RUNS_DIR / f"recon_epoch_{epoch}.png"
            #         save_image(
            #             torch.cat([imgs[:4], outputs[:4]], dim=0),
            #             save_path,
            #             nrow=4
            #         )
            #     except Exception as e:
            #         logger.warning(f"Failed to save reconstruction image for epoch {epoch}: {e}")
            
            # 학습률 스케줄러 업데이트
            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
            
            # 체크포인트 저장 (에폭 완료 후에만 저장)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, scaler)
            epoch_pbar.update(1)
        
        epoch_pbar.close()
            
    except KeyboardInterrupt:
        epoch_pbar.close()
        # 중단 시: 완료된 epoch만 저장 (현재 epoch이 완료되지 않았으므로 이전 epoch 저장)
        completed_epoch = epoch - 1
        if completed_epoch >= start_epoch:
            logger.info(f"\nTraining interrupted at epoch {epoch} (not completed). Saving checkpoint for completed epoch {completed_epoch}...")
            save_checkpoint(checkpoint_path, model, optimizer, completed_epoch, scaler)
            logger.info(f"Checkpoint saved. Training can be resumed from epoch {completed_epoch + 1}")
        else:
            logger.info(f"\nTraining interrupted at epoch {epoch} (before any epoch completed). No checkpoint saved.")
        return model
    
    return model


class ConditionalResize:
    """
    조건부 리사이즈 Transform 클래스.
    
    이미지가 목표 크기와 동일한 경우 리사이즈를 건너뜁니다.
    """
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, img):
        # PIL Image인 경우
        if isinstance(img, Image.Image):
            w, h = img.size
            # 이미 목표 크기면 그대로 반환
            if w == self.target_size and h == self.target_size:
                return img
            # 아니면 리사이즈
            return transforms.Resize((self.target_size, self.target_size))(img)
        return img


class TransformWrapper(Dataset):
    """
    데이터셋에 transform을 적용하는 래퍼 클래스.
    
    multiprocessing 환경에서도 안전하게 동작하도록 설계되었습니다.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img는 PIL Image이므로 transform 적용
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def train_classifier():
    """
    지도 학습 방식으로 MobileNetV3 분류기를 학습합니다.
    
    라벨 정보를 사용하여 정상/불량 이미지를 분류하는 모델을 학습합니다.
    """
    # 설정 값 오버라이드: 30 epoch 안정적 학습을 위해 설정
    Config.EPOCHS = 30
    Config.USE_EARLY_STOPPING = False
    logger.info("=" * 60)
    logger.info("Training Mode: Classifier (Supervised, using labels)")
    logger.info("=" * 60)
    logger.info("Config Override: EPOCHS=30, USE_EARLY_STOPPING=False (for stable 30 epochs)")
    logger.info("=" * 60)
    
    # ImageNet 정규화 (이미 전처리된 이미지이므로 리사이즈 없이 ToTensor + Normalize만)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 데이터 증강 설정 (학습 데이터에만 적용)
    # 참고: 입력 이미지는 이미 320x320 크기이므로 리사이즈는 수행하지 않음
    if getattr(Config, 'USE_AUGMENTATION', False):
        aug_transforms = [
            # ConditionalResize 제거 (이미 320x320)
            transforms.RandomHorizontalFlip(p=Config.AUG_HORIZONTAL_FLIP),  # 수평 뒤집기
            transforms.RandomRotation(degrees=Config.AUG_ROTATION),  # 회전
            transforms.ColorJitter(
                brightness=Config.AUG_BRIGHTNESS,
                contrast=Config.AUG_CONTRAST,
                saturation=Config.AUG_COLOR_JITTER,
                hue=Config.AUG_COLOR_JITTER * 0.5
            ),  # 색상 변화
            transforms.ToTensor(),
        ]
        
        # Random Erasing 추가
        if hasattr(Config, 'AUG_RANDOM_ERASING') and Config.AUG_RANDOM_ERASING > 0:
            aug_transforms.append(transforms.RandomErasing(p=Config.AUG_RANDOM_ERASING))
        
        aug_transforms.append(normalize)
        train_transform = transforms.Compose(aug_transforms)
        
        aug_info = f"flip={Config.AUG_HORIZONTAL_FLIP}, rotation={Config.AUG_ROTATION}°, "
        aug_info += f"color_jitter={Config.AUG_COLOR_JITTER}, brightness={Config.AUG_BRIGHTNESS}, contrast={Config.AUG_CONTRAST}"
        if hasattr(Config, 'AUG_RANDOM_ERASING') and Config.AUG_RANDOM_ERASING > 0:
            aug_info += f", random_erasing={Config.AUG_RANDOM_ERASING}"
        logger.info(f"Data augmentation enabled (no resize, already 320x320): {aug_info}")
    else:
        train_transform = transforms.Compose([
            # ConditionalResize 제거 (이미 320x320)
            transforms.ToTensor(),
            normalize,
        ])
        logger.info("Data augmentation disabled (no resize, already 320x320)")
    
    # 검증 데이터용 transform (증강 없음, 리사이즈 없음)
    val_transform = transforms.Compose([
        # ConditionalResize 제거 (이미 320x320)
        transforms.ToTensor(),
        normalize,
    ])

    # 다중 클래스 모드 설정 (config.yaml에서 읽거나 기본값 False)
    MULTI_CLASS = getattr(Config, 'MULTI_CLASS', False)  # 기본값: 2개 클래스 모드
    
    # 전체 데이터셋 생성 (라벨 정보 사용, 정상 및 불량 이미지 모두 포함)
    # transform은 나중에 train/val에 각각 적용
    full_dataset = MixedDataset(
        image_dir=Config.MIXED_IMAGE_DIRS,
        transform=None,  # 나중에 train/val transform 적용
        use_bilateral=Config.USE_BILATERAL,
        label_file=Config.LABEL_FILES if hasattr(Config, 'LABEL_FILES') and Config.LABEL_FILES else Config.LABEL_FILE,
        normal_keywords=Config.NORMAL_KEYWORDS,
        defect_keywords=Config.DEFECT_KEYWORDS,
        custom_pattern=Config.CUSTOM_PATTERN,
        training_mode='classifier',
        multi_class=MULTI_CLASS
    )

    # 클래스 비율 기반 가중치 (개선된 계산 방식)
    labels_np = np.array(full_dataset.labels)
    num_classes = 3 if MULTI_CLASS else 2
    
    if MULTI_CLASS:
        normal_count = (labels_np == 0).sum()
        damaged_count = (labels_np == 1).sum()
        pollution_count = (labels_np == 2).sum()
        total_count = len(labels_np)
        
        # 개선된 가중치 계산: 역빈도 기반 (더 극단적인 불균형에 대응)
        # 가장 많은 클래스의 샘플 수를 기준으로 가중치 계산
        max_count = max(normal_count, damaged_count, pollution_count)
        
        weight_normal = max_count / normal_count if normal_count > 0 else 1.0
        weight_damaged = max_count / damaged_count if damaged_count > 0 else 1.0
        weight_pollution = max_count / pollution_count if pollution_count > 0 else 1.0
        
        # 정규화 (가장 작은 가중치를 1.0으로 맞춤)
        min_weight = min(weight_normal, weight_damaged, weight_pollution)
        weight_normal = weight_normal / min_weight
        weight_damaged = weight_damaged / min_weight
        weight_pollution = weight_pollution / min_weight
        
        auto_weights = [weight_normal, weight_damaged, weight_pollution]
        
        # 수동 오버라이드가 있으면 적용 (length == 3)
        override = getattr(Config, 'CLASS_WEIGHT_OVERRIDE', None)
        if override and isinstance(override, (list, tuple)) and len(override) == 3:
            class_weights = torch.tensor(override, dtype=torch.float32, device=Config.DEVICE)
            logger.info(f"Class weights override applied: Normal={override[0]:.3f}, Damaged={override[1]:.3f}, Pollution={override[2]:.3f}")
        else:
            class_weights = torch.tensor(auto_weights, dtype=torch.float32, device=Config.DEVICE)
            logger.info(f"Class weights (auto, improved): Normal={weight_normal:.3f}, Damaged={weight_damaged:.3f}, Pollution={weight_pollution:.3f}")
            logger.info(f"Class distribution: Normal={normal_count} ({normal_count/total_count*100:.1f}%), "
                       f"Damaged={damaged_count} ({damaged_count/total_count*100:.1f}%), "
                       f"Pollution={pollution_count} ({pollution_count/total_count*100:.1f}%)")
    else:
        normal_count = (labels_np == 0).sum()
        defect_count = (labels_np == 1).sum()
        total_count = len(labels_np)
        max_count = max(normal_count, defect_count)
        weight_normal = max_count / normal_count if normal_count > 0 else 1.0
        weight_defect = max_count / defect_count if defect_count > 0 else 1.0
        min_weight = min(weight_normal, weight_defect)
        weight_normal = weight_normal / min_weight
        weight_defect = weight_defect / min_weight
        class_weights = torch.tensor([weight_normal, weight_defect], dtype=torch.float32, device=Config.DEVICE)
        logger.info(f"Class weights: Normal={weight_normal:.3f}, Defect={weight_defect:.3f}")

    # Train/Validation Split (80:20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 학습용과 검증용 데이터셋 생성 (각각 다른 transform 적용)
    # 학습용: 증강 적용, 검증용: 증강 없음
    train_subset = Subset(full_dataset, train_indices.indices)
    val_subset = Subset(full_dataset, val_indices.indices)
    
    # 각 Subset에 transform 적용 (전역 TransformWrapper 클래스 사용 - multiprocessing 호환)
    train_dataset = TransformWrapper(train_subset, train_transform)
    val_dataset = TransformWrapper(val_subset, val_transform)
    
    logger.info(f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%), "
                f"Validation={len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")

    # WeightedRandomSampler 추가 (불균형 데이터셋 대응)
    # 각 배치에서 클래스 비율을 균형있게 샘플링
    use_weighted_sampler = getattr(Config, 'USE_WEIGHTED_SAMPLER', True)
    train_sampler = None
    
    if use_weighted_sampler:
        # 학습 데이터셋의 라벨 가져오기
        train_labels = [full_dataset.labels[idx] for idx in train_indices.indices]
        train_labels_np = np.array(train_labels)
        
        # 클래스별 샘플 수 계산
        if MULTI_CLASS:
            class_counts = np.array([
                (train_labels_np == 0).sum(),  # Normal
                (train_labels_np == 1).sum(),  # Damaged
                (train_labels_np == 2).sum()   # Pollution
            ])
        else:
            class_counts = np.array([
                (train_labels_np == 0).sum(),  # Normal
                (train_labels_np == 1).sum()   # Defect
            ])
        
        # 각 클래스에 대한 가중치 계산 (역빈도)
        class_weights_for_sampler = 1.0 / (class_counts + 1e-6)  # 0으로 나누기 방지
        # 정규화
        class_weights_for_sampler = class_weights_for_sampler / class_weights_for_sampler.sum() * len(class_weights_for_sampler)
        
        # 각 샘플에 대한 가중치 할당
        sample_weights = class_weights_for_sampler[train_labels_np]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True  # 소수 클래스를 더 자주 샘플링하기 위해 replacement=True
        )
        
        logger.info(f"WeightedRandomSampler enabled:")
        if MULTI_CLASS:
            logger.info(f"  Class weights: Normal={class_weights_for_sampler[0]:.3f}, "
                       f"Damaged={class_weights_for_sampler[1]:.3f}, "
                       f"Pollution={class_weights_for_sampler[2]:.3f}")
            logger.info(f"  Class counts: Normal={class_counts[0]}, "
                       f"Damaged={class_counts[1]}, "
                       f"Pollution={class_counts[2]}")
        else:
            logger.info(f"  Class weights: Normal={class_weights_for_sampler[0]:.3f}, "
                       f"Defect={class_weights_for_sampler[1]:.3f}")
            logger.info(f"  Class counts: Normal={class_counts[0]}, Defect={class_counts[1]}")
        logger.info("")
        logger.info("  IMPORTANT: First batch label distribution will be checked")
        logger.info("     Expected: Balanced (e.g., Normal: 25 (78%), Defect: 7 (22%))")
        logger.info("     If all same class, WeightedRandomSampler may not be working")
    else:
        logger.info("WeightedRandomSampler disabled (standard shuffle)")
        logger.info("  WARNING: Without WeightedRandomSampler, batches may be imbalanced")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,  # sampler 사용 시 shuffle=False (자동)
        shuffle=(train_sampler is None),  # sampler가 없을 때만 shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY if Config.DEVICE.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )

    logger.info(f"Device: {Config.DEVICE}")
    if Config.DEVICE.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(Config.DEVICE.index)}")
    logger.info(f"Total batches per epoch: {len(train_loader)}")

    # 모델: MobileNetV3 분류기
    num_classes = 3 if MULTI_CLASS else 2
    try:
        # ImageNet 사전 학습 가중치 사용
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        logger.info("Loaded MobileNetV3 with ImageNet pretrained weights")
    except Exception as e:
        logger.warning(f"Failed to load ImageNet weights: {e}, using random initialization")
        # 호환되지 않을 경우 랜덤 초기화 사용
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
    
    # Dropout 추가 (과적합 방지)
    dropout_rate = getattr(Config, 'DROPOUT', 0.0)
    if dropout_rate > 0:
        # Dropout을 추가한 분류기 레이어 구성
        model.classifier = nn.Sequential(
            model.classifier[0],  # 기존 첫 번째 레이어
            model.classifier[1],  # 기존 두 번째 레이어
            model.classifier[2],  # 기존 세 번째 레이어
            nn.Dropout(p=dropout_rate),  # Dropout 추가
            nn.Linear(in_features, num_classes)  # 최종 분류 레이어
        )
        logger.info(f"Dropout enabled: {dropout_rate}")
    else:
        model.classifier[3] = nn.Linear(in_features, num_classes)
    
    model = model.to(Config.DEVICE)
    
    logger.info(f"Model: MobileNetV3 Large with {num_classes} classes")
    if MULTI_CLASS:
        logger.info("Classes: 0=Normal, 1=Damaged, 2=Pollution")
    else:
        logger.info("Classes: 0=Normal, 1=Defect")

    # Loss 함수 설정: Focal Loss (불균형 데이터셋 대응)
    use_focal_loss = getattr(Config, 'USE_FOCAL_LOSS', True)
    focal_gamma = getattr(Config, 'FOCAL_LOSS_GAMMA', 2.0)  # 요구사항: gamma=2.0
    label_smoothing = getattr(Config, 'LABEL_SMOOTHING', 0.0)
    
    if use_focal_loss:
        # Focal Loss 사용 (불균형 데이터셋에 매우 효과적)
        # gamma=2.0: 어려운 샘플(틀린 예측)에 더 집중
        # alpha=class_weights: 클래스별 가중치로 불균형 보정
        criterion = FocalLoss(
            alpha=class_weights,  # 클래스별 가중치 (자동 계산 또는 수동 설정)
            gamma=focal_gamma,    # 요구사항: 2.0
            label_smoothing=label_smoothing
        )
        logger.info("=" * 60)
        logger.info("Loss Function: Focal Loss")
        logger.info(f"  gamma={focal_gamma} (어려운 샘플에 집중)")
        logger.info(f"  alpha={class_weights.cpu().tolist()} (클래스별 가중치)")
        if label_smoothing > 0:
            logger.info(f"  label_smoothing={label_smoothing}")
        logger.info("=" * 60)
    else:
        # CrossEntropyLoss 사용
        if label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            logger.info(f"CrossEntropyLoss with Label Smoothing: {label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("CrossEntropyLoss (standard) - Focal Loss 권장")
    
    # 2-Stage 전이 학습 전략 설정
    STAGE1_EPOCHS = 3  # Stage 1: Backbone Freeze, Classifier만 학습
    STAGE2_START_EPOCH = 4  # Stage 2: 전체 Fine-tuning 시작
    
    # Stage 1: Backbone Freeze (초반 가중치 파괴 방지)
    logger.info("=" * 60)
    logger.info(f"Stage 1: Freezing Backbone (Epoch 1-{STAGE1_EPOCHS})")
    logger.info("=" * 60)
    logger.info("Strategy: Freeze model.features (Backbone) to preserve ImageNet weights")
    logger.info("         Train only model.classifier (Head) to prevent early weight destruction")
    
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Backbone (model.features) frozen - requires_grad=False")
    logger.info("Only classifier (model.classifier) will be trained in Stage 1")
    logger.info("=" * 60)
    
    # Optimizer 설정: AdamW 사용
    # Stage 1 초기 학습률: 1e-3 (0.001)로 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,  # 초기 학습률 강제 설정 (Config.LR 대신 직접 지정)
        weight_decay=1e-2  # AdamW 기본 weight_decay 값
    )
    logger.info(f"Optimizer: AdamW with lr=0.001 (1e-3, forced for Stage 1), weight_decay=1e-2")
    
    # Learning Rate Scheduler: CosineAnnealingLR (안정적 30 epoch 학습)
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        # CosineAnnealingLR: 학습률이 부드럽게 감소 (Warm Restart 없음, 진동 방지)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.EPOCHS,  # 전체 epoch 수
            eta_min=1e-6  # 최소 학습률
        )
        logger.info(f"CosineAnnealingLR scheduler enabled: T_max={Config.EPOCHS}, eta_min=1e-6")
        logger.info("Scheduler changed to CosineAnnealingLR (No Warm Restarts) for stable 30 epochs")
    else:
        logger.info("Learning rate scheduler disabled")

    # 체크포인트 로드 (학습 재개)
    checkpoint_path = Config.RUNS_DIR / Config.CHECKPOINT_NAME
    start_epoch = 1
    best_acc = 0.0
    best_val_acc_for_early_stopping = 0.0
    early_stopping_counter = 0
    
    logger.info("=" * 60)
    logger.info("Checking for checkpoint to resume training")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info("=" * 60)
    
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state from checkpoint")
            start_epoch = checkpoint.get('epoch', 1) + 1  # 다음 에폭부터 시작
            best_acc = checkpoint.get('best_acc', 0.0)
            best_val_acc_for_early_stopping = checkpoint.get('best_val_acc_for_early_stopping', 0.0)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            logger.info(f"Resuming training from epoch {start_epoch}")
            logger.info(f"Previous best accuracy: {best_acc:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting from epoch 1")
            start_epoch = 1
    else:
        logger.info("No checkpoint found, starting from epoch 1")
    
    best_model_path = Config.RUNS_DIR / "model_classifier_best.pth"
    loss_csv_path = Config.RUNS_DIR / "training_loss_classifier.csv"
    
    # CSV 파일이 있으면 append, 없으면 새로 생성
    csv_exists = loss_csv_path.exists()
    csv_file = open(loss_csv_path, 'a' if csv_exists else 'w', encoding='utf-8', newline='')
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    # 데이터 로더 디버깅: 첫 번째 배치의 라벨 분포 확인
    logger.info("=" * 60)
    logger.info("DATA LOADER DEBUGGING: First Batch Label Distribution Check")
    logger.info("=" * 60)
    logger.info("Note: Please check this log message")
    logger.info("  Expected: Two classes should be mixed (e.g., Normal: 25, Defect: 7)")
    logger.info("  If only one class appears, there may be an issue with WeightedRandomSampler or labeling data")
    logger.info("=" * 60)
    first_batch_checked = False

    try:
        for epoch in range(start_epoch, Config.EPOCHS + 1):
            # Stage 2 전환: Epoch 4부터 Backbone Unfreeze (20만 장 데이터셋 최적화)
            if epoch == STAGE2_START_EPOCH:
                logger.info("=" * 60)
                logger.info(f"Stage 2: Unfreezing Backbone (Epoch {STAGE2_START_EPOCH}+)")
            logger.info("=" * 60)
            for param in model.features.parameters():
                param.requires_grad = True
            logger.info("Backbone (model.features) unfrozen - requires_grad=True")
            logger.info("Full fine-tuning enabled for all parameters")
            
            # Stage 2: 학습률을 1e-4 (0.0001)로 낮춤 (Fine-tuning 최적 학습률)
            stage2_lr = 1e-4  # ImageNet 가중치 보존하면서 미세 조정을 위한 학습률
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage2_lr
            logger.info(f"Learning rate reduced to {stage2_lr:.6f} (1e-4) for fine-tuning")
            logger.info("  Strategy: Lower LR prevents destroying pretrained weights")
            logger.info("=" * 60)
            
            model.train()
            total_loss = 0.0
            total_batches = len(train_loader)

            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"Epoch {epoch}/{Config.EPOCHS}",
                unit="batch",
                ncols=100,
                position=0,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for batch_idx, (imgs, labels) in pbar:
                # 첫 번째 배치 디버깅 (한 번만) - 라벨 분포 검증
                if not first_batch_checked and batch_idx == 0:
                    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
                    if MULTI_CLASS:
                        normal_count = int((labels_np == 0).sum())
                        damaged_count = int((labels_np == 1).sum())
                        pollution_count = int((labels_np == 2).sum())
                        total = len(labels_np)
                        
                        logger.info("")
                        logger.info("=" * 60)
                        logger.info("DEBUG: Batch 0 Label Distribution (Multi-Class)")
                        logger.info("=" * 60)
                        logger.info(f"  Normal:    {normal_count:2d} ({normal_count/total*100:5.1f}%)")
                        logger.info(f"  Damaged:   {damaged_count:2d} ({damaged_count/total*100:5.1f}%)")
                        logger.info(f"  Pollution: {pollution_count:2d} ({pollution_count/total*100:5.1f}%)")
                        logger.info(f"  Total:     {total:2d}")
                        logger.info("=" * 60)
                        
                        # 라벨 분포 검증: 한쪽으로 쏠리지 않았는지 확인
                        max_count = max(normal_count, damaged_count, pollution_count)
                        max_ratio = max_count / total
                        
                        if normal_count == total or damaged_count == total or pollution_count == total:
                            logger.error("=" * 60)
                            logger.error("CRITICAL ERROR: All labels are the same class")
                            logger.error("   Possible causes:")
                            logger.error("   1. WeightedRandomSampler not working correctly")
                            logger.error("   2. Labeling data problem (all same class)")
                            logger.error("   3. Data loader shuffle/sampler issue")
                            logger.error("   Action: Check WeightedRandomSampler and data labels")
                            logger.error("=" * 60)
                        elif max_ratio > 0.9:
                            logger.warning("=" * 60)
                            logger.warning(f"WARNING: Label distribution is highly imbalanced")
                            logger.warning(f"   One class occupies {max_ratio*100:.1f}% of the batch")
                            logger.warning(f"   Expected: More balanced distribution (e.g., 60-80% max)")
                            logger.warning(f"   Check: WeightedRandomSampler weights and data distribution")
                            logger.warning("=" * 60)
                        else:
                            logger.info("Label distribution looks balanced")
                            logger.info("=" * 60)
                    else:
                        normal_count = int((labels_np == 0).sum())
                        defect_count = int((labels_np == 1).sum())
                        total = len(labels_np)
                        
                        logger.info("")
                        logger.info("=" * 60)
                        logger.info("DEBUG: Batch 0 Label Distribution (Binary)")
                        logger.info("=" * 60)
                        logger.info(f"  Normal: {normal_count:2d} ({normal_count/total*100:5.1f}%)")
                        logger.info(f"  Defect: {defect_count:2d} ({defect_count/total*100:5.1f}%)")
                        logger.info(f"  Total:  {total:2d}")
                        logger.info("=" * 60)
                        
                        # 라벨 분포 검증
                        max_count = max(normal_count, defect_count)
                        max_ratio = max_count / total
                        
                        if normal_count == total or defect_count == total:
                            logger.error("=" * 60)
                            logger.error("CRITICAL ERROR: All labels are the same class")
                            logger.error("   Possible causes:")
                            logger.error("   1. WeightedRandomSampler not working correctly")
                            logger.error("   2. Labeling data problem (all same class)")
                            logger.error("   3. Data loader shuffle/sampler issue")
                            logger.error("   Action: Check WeightedRandomSampler and data labels")
                            logger.error("=" * 60)
                        elif max_ratio > 0.9:
                            logger.warning("=" * 60)
                            logger.warning(f"WARNING: Label distribution is highly imbalanced")
                            logger.warning(f"   One class occupies {max_ratio*100:.1f}% of the batch")
                            logger.warning(f"   Expected: More balanced distribution (e.g., Normal: 25 (78%), Defect: 7 (22%))")
                            logger.warning(f"   Check: WeightedRandomSampler weights and data distribution")
                            logger.warning("=" * 60)
                        else:
                            logger.info("Label distribution looks balanced")
                            logger.info("=" * 60)
                    
                    first_batch_checked = True
                imgs = imgs.to(Config.DEVICE, non_blocking=True)
                labels = torch.tensor(labels, device=Config.DEVICE)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss_so_far = total_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{avg_loss_so_far:.4f}'})

            pbar.close()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(Config.DEVICE, non_blocking=True)
                    labels = torch.tensor(labels, device=Config.DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            val_acc = correct / total if total > 0 else 0.0

            logger.info(f"[{epoch}/{Config.EPOCHS}] Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, Val ACC: {val_acc:.4f}")

            # CSV에 기록
            writer.writerow([epoch, f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}', f'{val_acc:.6f}'])
            csv_file.flush()  # 즉시 디스크에 기록

            # 스케줄러 업데이트
            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Stage 정보 로깅
            if epoch < STAGE2_START_EPOCH:
                stage_info = f"[Stage 1: Backbone Frozen]"
            else:
                stage_info = f"[Stage 2: Full Fine-tuning]"
            logger.info(f"{stage_info} Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 베스트 모델 저장 (Val ACC 기준)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Best model updated (ACC={best_acc:.4f}) -> {best_model_path}")
                early_stopping_counter = 0  # 개선되면 카운터 리셋
                best_val_acc_for_early_stopping = val_acc
            else:
                # Early Stopping 체크 (최소 20 epoch 이후에만 발동)
                # CosineAnnealingWarmRestarts의 학습률 리셋으로 인한 일시적 정확도 하락 방지
                MIN_EPOCHS_FOR_EARLY_STOPPING = 20
                if Config.USE_EARLY_STOPPING and epoch >= MIN_EPOCHS_FOR_EARLY_STOPPING:
                    improvement = val_acc - best_val_acc_for_early_stopping
                    if improvement < Config.EARLY_STOPPING_MIN_DELTA:
                        early_stopping_counter += 1
                        logger.info(f"Early stopping counter: {early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE} "
                                  f"(improvement: {improvement:.6f} < {Config.EARLY_STOPPING_MIN_DELTA})")
                        
                        if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                            logger.info(f"Early stopping triggered at epoch {epoch}. "
                                      f"Best Val ACC: {best_acc:.4f}")
                            logger.info(f"Training stopped early to prevent overfitting.")
                            break
                    else:
                        early_stopping_counter = 0  # 개선이 있으면 리셋
                elif Config.USE_EARLY_STOPPING and epoch < MIN_EPOCHS_FOR_EARLY_STOPPING:
                    # 최소 epoch 미만에서는 Early Stopping 비활성화
                    early_stopping_counter = 0  # 카운터 리셋
                    logger.debug(f"Early stopping disabled until epoch {MIN_EPOCHS_FOR_EARLY_STOPPING} (current: {epoch})")
            
            # 체크포인트 저장 (에폭 완료 후에만 저장)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_val_acc_for_early_stopping': best_val_acc_for_early_stopping,
                'early_stopping_counter': early_stopping_counter,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            logger.debug(f"Checkpoint saved at epoch {epoch}")
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Training interrupted by user (Ctrl+C)")
        logger.info("=" * 60)
        # 중단 시: 완료된 epoch만 저장 (현재 epoch이 완료되지 않았으므로 이전 epoch 저장)
        completed_epoch = epoch - 1
        if completed_epoch >= start_epoch:
            logger.info(f"Saving checkpoint for completed epoch {completed_epoch}...")
            checkpoint_data = {
                'epoch': completed_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_val_acc_for_early_stopping': best_val_acc_for_early_stopping,
                'early_stopping_counter': early_stopping_counter,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved. Training can be resumed from epoch {completed_epoch + 1}")
        else:
            logger.info("No epoch completed, no checkpoint saved.")
        csv_file.close()
        raise  # KeyboardInterrupt를 다시 발생시켜 정상 종료 처리

    csv_file.close()
    logger.info(f"Training finished. Best Val ACC={best_acc:.4f}, model={best_model_path}")
    return model

    logger.info(f"Training finished. Best Val ACC={best_acc:.4f}, model={best_model_path}")
    return model

def train():
    """
    설정된 학습 모드에 따라 적절한 학습 함수를 호출합니다.
    
    지원하는 학습 모드: 'normal_only', 'contrastive', 'classifier'
    """
    model = None
    try:
        if Config.TRAINING_MODE == 'normal_only':
            model = train_normal_only()
        elif Config.TRAINING_MODE == 'contrastive':
            model = train_contrastive()
        elif Config.TRAINING_MODE == 'classifier':
            model = train_classifier()
        else:
            raise ValueError(f"Unknown training mode: {Config.TRAINING_MODE}. "
                            f"Must be 'normal_only', 'contrastive', or 'classifier'")
        
        # 모델 저장
        if model is not None:
            weight_path = Config.RUNS_DIR / Config.MODEL_NAME
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Model saved to: {weight_path}")
    except KeyboardInterrupt:
        # Ctrl+C로 중단 시 현재까지 학습된 모델 저장
        logger.info("\n" + "=" * 60)
        logger.info("Training interrupted by user (Ctrl+C)")
        if model is not None:
            try:
                weight_path = Config.RUNS_DIR / Config.MODEL_NAME
                torch.save(model.state_dict(), weight_path)
                logger.info(f"Model saved to: {weight_path} (interrupted training)")
                logger.info("=" * 60)
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        raise  # KeyboardInterrupt를 다시 발생시켜 정상 종료 처리


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("AutoEncoder Training (Mixed Folder) Started")
        logger.info(f"Image Directories ({len(Config.MIXED_IMAGE_DIRS)}):")
        for i, img_dir in enumerate(Config.MIXED_IMAGE_DIRS, 1):
            logger.info(f"  [{i}] {img_dir}")
        if hasattr(Config, 'LABEL_FILES') and Config.LABEL_FILES:
            logger.info(f"Label Files ({len(Config.LABEL_FILES)}):")
            for i, label_file in enumerate(Config.LABEL_FILES, 1):
                logger.info(f"  [{i}] {label_file}")
        else:
            logger.info(f"Label File: {Config.LABEL_FILE}")
        logger.info(f"Training Mode: {Config.TRAINING_MODE}")
        logger.info("=" * 60)
        train()
        logger.info("=" * 60)
        logger.info("AutoEncoder Training Completed Successfully")
        logger.info("=" * 60)
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


