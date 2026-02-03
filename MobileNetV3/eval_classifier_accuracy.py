"""
분류기 정확도 평가 스크립트
Classifier Accuracy Evaluation Script

목적 (Purpose):
    학습된 MobileNetV3 분류기 모델의 성능을 라벨 데이터로 평가합니다.
    Evaluates trained MobileNetV3 classifier performance on labeled data.
    ACC, F1-Score, Precision, Recall 등 메트릭을 계산합니다.
    Computes metrics: accuracy, F1-score, precision, recall.

사용 방법 (Usage):
    python eval_classifier_accuracy.py --model runs/model_classifier_best.pth
    
    추가 옵션 / Additional options:
    --config: config.yaml 경로 / Path to config.yaml
    --image-size: 입력 이미지 크기 (기본: 320) / Input image size (default: 320)
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms
from tqdm import tqdm

from train_autoencoder_mixed import MixedDataset, Config, get_config, ConditionalResize, TransformWrapper


def load_model(model_path: Path, device: torch.device, image_size: int = 320, num_classes: int = 2, dropout_rate: float = 0.0):
    """
    분류기 모델을 로드합니다. 2클래스/3클래스, Dropout 포함 구조를 지원합니다.
    Loads classifier model. Supports 2-class/3-class and Dropout-inclusive structure.
    """
    try:
        # 구조를 학습 시점과 동일하게 구성 (Dropout 포함)
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        if dropout_rate > 0:
            model.classifier = nn.Sequential(
                model.classifier[0],
                model.classifier[1],
                model.classifier[2],
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[3] = nn.Linear(in_features, num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"모델 로드 실패: {e}")


def evaluate_model(model, data_loader, device, num_classes: int = 2, class_names: List[str] = None):
    """
    모델 평가: 정확도, F1, Precision, Recall 계산
    Evaluates model: computes accuracy, F1, precision, recall.
    
    2클래스: TP, TN, FP, FN 기반 / 2-class: based on TP/TN/FP/FN
    3클래스: 클래스별 Precision/Recall/F1, 매크로 평균 / 3-class: per-class metrics, macro average
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="평가 중"):
            imgs = imgs.to(device, non_blocking=True)
            
            # 라벨 처리 개선: 이미 tensor면 그대로 사용, 아니면 변환
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 메트릭 계산
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 라벨 값 검증 (디버깅용)
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    if len(unique_labels) > num_classes or max(unique_labels) >= num_classes:
        print(f"⚠️ 경고: 라벨 값이 유효 범위를 벗어남. 라벨 범위: {unique_labels}, 예상 클래스 수: {num_classes}")
    if len(unique_preds) > num_classes or max(unique_preds) >= num_classes:
        print(f"⚠️ 경고: 예측 값이 유효 범위를 벗어남. 예측 범위: {unique_preds}, 예상 클래스 수: {num_classes}")
    
    # 클래스별 분포 출력 (디버깅용)
    print(f"\n[디버깅] 실제 라벨 분포:")
    for i in range(num_classes):
        count = (all_labels == i).sum()
        print(f"  클래스 {i} ({class_names[i] if class_names and i < len(class_names) else 'unknown'}): {count}개 ({count/len(all_labels)*100:.2f}%)")
    print(f"[디버깅] 예측 라벨 분포:")
    for i in range(num_classes):
        count = (all_preds == i).sum()
        print(f"  클래스 {i} ({class_names[i] if class_names and i < len(class_names) else 'unknown'}): {count}개 ({count/len(all_preds)*100:.2f}%)")
    
    acc = (all_preds == all_labels).mean() if len(all_labels) > 0 else 0.0
    
    if num_classes == 2:
        tp = int(((all_preds == 1) & (all_labels == 1)).sum())  # 불량을 불량으로 예측
        tn = int(((all_preds == 0) & (all_labels == 0)).sum())  # 정상을 정상으로 예측
        fp = int(((all_preds == 1) & (all_labels == 0)).sum())  # 정상을 불량으로 예측
        fn = int(((all_preds == 0) & (all_labels == 1)).sum())  # 불량을 정상으로 예측
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': len(all_labels),
            'num_classes': num_classes
        }
    else:
        # 클래스별 메트릭
        per_class = []
        class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        for c in range(num_classes):
            tp = int(((all_preds == c) & (all_labels == c)).sum())
            fp = int(((all_preds == c) & (all_labels != c)).sum())
            fn = int(((all_preds != c) & (all_labels == c)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class.append({
                'class': class_names[c] if c < len(class_names) else str(c),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        macro_precision = np.mean([c['precision'] for c in per_class])
        macro_recall = np.mean([c['recall'] for c in per_class])
        macro_f1 = np.mean([c['f1'] for c in per_class])
        
        # 혼동 행렬
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_labels, all_preds):
            confusion[t, p] += 1
        
        return {
            'acc': acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class': per_class,
            'confusion': confusion,
            'total': len(all_labels),
            'num_classes': num_classes,
            'class_names': class_names
        }


def main():
    parser = argparse.ArgumentParser(description="분류기 정확도 평가 (라벨 기반)")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/model_classifier_best.pth",
        help="평가할 모델 파일 경로",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config.yaml 경로 (None이면 기본 경로 사용)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="이미지 크기 (기본값: 320)",
    )
    args = parser.parse_args()
    
    # Config 로드
    config = get_config(args.config)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 클래스 개수 / 이름 설정
    num_classes = 3 if getattr(config, 'MULTI_CLASS', False) else 2
    class_names = ['normal', 'Damaged', 'Pollution'] if num_classes == 3 else ['normal', 'defect']
    
    # 모델 로드
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # 상대 경로인 경우 RUNS_DIR과 결합
        # "runs/model.pth" -> "model.pth"로 변환 (RUNS_DIR에 이미 runs가 포함됨)
        model_str = str(model_path)
        # "runs/" 또는 "runs\"로 시작하면 제거
        if model_str.startswith("runs/") or model_str.startswith("runs\\"):
            model_str = model_str[5:]  # "runs/" 또는 "runs\" 제거
        model_path = config.RUNS_DIR / model_str
    else:
        # 절대 경로인 경우 그대로 사용
        model_path = model_path.resolve()
    
    print(f"\n[모델 경로 정보]")
    print(f"  입력 경로: {args.model}")
    print(f"  RUNS_DIR: {config.RUNS_DIR}")
    print(f"  최종 모델 경로: {model_path}")
    print(f"  존재 여부: {'✓ 존재' if model_path.exists() else '✗ 없음'}")
    
    if not model_path.exists():
        # 가능한 경로들 제시
        possible_paths = [
            config.RUNS_DIR / "model_classifier_best.pth",
            config.RUNS_DIR / "autoencoder_bilateral.pth",
        ]
        existing_models = list(config.RUNS_DIR.glob("*.pth"))
        
        error_msg = f"모델 파일을 찾을 수 없습니다: {model_path}\n"
        error_msg += f"RUNS_DIR: {config.RUNS_DIR}\n"
        if existing_models:
            error_msg += f"\n찾은 모델 파일들:\n"
            for m in existing_models[:10]:  # 최대 10개만 표시
                error_msg += f"  - {m.name}\n"
        else:
            error_msg += f"\n{config.RUNS_DIR} 폴더에 .pth 파일이 없습니다.\n"
        raise FileNotFoundError(error_msg)
    
    print(f"모델 로드: {model_path} (num_classes={num_classes})")
    dropout_rate = getattr(config, 'DROPOUT', 0.0)
    model = load_model(model_path, device, args.image_size, num_classes=num_classes, dropout_rate=dropout_rate)
    
    # 데이터셋 생성 (Validation 데이터만 사용)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 이미 320x320인 이미지는 리사이즈 건너뛰기 (품질 보존)
    transform = transforms.Compose([
        ConditionalResize(args.image_size),  # 320x320이면 건너뛰기
        transforms.ToTensor(),
        normalize,
    ])
    
    # 경로 정보 출력 (디버깅용)
    print(f"\n[경로 정보]")
    if hasattr(config, 'MIXED_IMAGE_DIRS'):
        print(f"이미지 디렉토리 ({len(config.MIXED_IMAGE_DIRS)}개):")
        for i, img_dir in enumerate(config.MIXED_IMAGE_DIRS, 1):
            exists = "✓" if Path(img_dir).exists() else "✗"
            print(f"  [{i}] {exists} {img_dir}")
    else:
        print(f"이미지 디렉토리: {getattr(config, 'MIXED_IMAGE_DIR', 'N/A')}")
    
    if hasattr(config, 'LABEL_FILES') and config.LABEL_FILES:
        print(f"라벨 파일/폴더 ({len(config.LABEL_FILES)}개):")
        for i, label_file in enumerate(config.LABEL_FILES, 1):
            exists = "✓" if Path(label_file).exists() else "✗"
            print(f"  [{i}] {exists} {label_file}")
    elif hasattr(config, 'LABEL_FILE') and config.LABEL_FILE:
        exists = "✓" if Path(config.LABEL_FILE).exists() else "✗"
        print(f"라벨 파일: {exists} {config.LABEL_FILE}")
    else:
        print("라벨 파일: 없음 (파일명 패턴으로 분류)")
    print()
    
    # 전체 데이터셋 생성 (학습 시와 동일한 설정)
    # transform은 나중에 적용하므로 None으로 설정 (학습 시와 동일)
    full_dataset = MixedDataset(
        image_dir=config.MIXED_IMAGE_DIRS,
        transform=None,  # 학습 시와 동일하게 None으로 설정
        use_bilateral=config.USE_BILATERAL,
        label_file=config.LABEL_FILES if hasattr(config, 'LABEL_FILES') and config.LABEL_FILES else config.LABEL_FILE,
        normal_keywords=config.NORMAL_KEYWORDS,
        defect_keywords=config.DEFECT_KEYWORDS,
        custom_pattern=config.CUSTOM_PATTERN,
        training_mode='classifier',
        multi_class=getattr(config, 'MULTI_CLASS', False)
    )
    
    # Train/Validation Split (학습 시와 동일한 시드 사용)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 학습 시와 동일한 방식으로 split (인덱스 기반)
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Subset 생성 후 transform 적용 (학습 시와 동일한 방식)
    val_subset = Subset(full_dataset, val_indices.indices)
    val_dataset = TransformWrapper(val_subset, transform)
    
    print(f"전체 데이터: {total_size}개")
    print(f"검증 데이터: {len(val_dataset)}개 ({len(val_dataset)/total_size*100:.1f}%)")
    
    # DataLoader 생성
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 평가 실행
    print("\n" + "=" * 60)
    print("분류기 정확도 평가 시작")
    print("=" * 60)
    
    metrics = evaluate_model(model, val_loader, device, num_classes=num_classes, class_names=class_names)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"정확도 (ACC):     {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    
    if num_classes == 2:
        print(f"정밀도 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"재현율 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:         {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"\n혼동 행렬:")
        print(f"  TP (True Positive):  {metrics['tp']:5d} - 불량을 불량으로 정확히 예측")
        print(f"  TN (True Negative):  {metrics['tn']:5d} - 정상을 정상으로 정확히 예측")
        print(f"  FP (False Positive): {metrics['fp']:5d} - 정상을 불량으로 잘못 예측")
        print(f"  FN (False Negative): {metrics['fn']:5d} - 불량을 정상으로 잘못 예측")
        print(f"  총 데이터:           {metrics['total']:5d}")
    else:
        print(f"매크로 Precision:  {metrics['macro_precision']:.4f} ({metrics['macro_precision']*100:.2f}%)")
        print(f"매크로 Recall:     {metrics['macro_recall']:.4f} ({metrics['macro_recall']*100:.2f}%)")
        print(f"매크로 F1:         {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
        print("\n클래스별 지표:")
        for c in metrics['per_class']:
            print(f"  [{c['class']}] P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f} | TP={c['tp']} FP={c['fp']} FN={c['fn']}")
        
        print("\n혼동 행렬 (row=true, col=pred):")
        cnf = metrics['confusion']
        header = "       " + " ".join([f"{n:>10}" for n in class_names])
        print(header)
        for i, row in enumerate(cnf):
            name = class_names[i] if i < len(class_names) else str(i)
            row_str = " ".join([f"{v:10d}" for v in row])
            print(f"{name:>6} {row_str}")
        print(f"\n총 데이터: {metrics['total']}")
    
    print("=" * 60)
    
    # 목표 정확도 확인
    target_acc = 0.90
    if metrics['acc'] >= target_acc:
        print(f"✅ 목표 정확도 {target_acc*100:.0f}% 달성!")
    else:
        print(f"⚠️ 목표 정확도 {target_acc*100:.0f}% 미달 (현재: {metrics['acc']*100:.2f}%)")
        print("   더 많은 데이터나 하이퍼파라미터 조정이 필요할 수 있습니다.")


if __name__ == "__main__":
    main()

