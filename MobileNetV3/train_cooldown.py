"""
Cool Down 학습 스크립트
Cool Down Training Script

목적 (Purpose):
    기존 학습된 모델을 로드하여 낮은 학습률로 안정적으로 추가 학습(fine-tuning)을 수행합니다.
    Loads a pre-trained model and performs fine-tuning with a lower learning rate for stability.
    Loss를 최소화하여 정확도를 85~95% 수준으로 향상시킵니다.
    Minimizes loss to improve accuracy to 85~95%.

사용 방법 (Usage):
    python train_cooldown.py
    
    - config.yaml 및 train_autoencoder_mixed.py의 Config를 사용합니다.
    - Uses Config from config.yaml and train_autoencoder_mixed.py.
"""

from pathlib import Path
import logging
from datetime import datetime
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

# 기존 학습 스크립트의 클래스와 함수들을 import
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_autoencoder_mixed import (
    Config, get_config, setup_logging, 
    MixedDataset, TransformWrapper, FocalLoss,
    load_labels_from_file
)

logger = setup_logging()


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    체크포인트에서 모델, optimizer, scheduler 상태를 로드합니다.
    Loads model, optimizer, and scheduler state from checkpoint.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로 / Path to checkpoint file
        model: 로드할 PyTorch 모델 / PyTorch model to load
        optimizer: (선택) optimizer / Optional optimizer
        scheduler: (선택) 학습률 스케줄러 / Optional LR scheduler
    
    Returns:
        start_epoch: 시작할 에폭 번호 (없으면 1) / Start epoch (1 if no checkpoint)
        best_acc: 최고 정확도 (없으면 0.0) / Best accuracy (0.0 if no checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found at {checkpoint_path}, starting from epoch 1")
        return 1, 0.0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ Loaded model from checkpoint")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"✓ Loaded optimizer state from checkpoint")
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Loaded scheduler state from checkpoint")
        
        start_epoch = checkpoint.get('epoch', 1) + 1  # 다음 에폭부터 시작
        best_acc = checkpoint.get('best_acc', 0.0)  # 최고 정확도 불러오기
        logger.info(f"✓ Resuming training from epoch {start_epoch}")
        logger.info(f"✓ Previous best accuracy: {best_acc:.4f}")
        return start_epoch, best_acc
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}, starting from epoch 1")
        return 1, 0.0


def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc, scheduler=None):
    """
    체크포인트를 저장합니다. / Saves checkpoint.
    모델, optimizer, scheduler 상태, 에폭 번호, 최고 정확도를 포함합니다.
    Includes model, optimizer, scheduler state, epoch, and best accuracy.
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved at epoch {epoch}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_best_model(model_path, num_classes, dropout_rate=0.0):
    """
    기존 학습된 최적 모델(best.pth)을 로드합니다.
    Loads pre-trained best model (best.pth).
    
    Args:
        model_path: 모델 가중치 파일 경로 (.pth) / Path to model weights (.pth)
        num_classes: 클래스 수 (2 또는 3) / Number of classes (2 or 3)
        dropout_rate: Dropout 비율 (학습 시 사용한 값과 동일해야 함) / Dropout rate (must match training)
    
    Returns:
        로드된 PyTorch 모델 / Loaded PyTorch model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # MobileNetV3 구조 생성
    try:
        model = models.mobilenet_v3_large(weights=None)  # 가중치는 best.pth에서 로드
        in_features = model.classifier[3].in_features
        logger.info("✓ Created MobileNetV3 structure")
    except Exception as e:
        raise RuntimeError(f"Failed to create MobileNetV3: {e}")
    
    # Dropout 설정 (기존 학습 시 사용한 구조와 동일하게)
    if dropout_rate > 0:
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        logger.info(f"✓ Dropout enabled: {dropout_rate}")
    else:
        model.classifier[3] = nn.Linear(in_features, num_classes)
    
    # 가중치 로드
    try:
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        logger.info(f"✓ Loaded weights from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    model = model.to(Config.DEVICE)
    return model


def train_cooldown():
    """Cool Down 학습: 낮은 학습률로 안정적으로 추가 학습"""
    logger.info("=" * 60)
    logger.info("Cool Down Training Started")
    logger.info("=" * 60)
    logger.info("Strategy: Low learning rate (1e-5) for stable fine-tuning")
    logger.info("Goal: Improve accuracy from 76% to 85~95%")
    logger.info("=" * 60)
    
    # ImageNet 정규화
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 데이터 증강 설정 (기존과 동일하게 적용)
    # 중요: 이미지가 이미 320x320이므로 리사이즈 제거
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
        
        # Random Erasing 추가 (설정된 경우)
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
    
    # 검증 데이터용 transform (증강 없음)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # 다중 클래스 모드 설정
    MULTI_CLASS = getattr(Config, 'MULTI_CLASS', False)
    num_classes = 3 if MULTI_CLASS else 2
    
    # 전체 데이터셋 생성 (기존 설정과 동일)
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
    
    # 클래스 가중치 계산 (기존과 동일)
    labels_np = np.array(full_dataset.labels)
    
    if MULTI_CLASS:
        normal_count = (labels_np == 0).sum()
        damaged_count = (labels_np == 1).sum()
        pollution_count = (labels_np == 2).sum()
        total_count = len(labels_np)
        
        max_count = max(normal_count, damaged_count, pollution_count)
        weight_normal = max_count / normal_count if normal_count > 0 else 1.0
        weight_damaged = max_count / damaged_count if damaged_count > 0 else 1.0
        weight_pollution = max_count / pollution_count if pollution_count > 0 else 1.0
        
        min_weight = min(weight_normal, weight_damaged, weight_pollution)
        weight_normal = weight_normal / min_weight
        weight_damaged = weight_damaged / min_weight
        weight_pollution = weight_pollution / min_weight
        
        auto_weights = [weight_normal, weight_damaged, weight_pollution]
        
        # 수동 오버라이드가 있으면 적용
        override = getattr(Config, 'CLASS_WEIGHT_OVERRIDE', None)
        if override and isinstance(override, (list, tuple)) and len(override) == 3:
            class_weights = torch.tensor(override, dtype=torch.float32, device=Config.DEVICE)
            logger.info(f"Class weights override: Normal={override[0]:.3f}, Damaged={override[1]:.3f}, Pollution={override[2]:.3f}")
        else:
            class_weights = torch.tensor(auto_weights, dtype=torch.float32, device=Config.DEVICE)
            logger.info(f"Class weights (auto): Normal={weight_normal:.3f}, Damaged={weight_damaged:.3f}, Pollution={weight_pollution:.3f}")
    else:
        normal_count = (labels_np == 0).sum()
        defect_count = (labels_np == 1).sum()
        max_count = max(normal_count, defect_count)
        weight_normal = max_count / normal_count if normal_count > 0 else 1.0
        weight_defect = max_count / defect_count if defect_count > 0 else 1.0
        min_weight = min(weight_normal, weight_defect)
        weight_normal = weight_normal / min_weight
        weight_defect = weight_defect / min_weight
        class_weights = torch.tensor([weight_normal, weight_defect], dtype=torch.float32, device=Config.DEVICE)
        logger.info(f"Class weights: Normal={weight_normal:.3f}, Defect={weight_defect:.3f}")
    
    # Train/Validation Split (기존과 동일한 시드 사용)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 기존과 동일한 시드
    )
    
    # 학습용과 검증용 데이터셋 생성
    train_subset = Subset(full_dataset, train_indices.indices)
    val_subset = Subset(full_dataset, val_indices.indices)
    
    train_dataset = TransformWrapper(train_subset, train_transform)
    val_dataset = TransformWrapper(val_subset, val_transform)
    
    logger.info(f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%), "
                f"Validation={len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    
    # WeightedRandomSampler 설정 (필수: 불균형 데이터셋 대응)
    # Cool Down 학습에서는 필수로 적용하여 배치 내 클래스 균형 유지
    train_labels = [full_dataset.labels[idx] for idx in train_indices.indices]
    train_labels_np = np.array(train_labels)
    
    if MULTI_CLASS:
        train_labels = [full_dataset.labels[idx] for idx in train_indices.indices]
        train_labels_np = np.array(train_labels)
        
        if MULTI_CLASS:
        class_counts = np.array([
            (train_labels_np == 0).sum(),
            (train_labels_np == 1).sum(),
            (train_labels_np == 2).sum()
        ])
        logger.info(f"Class counts (train): Normal={class_counts[0]}, Damaged={class_counts[1]}, Pollution={class_counts[2]}")
    else:
        class_counts = np.array([
            (train_labels_np == 0).sum(),
            (train_labels_np == 1).sum()
        ])
        logger.info(f"Class counts (train): Normal={class_counts[0]}, Defect={class_counts[1]}")
    
    # 각 클래스에 대한 가중치 계산 (역빈도)
    class_weights_for_sampler = 1.0 / (class_counts + 1e-6)  # 0으로 나누기 방지
    # 정규화
    class_weights_for_sampler = class_weights_for_sampler / class_weights_for_sampler.sum() * len(class_weights_for_sampler)
    sample_weights = class_weights_for_sampler[train_labels_np]
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True  # 소수 클래스를 더 자주 샘플링하기 위해 replacement=True
    )
    
    if MULTI_CLASS:
        logger.info(f"WeightedRandomSampler enabled (REQUIRED for imbalanced dataset):")
        logger.info(f"  Class weights: Normal={class_weights_for_sampler[0]:.3f}, "
                   f"Damaged={class_weights_for_sampler[1]:.3f}, "
                   f"Pollution={class_weights_for_sampler[2]:.3f}")
    else:
        logger.info(f"WeightedRandomSampler enabled (REQUIRED for imbalanced dataset):")
        logger.info(f"  Class weights: Normal={class_weights_for_sampler[0]:.3f}, "
                   f"Defect={class_weights_for_sampler[1]:.3f}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,  # WeightedRandomSampler 필수 적용 (불균형 데이터셋)
        shuffle=False,  # sampler 사용 시 shuffle은 False
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
    
    # 모델 로드 (기존 best 모델)
    best_model_path = Config.RUNS_DIR / "model_classifier_best.pth"
    dropout_rate = getattr(Config, 'DROPOUT', 0.0)
    
    logger.info("=" * 60)
    logger.info("Loading best model for cooldown training")
    logger.info(f"Model path: {best_model_path}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Dropout rate: {dropout_rate}")
    logger.info("=" * 60)
    
    model = load_best_model(best_model_path, num_classes, dropout_rate)
    
    logger.info(f"Model: MobileNetV3 Large with {num_classes} classes")
    if MULTI_CLASS:
        logger.info("Classes: 0=Normal, 1=Damaged, 2=Pollution")
    else:
        logger.info("Classes: 0=Normal, 1=Defect")
    
    # Loss 함수 설정 (기존과 동일)
    use_focal_loss = getattr(Config, 'USE_FOCAL_LOSS', True)
    focal_gamma = getattr(Config, 'FOCAL_LOSS_GAMMA', 2.0)
    label_smoothing = getattr(Config, 'LABEL_SMOOTHING', 0.0)
    
    if use_focal_loss:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        logger.info(f"Loss Function: Focal Loss (gamma={focal_gamma})")
    else:
        if label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Loss Function: CrossEntropyLoss")
    
    # Optimizer: AdamW with 매우 낮은 학습률 (1e-5)
    COOLDOWN_LR = 1e-5  # Cool Down 학습률 (고정)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COOLDOWN_LR,  # 매우 낮은 학습률로 고정
        weight_decay=1e-2
    )
    logger.info("=" * 60)
    logger.info(f"Optimizer: AdamW with lr={COOLDOWN_LR} (FIXED, no scheduler)")
    logger.info("Strategy: Low LR for stable fine-tuning without oscillation")
    logger.info("=" * 60)
    
    # Scheduler: ReduceLROnPlateau (선택적, 매우 보수적으로 설정)
    USE_SCHEDULER = False  # Cool Down에서는 스케줄러 비활성화 권장
    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # val_acc를 모니터링
            factor=0.5,  # 학습률을 절반으로 감소
            patience=5,  # 5 epoch 동안 개선 없으면 감소
            min_lr=1e-6,  # 최소 학습률
            verbose=True
        )
        logger.info("ReduceLROnPlateau scheduler enabled (conservative)")
    else:
        logger.info("Scheduler disabled (fixed LR for stability)")
    
    # Cool Down Epochs: 총 30 epoch 목표, 이미 8 epoch 학습 완료
    # 기존 학습 체크포인트에서 이어서 학습
    TOTAL_TARGET_EPOCHS = 30
    checkpoint_path = Config.RUNS_DIR / getattr(Config, 'CHECKPOINT_NAME', 'checkpoint.pth')
    
    # 체크포인트 로드 (기존 학습에서 이어서)
    logger.info("=" * 60)
    logger.info("Checking for existing checkpoint to resume training")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info("=" * 60)
    
    start_epoch, best_acc_from_checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    if start_epoch == 1:
        # 체크포인트가 없으면 best.pth 모델에서 시작
        logger.info("=" * 60)
        logger.info("No checkpoint found. Loading best model instead.")
        logger.info("=" * 60)
        best_model_path = Config.RUNS_DIR / "model_classifier_best.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
            logger.info(f"✓ Loaded best model from {best_model_path}")
            # best.pth 모델은 epoch 8에서 저장되었으므로, epoch 9부터 시작
            start_epoch = 9
            logger.info(f"✓ Starting from epoch {start_epoch} (assuming 8 epochs completed previously)")
        else:
            logger.warning(f"Best model not found at {best_model_path}")
            logger.warning("Starting from epoch 1 with best.pth weights (already loaded)")
            start_epoch = 9  # 기존 학습이 8 epoch까지 했으므로 9부터 시작
    else:
        logger.info(f"✓ Resuming from epoch {start_epoch}")
    
    # 총 목표 epoch까지 학습
    remaining_epochs = TOTAL_TARGET_EPOCHS - (start_epoch - 1)
    if remaining_epochs <= 0:
        logger.info(f"Target epochs ({TOTAL_TARGET_EPOCHS}) already reached!")
        logger.info(f"Checkpoint shows {start_epoch - 1} epochs completed.")
        return model
    
    logger.info(f"Target total epochs: {TOTAL_TARGET_EPOCHS}")
    logger.info(f"Starting epoch: {start_epoch}")
    logger.info(f"Remaining epochs: {remaining_epochs}")
    
    best_acc = best_acc_from_checkpoint
    final_model_path = Config.RUNS_DIR / "model_classifier_final_95.pth"
    loss_csv_path = Config.RUNS_DIR / "training_loss_cooldown.csv"
    
    # CSV 파일이 있으면 append, 없으면 새로 생성
    csv_exists = loss_csv_path.exists()
    csv_file = open(loss_csv_path, 'a' if csv_exists else 'w', encoding='utf-8', newline='')
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    logger.info("=" * 60)
    logger.info(f"Starting Cool Down Training (Epoch {start_epoch} to {TOTAL_TARGET_EPOCHS})")
    logger.info(f"Total remaining epochs: {remaining_epochs}")
    logger.info("=" * 60)
    
    for epoch in range(start_epoch, TOTAL_TARGET_EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        
        pbar = tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc=f"Cooldown Epoch {epoch}/{TOTAL_TARGET_EPOCHS}",
            unit="batch",
            ncols=100,
            position=0,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, (imgs, labels) in pbar:
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
        
        # 명확한 로그 출력 (Train Loss, Val Loss, Val ACC)
        logger.info("=" * 60)
        logger.info(f"Epoch [{epoch}/{TOTAL_TARGET_EPOCHS}]")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.6f}")
        logger.info(f"  Val ACC:    {val_acc:.6f} ({val_acc*100:.2f}%)")
        logger.info("=" * 60)
        
        # CSV에 기록
        writer.writerow([epoch, f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}', f'{val_acc:.6f}'])
        csv_file.flush()  # 즉시 디스크에 기록
        
        # Scheduler 업데이트 (ReduceLROnPlateau인 경우)
        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current LR: {current_lr:.8f}")
        
        # 베스트 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"✓ Best model updated (ACC={best_acc:.4f}) -> {final_model_path}")
        
        # 매 epoch마다 체크포인트 저장 (재개 가능하도록)
        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc, scheduler)
    
    csv_file.close()  # CSV 파일 닫기
    
    logger.info("=" * 60)
    logger.info("Cool Down Training Completed")
    logger.info(f"Total epochs completed: {TOTAL_TARGET_EPOCHS}")
    logger.info(f"Best Val ACC: {best_acc:.4f} ({best_acc*100:.2f}%)")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info(f"Loss history saved to: {loss_csv_path}")
    logger.info("=" * 60)
    
    return model


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Cool Down Training Script Started")
        logger.info("=" * 60)
        train_cooldown()
        logger.info("=" * 60)
        logger.info("Cool Down Training Completed Successfully")
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

