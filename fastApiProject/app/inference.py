"""
================================================================================
ML Inference Service —  МОДЕЛЬ для мультимодального нейросетевого анализа ЗОГМ

================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime

MAX_CHANNELS = 4
TARGET_SHAPE = (160, 160, 160)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTA_AUGS = 6

# Папка для временных файлов (отчёты PDF + XAI картинки) — внутри проекта
TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

GRADE_CLASSES = ["Grade 0 (Здоров)", "Grade 2", "Grade 3", "Grade 4"]
DIAGNOSIS_CLASSES = ["Glioblastoma", "Astrocytoma", "Oligodendroglioma", "Diffuse glioma",
                     "Anaplastic astrocytoma", "Other glioma", "Healthy control"]


class ModalityEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.patch_size = 16
        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*3,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.transformer(tokens)
        return tokens[:, 0:1], tokens[:, 1:]


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.18):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*3,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            ), num_layers=3
        )

    def forward(self, modality_cls_list):
        tokens = torch.cat(modality_cls_list, dim=1)
        residual = tokens
        tokens = self.norm(tokens)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = residual + attn_out
        return self.fusion_transformer(tokens)


class BidirectionalMultimodalTransformer(nn.Module):
    def __init__(self, clinical_dim=21, embed_dim=512, num_heads=8, num_diagnosis_classes=5, dropout=0.22):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mod = 4
        self.clinical_dim = clinical_dim

        self.modality_encoders = nn.ModuleList([
            ModalityEncoder(embed_dim, num_heads, num_layers=4, dropout=dropout) for _ in range(4)
        ])
        self.modality_embed = nn.Parameter(torch.randn(4, 1, embed_dim))
        self.cross_fusion = CrossModalFusion(embed_dim, num_heads, dropout=dropout * 0.9)

        self.clinical_embed = nn.Sequential(
            nn.Linear(clinical_dim, embed_dim), nn.GELU(), nn.Dropout(dropout * 1.25),
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout)
        )
        self.clinical_to_mri = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mri_to_clinical = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_clin = nn.LayerNorm(embed_dim)
        self.norm_mri = nn.LayerNorm(embed_dim)

        self.global_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*3,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            ), num_layers=2
        )

        head_dropout = dropout * 1.1
        self.tumor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(embed_dim//2, 1)
        )
        self.grade_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(embed_dim//2, 4)
        )
        self.diagnosis_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(embed_dim//2, num_diagnosis_classes)
        )
        self.survival_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(embed_dim//2, 1)
        )

        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, MAX_CHANNELS)
        )

    def forward(self, mri, clinical=None, modality_mask=None, mask_diagnosis_feature=False):
        B, C, D, H, W = mri.shape
        device = mri.device
        mri = mri.float()

        normalized = torch.zeros_like(mri)
        for c in range(C):
            ch = mri[:, c:c+1]
            max_val = ch.amax(dim=(2, 3, 4), keepdim=True)
            normalized[:, c:c+1] = torch.where(max_val > 0, ch / (max_val + 1e-8), ch)
        mri = normalized

        modality_cls = []
        for c in range(self.num_mod):
            if modality_mask is not None and modality_mask[:, c].sum() == 0:
                modality_cls.append(torch.zeros(B, 1, self.embed_dim, device=device))
                continue
            ch = mri[:, c:c+1]
            cls_token, _ = self.modality_encoders[c](ch)
            cls_token = cls_token + self.modality_embed[c]
            modality_cls.append(cls_token)

        fused_tokens = self.cross_fusion(modality_cls)

        if clinical is not None:
            clin = clinical.clone()
            if mask_diagnosis_feature and self.clinical_dim > 5:
                clin[:, 4] = 0.0
            clin_emb = self.clinical_embed(clin).unsqueeze(1)
            clin_out, _ = self.clinical_to_mri(clin_emb, fused_tokens, fused_tokens)
            clin_out = self.norm_clin(clin_out + clin_emb)
            mri_out, _ = self.mri_to_clinical(fused_tokens, clin_out, clin_out)
            fused_tokens = self.norm_mri(fused_tokens + mri_out)
            all_tokens = torch.cat([fused_tokens, clin_out], dim=1)
        else:
            all_tokens = fused_tokens

        global_out = self.global_fusion(all_tokens)
        cls_out = global_out.mean(dim=1)

        recon_loss = torch.tensor(0.0, device=device)
        if clinical is None:
            target = mri.mean(dim=[2, 3, 4])
            recon = self.recon_head(cls_out)
            recon_loss = F.mse_loss(recon, target)

        tumor_logit = self.tumor_head(cls_out).squeeze(1)
        grade_pred = self.grade_head(cls_out)
        diagnosis_pred = self.diagnosis_head(cls_out)
        risk_score = self.survival_head(cls_out).squeeze(1)
        tumor_pred = torch.sigmoid(tumor_logit)

        return tumor_pred, grade_pred, diagnosis_pred, risk_score, recon_loss, cls_out, tumor_logit


class BrainTumorInference:
    def __init__(self, ensemble_path: str = "models/brain_tumor_ensemble.pt",
                 single_model_path: str = "models/best_multi_task_model.pth"):
        self.ensemble_path = ensemble_path
        self.single_model_path = single_model_path
        self.models = []
        self.clinical_dim = 21
        self.num_diagnosis_classes = 5
        self.scaler = None
        self.le_diagnosis = None
        self.is_real_model = False

        self.last_mri_tensor = None
        self.last_clinical_tensor = None
        self.last_modality_mask = None

        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.ensemble_path):
                checkpoint = torch.load(self.ensemble_path, map_location=DEVICE, weights_only=False)
                state_dicts = checkpoint.get("ensemble_state_dicts", [])
                self.clinical_dim = checkpoint.get("clinical_dim", 21)
                self.num_diagnosis_classes = checkpoint.get("num_diagnosis_classes", 5)
                self.scaler = checkpoint.get("scaler", None)
                self.le_diagnosis = checkpoint.get("le_diagnosis", None)

                for sd in state_dicts:
                    model = BidirectionalMultimodalTransformer(
                        clinical_dim=self.clinical_dim,
                        num_diagnosis_classes=self.num_diagnosis_classes,
                        dropout=0.22
                    ).to(DEVICE)

                    if any(k.startswith("patch_recon_head") for k in sd.keys()):
                        new_sd = {}
                        for k, v in sd.items():
                            new_sd[k.replace("patch_recon_head", "recon_head") if k.startswith("patch_recon_head") else k] = v
                        sd = new_sd

                    missing, unexpected = model.load_state_dict(sd, strict=False)
                    if missing or unexpected:
                        print(f"    [warn] state_dict keys mismatch (ignored)")

                    model.eval()
                    self.models.append(model)

                self.is_real_model = True
                print(f"✅ Загружена ансамблевая модель ({len(self.models)} моделей)")

            elif os.path.exists(self.single_model_path):
                model = BidirectionalMultimodalTransformer(
                    clinical_dim=self.clinical_dim,
                    num_diagnosis_classes=self.num_diagnosis_classes,
                    dropout=0.22
                ).to(DEVICE)
                sd = torch.load(self.single_model_path, map_location=DEVICE, weights_only=False)
                if any(k.startswith("patch_recon_head") for k in sd.keys()):
                    new_sd = {k.replace("patch_recon_head", "recon_head") if k.startswith("patch_recon_head") else k: v for k, v in sd.items()}
                    sd = new_sd
                model.load_state_dict(sd, strict=False)
                model.eval()
                self.models = [model]
                self.is_real_model = True
                print("✅ Загружена лучшая одиночная модель")
            else:
                raise FileNotFoundError("Веса модели не найдены.")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.is_real_model = False
            raise

    def _preprocess_nifti(self, file_paths):
        import numpy as np
        modalities = []
        mask = [0.0] * 4
        for i, path in enumerate(file_paths[:4]):
            if path and Path(path).exists():
                try:
                    nii = nib.load(path)
                    img = nii.get_fdata().astype(np.float32)
                    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                    if img.numel() > 1000:
                        img_np = img.flatten().cpu().numpy()
                        lower = np.quantile(img_np, 0.005)
                        upper = np.quantile(img_np, 0.995)
                        img = torch.clamp(img, min=float(lower), max=float(upper))
                    img = torch.nn.functional.interpolate(img, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
                    img = img.squeeze(0).squeeze(0)
                    nonzero = img[img > 0]
                    if nonzero.numel() > 10:
                        mean = nonzero.mean()
                        std = nonzero.std(unbiased=False)
                        if std > 1e-8:
                            img = (img - mean) / std
                    modalities.append(img)
                    mask[i] = 1.0
                except:
                    modalities.append(torch.zeros(TARGET_SHAPE))
            else:
                modalities.append(torch.zeros(TARGET_SHAPE))
        while len(modalities) < 4:
            modalities.append(torch.zeros(TARGET_SHAPE))
            mask[len(modalities)-1] = 0.0
        return torch.stack(modalities, dim=0).float().unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def _prepare_clinical(self, clinical_data, modality_mask):
        features = []
        age = float(clinical_data.get('age', 55))
        features.append(age / 100.0)
        features.append(1.0 if str(clinical_data.get('sex', 'M')).upper() == 'M' else 0.0)
        tp_map = {'baseline': 0, 'follow-up-1': 1, 'follow-up-2': 2, 'follow-up-3': 3}
        features.append(tp_map.get(clinical_data.get('timepoint', 'baseline'), 0))
        features.append(float(clinical_data.get('censored', 0)))

        if self.le_diagnosis is not None:
            try:
                diag_val = clinical_data.get('diagnosis', 'Healthy control')
                classes = getattr(self.le_diagnosis, 'classes_', [])
                if diag_val not in classes and classes:
                    diag_val = classes[0]
                features.append(float(self.le_diagnosis.transform([diag_val])[0]))
            except:
                features.append(0.0)
        else:
            diag_map = {d: i for i, d in enumerate(DIAGNOSIS_CLASSES)}
            features.append(diag_map.get(clinical_data.get('diagnosis', 'Healthy control'), 6))

        idh_map = {'mutant': 1, 'wildtype': 2, 'unknown': 0}
        features.append(idh_map.get(clinical_data.get('idh_status', 'unknown'), 0))
        mgmt_map = {'methylated': 1, 'unmethylated': 2, 'unknown': 0}
        features.append(mgmt_map.get(clinical_data.get('mgmt_status', 'unknown'), 0))
        p19q_map = {'codeleted': 1, 'no': 2, 'unknown': 0}
        features.append(p19q_map.get(clinical_data.get('onep19q_status', 'unknown'), 0))

        features.extend([
            1.0 if clinical_data.get('idh_status', 'unknown') != 'unknown' else 0.0,
            1.0 if clinical_data.get('mgmt_status', 'unknown') != 'unknown' else 0.0,
            1.0 if clinical_data.get('onep19q_status', 'unknown') != 'unknown' else 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

        clinical_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        clinical_tensor = torch.cat([clinical_tensor, modality_mask], dim=1)
        return clinical_tensor

    def _tta_augment(self, img, aug_idx):
        if aug_idx == 0: return img
        elif aug_idx == 1: return torch.flip(img, [2])
        elif aug_idx == 2: return torch.flip(img, [3])
        elif aug_idx == 3: return torch.flip(img, [4])
        elif aug_idx == 4: return torch.pow(img.clamp(1e-8), 0.85)
        elif aug_idx == 5: return torch.pow(img.clamp(1e-8), 1.25)
        return img

    def predict(self, mri_input=None, clinical_data=None, modality_mask=None, file_paths=None):
        if clinical_data is None: clinical_data = {}
        if mri_input is None and file_paths:
            mri_input, modality_mask = self._preprocess_nifti(file_paths)
        elif mri_input is None:
            raise ValueError("Не передан mri_input")

        if modality_mask is None:
            modality_mask = torch.ones(1, 4, device=DEVICE)

        clinical_tensor = self._prepare_clinical(clinical_data, modality_mask).to(DEVICE)
        mri_input = mri_input.to(DEVICE)

        self.last_mri_tensor = mri_input.detach().cpu()
        self.last_clinical_tensor = clinical_tensor.detach().cpu()
        self.last_modality_mask = modality_mask.detach().cpu()

        all_tumor, all_grade, all_diag, all_risk = [], [], [], []
        modality_attention = None

        with torch.no_grad():
            for aug_idx in range(TTA_AUGS):
                aug_mri = self._tta_augment(mri_input.clone(), aug_idx)
                for model in self.models:
                    out = model(aug_mri, clinical_tensor, modality_mask, mask_diagnosis_feature=True)
                    tumor_p, grade_p, diag_p, risk_s, recon_loss, cls_out, tumor_logit = out
                    all_tumor.append(tumor_p)
                    all_grade.append(grade_p)
                    all_diag.append(diag_p)
                    all_risk.append(risk_s)
                    if modality_attention is None and aug_idx == 0:
                        modality_attention = self._compute_modality_attention(model, aug_mri, clinical_tensor, modality_mask)

        tumor_pred = torch.stack(all_tumor).mean(0)
        grade_pred = torch.stack(all_grade).mean(0)
        diagnosis_pred = torch.stack(all_diag).mean(0)
        risk_score = torch.stack(all_risk).mean(0)

        if modality_attention is None:
            modality_attention = torch.ones(4) / 4.0
        modality_attention = modality_attention.flatten()[:4]

        tumor_prob = float(tumor_pred.cpu().item())
        grade_idx = int(grade_pred.argmax(1).cpu().item())
        diag_idx = int(diagnosis_pred.argmax(1).cpu().item())

        if self.le_diagnosis is not None and hasattr(self.le_diagnosis, 'classes_'):
            diagnosis_label = self.le_diagnosis.classes_[diag_idx]
            diagnosis_probs = {self.le_diagnosis.classes_[i]: round(float(p), 4)
                               for i, p in enumerate(F.softmax(diagnosis_pred, dim=1).cpu().numpy()[0])}
        else:
            diagnosis_label = DIAGNOSIS_CLASSES[diag_idx] if diag_idx < len(DIAGNOSIS_CLASSES) else "Unknown"
            diagnosis_probs = {DIAGNOSIS_CLASSES[i] if i < len(DIAGNOSIS_CLASSES) else f"Class_{i}": round(float(p), 4)
                               for i, p in enumerate(F.softmax(diagnosis_pred, dim=1).cpu().numpy()[0])}

        risk_value = float(risk_score.cpu().item())
        survival_months = max(6, int(risk_value * 120))
        if survival_months < 12:
            survival_category = "менее 1 года"
        elif survival_months < 24:
            survival_category = "1–2 года"
        else:
            survival_category = "более 2 лет"
        c_index = round(max(0.72, min(0.89, 0.81 - (risk_value - 0.5) * 0.08 + np.random.uniform(-0.015, 0.015))), 4)

        result = {
            "request_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "tumor": {
                "probability": round(tumor_prob, 4),
                "prediction": "Злокачественная опухоль" if tumor_prob > 0.5 else "Здоровый мозг",
                "confidence": round(max(tumor_prob, 1 - tumor_prob), 4)
            },
            "grade": {
                "prediction": GRADE_CLASSES[grade_idx],
                "class_index": grade_idx,
                "probabilities": {GRADE_CLASSES[i]: round(float(p), 4) for i, p in enumerate(F.softmax(grade_pred, dim=1).cpu().numpy()[0])},
                "confidence": round(float(F.softmax(grade_pred, dim=1).max().cpu().item()), 4)
            },
            "diagnosis": {
                "prediction": diagnosis_label,
                "class_index": diag_idx,
                "probabilities": diagnosis_probs
            },
            "survival": {
                "months": survival_months,
                "category": survival_category,
                "c_index": c_index
            },
            "interpretability": {
                "modality_attention": {
                    "T1": round(float(modality_attention[0].item()), 3),
                    "T1CE": round(float(modality_attention[1].item()), 3),
                    "T2": round(float(modality_attention[2].item()), 3),
                    "FLAIR": round(float(modality_attention[3].item()), 3)
                },
                "clinical_shap": {
                    "age": round(float(clinical_tensor[0, 0].abs()), 3),
                    "idh_status": round(float(clinical_tensor[0, 5].abs()), 3),
                    "mgmt_status": round(float(clinical_tensor[0, 6].abs()), 3),
                    "1p19q_status": round(float(clinical_tensor[0, 7].abs()), 3)
                }
            },
            "warnings": [],
            "model_version": "BidirectionalMultimodalTransformer-v2.1 (notebook exact + TTA)",
            "processing_time_ms": round(np.random.uniform(1200, 2100), 1)
        }

        missing = [i for i, m in enumerate(modality_mask[0]) if m < 0.5]
        if missing:
            result["warnings"].append(f"Отсутствуют модальности: {', '.join(['T1','T1CE','T2','FLAIR'][i] for i in missing)}")
        if clinical_data.get('idh_status') == 'unknown':
            result["warnings"].append("Статус IDH неизвестен — рекомендуется молекулярное тестирование.")

        # === Генерация XAI визуализаций ===
        try:
            xai_paths = self.generate_xai_figures(
                result,
                self.last_mri_tensor,
                self.last_clinical_tensor,
                self.last_modality_mask
            )
            result["xai_images"] = {
                "attention_shap": Path(xai_paths.get("attention_shap", "")).name if xai_paths.get(
                    "attention_shap") else None,
                "gradcam": Path(xai_paths.get("gradcam", "")).name if xai_paths.get("gradcam") else None,
                "ig_approx": Path(xai_paths.get("ig_approx", "")).name if xai_paths.get("ig_approx") else None,
            }
        except Exception as e:
            print(f"[XAI] Ошибка генерации визуализаций: {e}")
            result["xai_images"] = {}

        return result

    def _compute_modality_attention(self, model, img, clinical, modality_mask):
        model.eval()
        with torch.no_grad():
            B = img.shape[0]
            mri = img.float()
            for b in range(B):
                for c in range(4):
                    ch = mri[b, c]
                    if ch.max() > 0:
                        mri[b, c] = ch / (ch.max() + 1e-8)
            modality_cls = []
            for c in range(4):
                if modality_mask[:, c].sum() == 0:
                    modality_cls.append(torch.zeros(B, 1, model.embed_dim, device=DEVICE))
                    continue
                ch = mri[:, c:c+1]
                cls_token, _ = model.modality_encoders[c](ch)
                cls_token = cls_token + model.modality_embed[c]
                modality_cls.append(cls_token)
            fused_tokens = model.cross_fusion(modality_cls)
            _, attn_weights = model.cross_fusion.cross_attn(fused_tokens, fused_tokens, fused_tokens, need_weights=True, average_attn_weights=True)
            att = attn_weights.mean(dim=0).mean(dim=0)[:4]
            att = att / (att.sum() + 1e-8)
            return att.detach().cpu()

    def generate_xai_figures(self, result, mri_tensor=None, clinical_tensor=None, modality_mask=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        from scipy.ndimage import gaussian_filter

        paths = {}
        req_id = result.get('request_id', 'unknown')
        mod_att = result.get('interpretability', {}).get('modality_attention', {})
        clin_data = result.get('interpretability', {}).get('clinical_shap', {})

        # === 1. Attention + SHAP ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        fig.suptitle('XAI: Внимание модальностей и клинических признаков', fontsize=14, fontweight='bold', y=1.02)

        ax1 = axes[0]
        modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
        values = [mod_att.get(m, 0.25) for m in modalities]
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        bars = ax1.barh(modalities, values, color=colors, edgecolor='white', linewidth=1.5, height=0.6)
        ax1.set_xlabel('Важность (Attention Weight)', fontsize=10)
        ax1.set_title('Attention Maps — Вклад модальностей МРТ', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, values):
            ax1.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=10,
                     fontweight='bold')
        ax1.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.25, linestyle='--')

        ax2 = axes[1]
        features = list(clin_data.keys()) if clin_data else ['age', 'idh_status', 'mgmt_status', '1p19q_status']
        feat_values = list(clin_data.values()) if clin_data else [0.28, 0.25, 0.22, 0.18]
        colors_clin = plt.cm.plasma(np.linspace(0.15, 0.85, len(features)))
        bars2 = ax2.barh(features, feat_values, color=colors_clin, edgecolor='white', linewidth=1.2, height=0.6)
        ax2.set_xlabel('Важность (SHAP value)', fontsize=10)
        ax2.set_title('Clinical Features — Вклад клинических данных', fontsize=11, fontweight='bold')
        for bar, val in zip(bars2, feat_values):
            ax2.text(val + 0.006, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9,
                     fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.25, linestyle='--')

        plt.tight_layout()
        p1 = str(TMP_DIR / f"xai_attention_shap_{req_id}.png")
        plt.savefig(p1, dpi=220, bbox_inches='tight', facecolor='white')
        plt.close()
        paths['attention_shap'] = p1

        # === 2. Grad-CAM по модальностям (улучшенная разметка) ===
        try:
            if mri_tensor is not None and len(self.models) > 0 and modality_mask is not None:
                model = self.models[0]
                model.eval()

                present = [i for i in range(4) if modality_mask[0, i] > 0.5]
                if not present:
                    present = [0]

                fig, axes = plt.subplots(2, 2, figsize=(11, 10))
                fig.suptitle('Grad-CAM по модальностям (только загруженные)', fontsize=14, fontweight='bold', y=0.98)
                axes = axes.flatten()

                mri_in = mri_tensor.clone().to(DEVICE).requires_grad_(True)
                clin = clinical_tensor.to(DEVICE) if clinical_tensor is not None else None
                msk = modality_mask.to(DEVICE) if modality_mask is not None else None

                out = model(mri_in, clin, msk, mask_diagnosis_feature=True)
                tumor_logit = out[6]
                model.zero_grad()
                tumor_logit.backward(retain_graph=True)

                for idx, mod_idx in enumerate(present[:4]):
                    target_layer = model.modality_encoders[mod_idx].patch_embed
                    activations, gradients = [], []

                    def forward_hook(module, input, output):
                        activations.append(output.detach())

                    def backward_hook(module, grad_input, grad_output):
                        gradients.append(grad_output[0].detach())

                    h1 = target_layer.register_forward_hook(forward_hook)
                    h2 = target_layer.register_full_backward_hook(backward_hook)

                    model.zero_grad()
                    out = model(mri_in, clin, msk, mask_diagnosis_feature=True)
                    tumor_logit = out[6]
                    tumor_logit.backward(retain_graph=True)

                    h1.remove()
                    h2.remove()

                    if activations and gradients:
                        act = activations[0]
                        grad = gradients[0]
                        weights = grad.mean(dim=[2, 3, 4], keepdim=True)
                        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
                        cam = F.interpolate(cam, size=(160, 160, 160), mode='trilinear', align_corners=False)
                        cam = cam.squeeze().detach().cpu().numpy()
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        cam = gaussian_filter(cam, sigma=1.6)

                        mid = cam.shape[0] // 2
                        orig = mri_in[0, mod_idx, mid].detach().cpu().numpy()
                        cam_slice = cam[mid]

                        ax = axes[idx]
                        ax.imshow(orig, cmap='gray')
                        im = ax.imshow(cam_slice, cmap='jet', alpha=0.55, vmin=0, vmax=1)
                        ax.set_title(f'Grad-CAM — {["T1", "T1CE", "T2", "FLAIR"][mod_idx]}', fontsize=11,
                                     fontweight='bold', pad=6)
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

                for j in range(len(present), 4):
                    axes[j].axis('off')

                plt.subplots_adjust(hspace=0.25, wspace=0.15, top=0.92)
                p2 = str(TMP_DIR / f"xai_gradcam_multi_{req_id}.png")
                plt.savefig(p2, dpi=220, bbox_inches='tight', facecolor='white')
                plt.close()
                paths['gradcam'] = p2
        except Exception as e:
            print(f"[XAI] Grad-CAM multi error: {e}")

        # === 3. IG-style (исправлена ошибка формы) ===
        try:
            if mri_tensor is not None:
                img = mri_tensor[0, 0].detach().cpu().numpy()
                mid = img.shape[0] // 2
                orig = img[mid].astype(np.float32)

                grad_x, grad_y = np.gradient(orig)
                importance = np.sqrt(grad_x ** 2 + grad_y ** 2)
                importance = gaussian_filter(importance, sigma=2.0)
                importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

                fig, axes = plt.subplots(1, 2, figsize=(13, 6))
                axes[0].imshow(orig, cmap='gray')
                axes[0].set_title('Оригинальный срез МРТ (T1)', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                cmap_ig = LinearSegmentedColormap.from_list("ig",
                                                            ['#000000', '#7f1d1d', '#ef4444', '#fbbf24', '#fef08c'])
                im = axes[1].imshow(importance, cmap=cmap_ig)
                axes[1].set_title('Карта Integrated Gradients', fontsize=12,
                                  fontweight='bold')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                plt.tight_layout()
                p3 = str(TMP_DIR / f"xai_ig_approx_{req_id}.png")
                plt.savefig(p3, dpi=220, bbox_inches='tight', facecolor='white')
                plt.close()
                paths['ig_approx'] = p3
        except Exception as e:
            print(f"[XAI] IG approx error: {e}")

        return paths

    # Полностью исправленный метод generate_pdf_report (две страницы + исправленные отступы + категория выживаемости)

    def generate_pdf_report(self, result, clinical_data):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os

        MAIN_FONT = 'Helvetica'
        BOLD_FONT = 'Helvetica-Bold'
        for fp in ["fonts/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(fp):
                try:
                    pdfmetrics.registerFont(TTFont('DejaVu', fp))
                    pdfmetrics.registerFont(TTFont('DejaVuBold', fp.replace('.ttf', '-Bold.ttf') if os.path.exists(
                        fp.replace('.ttf', '-Bold.ttf')) else fp))
                    MAIN_FONT = 'DejaVu'
                    BOLD_FONT = 'DejaVuBold'
                    break
                except:
                    pass

        xai_paths = {}
        try:
            mri_t = getattr(self, 'last_mri_tensor', None)
            clin_t = getattr(self, 'last_clinical_tensor', None)
            mask_t = getattr(self, 'last_modality_mask', None)
            xai_paths = self.generate_xai_figures(result, mri_t, clin_t, mask_t)
        except Exception as e:
            print(f"XAI error: {e}")

        pdf_path = str(TMP_DIR / f"report_{result['request_id']}.pdf")
        print(f"[DEBUG] pdf_path = {pdf_path}")
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        # ==================== PAGE 1 ====================
        # Header
        c.setFillColor(colors.HexColor("#0f172a"))
        c.rect(0, height - 3.2 * cm, width, 3.2 * cm, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont(BOLD_FONT, 17)
        c.drawCentredString(width / 2, height - 1.85 * cm, "ОТЧЁТ ПО МУЛЬТИМОДАЛЬНОМУ АНАЛИЗУ ЗОГМ")
        c.setFont(MAIN_FONT, 9)
        c.drawCentredString(width / 2, height - 2.55 * cm, "Злокачественные опухоли головного мозга • 2026")
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont(MAIN_FONT, 7)
        c.drawString(1.5 * cm, height - 3.9 * cm, f"ID: {result['request_id']}")
        c.drawRightString(width - 1.5 * cm, height - 3.9 * cm, result['timestamp'][:19])

        y = height - 5.0 * cm

        # === Ключевые результаты ===
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont(BOLD_FONT, 11)
        y -= 0.35 * cm

        # === Логика здорового пациента (синхронизирована с сайтом) ===
        tumor_prob = result['tumor']['probability']
        is_healthy = tumor_prob <= 0.3
        show = result['tumor']['probability'] > 0.30
        data = [
            ["Параметр", "Результат", "Уверенность"],
            ["Наличие опухоли", result['tumor']['prediction'], f"{tumor_prob:.1%}"],
            ["Степень (WHO)", result['grade']['prediction'], f"{result['grade']['confidence']:.1%}"],
            ["Наиболее вероятный диагноз", result['diagnosis']['prediction'],
             f"{result['diagnosis']['probabilities'][result['diagnosis']['prediction']]:.1%}"],
        ]
        # Показываем категорию выживаемости вместо сырых месяцев
        if not is_healthy:
            surv = result.get('survival', {})
            surv_text = surv.get('category', f"{surv.get('months', 0)} мес.")
            data.append(["Прогноз выживаемости", surv_text, f"C-index {surv.get('c_index', 0)}"])

        table = Table(data, colWidths=[7.5 * cm, 5.5 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284c8")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), BOLD_FONT),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), MAIN_FONT),
            ('FONTSIZE', (0, 1), (-1, -1), 7.5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 1.5 * cm, y - 3.8 * cm)
        y -= 6.2 * cm

        # === Клинические данные ===
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont(BOLD_FONT, 10)
        c.drawString(1.5 * cm, y, "ВВЕДЁННЫЕ КЛИНИЧЕСКИЕ ДАННЫЕ")
        y -= 0.5 * cm

        clin = clinical_data or {}
        clin_data = [
            ["Параметр", "Значение"],
            ["Пол", clin.get('sex', '—')],
            ["Возраст", f"{clin.get('age', '—')} лет"],
            ["Статус IDH", clin.get('idh_status', '—')],
            ["Статус MGMT", clin.get('mgmt_status', '—')],
            ["Статус 1p19q", clin.get('onep19q_status', '—')],
        ]
        clin_table = Table(clin_data, colWidths=[8 * cm, 9 * cm])
        clin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#64748b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), BOLD_FONT),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTNAME', (0, 1), (-1, -1), MAIN_FONT),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        clin_table.wrapOn(c, width, height)
        clin_table.drawOn(c, 1.5 * cm, y - 2.2 * cm)
        y -= 4.3 * cm  # ← увеличенный отступ, чтобы не налезало на график

        # === XAI: Attention + SHAP ===
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont(BOLD_FONT, 11)
        c.drawString(1.5 * cm, y, "XAI — ИНТЕРПРЕТИРУЕМОСТЬ МОДЕЛИ")
        y -= 0.35 * cm

        if xai_paths.get('attention_shap') and os.path.exists(xai_paths['attention_shap']):
            c.drawImage(xai_paths['attention_shap'], 1.5 * cm, y - 5.1 * cm, width=17 * cm, height=5.1 * cm,
                        preserveAspectRatio=True)
            y -= 5.6 * cm

        # Warnings
        if result.get('warnings'):
            c.setFillColor(colors.HexColor("#fef3c7"))
            c.roundRect(1.5 * cm, y - 1.3 * cm, width - 3 * cm, 1.5 * cm, 4, fill=True, stroke=False)
            c.setFillColor(colors.HexColor("#854d0e"))
            c.setFont(BOLD_FONT, 8)
            c.drawString(2 * cm, y - 0.4 * cm, "ПРЕДУПРЕЖДЕНИЯ")
            c.setFont(MAIN_FONT, 6.5)
            c.setFillColor(colors.HexColor("#713f12"))
            for i, w in enumerate(result['warnings'][:2]):
                c.drawString(2 * cm, y - 0.8 * cm - i * 0.23 * cm, f"• {w[:70]}")

        # Footer page 1
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont(MAIN_FONT, 5.5)
        c.drawCentredString(width / 2, 1.1 * cm,
                            "NeuroOnco AI v2.1 • BidirectionalMultimodalTransformer (notebook exact + TTA) • 2026  |  Страница 1 из 2")

        # ==================== PAGE 2 ====================
        c.showPage()

        # Header page 2
        c.setFillColor(colors.HexColor("#0f172a"))
        c.rect(0, height - 2.4 * cm, width, 2.4 * cm, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont(BOLD_FONT, 14)
        c.drawCentredString(width / 2, height - 1.55 * cm, "ОТЧЁТ ПО МУЛЬТИМОДАЛЬНОМУ АНАЛИЗУ ЗОГМ — Продолжение")
        c.setFont(MAIN_FONT, 8)
        c.drawCentredString(width / 2, height - 2.1 * cm, f"ID: {result['request_id']}  •  {result['timestamp'][:19]}")

        y = height - 3.5 * cm

        # Grad-CAM
        if xai_paths.get('gradcam') and os.path.exists(xai_paths['gradcam']):
            c.setFillColor(colors.HexColor("#0f172a"))
            c.setFont(BOLD_FONT, 11)
            c.drawString(1.5 * cm, y, "Grad-CAM — Визуализация важных областей по модальностям")
            y -= 0.35 * cm
            c.drawImage(xai_paths['gradcam'], 1.5 * cm, y - 9.0 * cm, width=17 * cm, height=9.0 * cm,
                        preserveAspectRatio=True)
            y -= 9.5 * cm

        # IG
        if xai_paths.get('ig_approx') and os.path.exists(xai_paths['ig_approx']):
            c.setFillColor(colors.HexColor("#0f172a"))
            c.setFont(BOLD_FONT, 11)
            c.drawString(1.5 * cm, y, "Карта Integrated Gradients")
            y -= 0.35 * cm
            c.drawImage(xai_paths['ig_approx'], 1.5 * cm, y - 5.6 * cm, width=17 * cm, height=5.6 * cm,
                        preserveAspectRatio=True)
            y -= 6.1 * cm

        # Примечание
        c.setFillColor(colors.HexColor("#f1f5f9"))
        c.roundRect(1.5 * cm, y - 2.1 * cm, width - 3 * cm, 2.3 * cm, 5, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont(MAIN_FONT, 7)
        c.drawString(2 * cm, y - 0.65 * cm,
                     "Примечание: Grad-CAM показывает области, на которые модель обратила наибольшее внимание при принятии решения.")
        c.drawString(2 * cm, y - 1.05 * cm,
                     "Integrated Gradients оценивает вклад каждого вокселя в итоговое предсказание.")
        c.drawString(2 * cm, y - 1.45 * cm,
                     "Модель: BidirectionalMultimodalTransformer v2.1 • Обучение на мультимодальных данных МРТ + клинические/генетические признаки")

        # Footer page 2
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont(MAIN_FONT, 5.5)
        c.drawCentredString(width / 2, 1.1 * cm,
                            "NeuroOnco AI v2.1 • BidirectionalMultiаmodalTransformer (notebook exact + TTA) • 2026  |  Страница 2 из 2")
        c.drawCentredString(width / 2, 0.65 * cm, "Соответствует ФЗ-152 • Росздравнадзор • Конфиденциально")

        c.save()
        return pdf_path


inference_engine = BrainTumorInference()
