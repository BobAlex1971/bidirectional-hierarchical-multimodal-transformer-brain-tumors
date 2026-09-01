# bidirectional-hierarchical-multimodal-transformer-brain-tumors
Двунаправленный иерархический мультимодальный трансформер для диагностики злокачественных опухолей головного мозга (обнаружение, WHO Grade, гистологический диагноз, прогноз выживаемости) с поддержкой неполных модальностей, ансамблем и интерпретируемостью.

## Веб-сервис

**Мультимодальный веб-сервис поддержки принятия клинических решений при диагностике злокачественных опухолей головного мозга (ЗОГМ)**

Система объединяет МРТ (T1 / T1CE / T2 / FLAIR) и клинико-генетические данные в едином иерархическом трансформере и выдаёт:

| Задача | Результат |
|--------|-----------|
| Обнаружение опухоли | Вероятность + бинарный вердикт |
| WHO Grade | Grade 0 / 2 / 3 / 4 |
| Гистологический диагноз | Glioblastoma, Astrocytoma, Oligodendroglioma и др. |
| Прогноз выживаемости | Категория + risk-score (C-index ≈ 0.88) |
| Объяснимость (XAI) | Attention, Grad-CAM, Integrated Gradients |
| Отчёт | PDF с результатами и визуализациями |

> **Важно:** система предназначена **только для научных и образовательных целей**. Это не медицинское изделие. Не используйте для клинических решений.

---

## Возможности

- Работа с **неполными данными** — можно загрузить от 1 до 4 МРТ-последовательностей
- Поддержка готового предобработанного `.pt`-тензора
- Ансамбль моделей + Test-Time Augmentation (TTA)
- Веб-интерфейс + REST API
- HTTPS (самоподписанные сертификаты для разработки)
- Docker / docker-compose из коробки

---

## Архитектура модели

**Bidirectional Hierarchical Multimodal Transformer**

```
МРТ (4 модальности)
    ↓
ModalityEncoder × 4  (3D Conv patch + Transformer)
    ↓
CrossModalFusion
    ↓
↔ Bidirectional Attention (MRI ↔ Clinical)
    ↓
Global Fusion Transformer
    ↓
┌────────────┬────────────┬──────────────┬─────────────┐
│ Tumor Head │ Grade Head │ Diagnosis    │ Survival    │
│ (BCE)      │ (Focal)    │ Head (CE)    │ Head (Cox)  │
└────────────┴────────────┴──────────────┴─────────────┘
```

- Клинические признаки: возраст, пол, IDH, MGMT, 1p/19q, timepoint, censoring + modality mask
- Обучение: multi-task + Uncertainty Weighted Loss + bagging (5 моделей) 

Код обучения — в `Learning.ipynb`.

---

## Структура проекта

```
fastApiProject/
├── app/
│   ├── main.py                 # FastAPI-приложение
│   ├── inference.py            # Загрузка модели + predict + XAI + PDF
│   ├── generate_certs.sh       # Генерация самоподписанных SSL-сертификатов
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html          # Веб-интерфейс
│   ├── static/                 # (при необходимости)
│   ├── models/                 # Веса: brain_tumor_ensemble.pt
│   ├── certs/                  # key.pem, cert.pem
│   ├── fonts/                  # DejaVu для PDF (кириллица)
│   └── tmp/                    # XAI-картинки и PDF-отчёты
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── Learning.ipynb              # Ноутбук обучения
└── requirements.txt            
```

> В Docker контекст сборки — корневая папка, а код копируется из `app/`.

---

## Быстрый старт

### 1. Локально (разработка)

```bash
cd app
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# (опционально) HTTPS
bash generate_certs.sh
# или
python generate_certs.py   # если openssl недоступен

# Положите веса модели:
#   models/brain_tumor_ensemble.pt
#   или models/best_multi_task_model.pth

python main.py
```

- HTTP:  http://127.0.0.1:8000  
- HTTPS: https://127.0.0.1:8000 (если есть `certs/key.pem` и `certs/cert.pem`)  
- Swagger: `/docs`

### 2. Docker

```bash
# Из корня проекта (где Dockerfile и docker-compose.yml)
docker compose up --build
```

Сервис будет доступен на http://localhost:8000.

Для GPU раскомментируйте секцию `deploy` в `docker-compose.yml`.

---

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Веб-интерфейс |
| `POST` | `/analyze` | Анализ (multipart: МРТ + клинические поля) |
| `GET` | `/example` | Демо-результат без файлов |
| `GET` | `/report/{request_id}` | Скачать PDF-отчёт |
| `GET` | `/reports` | Список последних отчётов (кэш) |
| `GET` | `/model/info` | Информация о загруженной модели |
| `GET` | `/health` | Health-check |

### Пример запроса `/analyze`

Поля формы:

- `age`, `sex`, `idh_status`, `mgmt_status`, `onep19q_status`, `timepoint`, `censored`
- файлы: `mri_t1`, `mri_t1ce`, `mri_t2`, `mri_flair` (`.nii` / `.nii.gz`)
- опционально: `preprocessed_pt` (готовый тензор)

Ответ — JSON с предсказаниями, interpretability и путями к XAI-изображениям.

---
## Данные

Веса модели обучены на объединённом наборе публичных датасетов.  
**Исходные данные в репозиторий не входят** — получайте их самостоятельно по правилам каждого источника.

| Датасет | Описание | Ссылка |
|---------|----------|--------|
| **UPENN-GBM** | [TCIA](https://www.cancerimagingarchive.net/collection/upenn-gbm/) · [DOI: 10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49) |
| **UCSF-PDGM** | [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) · [DOI: 10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37) |
| **ReMIND** | [TCIA](https://www.cancerimagingarchive.net/collection/remind/) · [DOI: 10.7937/3RAG-D070](https://doi.org/10.7937/3RAG-D070) |
| **UCSD-PTGBM** | [TCIA](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/) · [DOI: 10.7937/fwv2-dt74](https://doi.org/10.7937/fwv2-dt74) |
| **IXI** |  [brain-development.org](https://brain-development.org/ixi-dataset/) |
| **OASIS-1** | [OASIS](https://sites.wustl.edu/oasisbrains/home/oasis-1/) · [DOI: 10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498) |

Перед использованием ознакомьтесь с лицензией и условиями доступа каждого датасета.

## Веса модели

Скачайте обученный ансамбль из [Releases](https://github.com/<user>/<repo>/releases):

- `brain_tumor_ensemble.pt` — 5 моделей + метаданные (clinical_dim, le_diagnosis и т.д.)

Положите файл в `app/models/`.

---

## Требования к весам

При старте `inference.py` ищет:

1. `models/brain_tumor_ensemble.pt` (предпочтительно)  
2. иначе `models/best_multi_task_model.pth`

Формат ансамбля:

```python
{
  "ensemble_state_dicts": [...],
  "clinical_dim": 21,
  "num_diagnosis_classes": 5,
  "scaler": ...,
  "le_diagnosis": ...
}
```

Без весов сервис **не запустится** (`is_real_model = False` → исключение при импорте).

---

## SSL (разработка)

```bash
cd app
bash generate_certs.sh
```

Создаёт `certs/key.pem` и `certs/cert.pem`.  
`main.py` автоматически включает HTTPS, если файлы найдены.

---

## Зависимости

Основные (см. `app/requirements.txt`):

- FastAPI, Uvicorn, Jinja2, python-multipart  
- PyTorch, nibabel, numpy, scikit-learn  
- ReportLab / fpdf2 (PDF), matplotlib (XAI)  
- pandas, pydantic, python-dotenv  

Системные пакеты (уже в Dockerfile): `libgl1`, `libglib2.0-0`, шрифты DejaVu и др.

---

## Дисклеймер

- Программное обеспечение предоставляется **as is** исключительно в исследовательских и учебных целях.
- Не является зарегистрированным медицинским изделием.
- Не предназначено для постановки диагноза и принятия клинических решений.
- При обработке данных пациентов соблюдайте действующие требования по защите персональных данных.
- Авторы не несут ответственности за последствия использования системы в клинической практике.

---

## Лицензия

Рекомендуется **MIT** для кода и весов модели.  
Исходные датасеты (UPENN-GBM, UCSF-PDGM, ReMIND, IXI и др.) **не распространяются** вместе с репозиторием — их лицензии необходимо соблюдать отдельно.

