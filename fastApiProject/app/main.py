"""
================================================================================
Веб-сервис мультимодального нейросетевого анализа в диагностике онкологических
заболеваний (ЗОГМ)
================================================================================
"""

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
import torch

from inference import inference_engine

app = FastAPI(
    title="NeuroOnco AI — Мультимодальный анализ ЗОГМ",
    description="REST API поддержки принятия клинических решений на базе иерархического мультимодального трансформера (раздел 3.5)",
    version="2.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# Раздача XAI-картинок
app.mount("/xai", StaticFiles(directory=str(TMP_DIR)), name="xai")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

RESULTS_CACHE = {}

# ====================== ЭНДПОИНТЫ ======================
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print(" NeuroOnco AI Web Service запущен по HTTPS")
    print(" Доступ: https://127.0.0.1:8000")
    print(" Swagger UI: https://127.0.0.1:8000/docs")
    print("="*60 + "\n")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница веб-интерфейса"""
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze")
async def analyze(
    age: int = Form(55),
    sex: str = Form("M"),
    idh_status: str = Form("unknown"),
    mgmt_status: str = Form("unknown"),
    onep19q_status: str = Form("unknown"),
    timepoint: str = Form("baseline"),
    censored: int = Form(0),
    mri_t1: Optional[UploadFile] = File(None),
    mri_t1ce: Optional[UploadFile] = File(None),
    mri_t2: Optional[UploadFile] = File(None),
    mri_flair: Optional[UploadFile] = File(None),
    preprocessed_pt: Optional[UploadFile] = File(None),
):
    clinical_data = {
        "age": age,
        "sex": sex,
        "idh_status": idh_status,
        "mgmt_status": mgmt_status,
        "onep19q_status": onep19q_status,
        "timepoint": timepoint,
        "censored": censored,
        "diagnosis": "unknown"
    }

    upload_dir = TMP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir.mkdir(parents=True, exist_ok=True)

    mri_input = None
    modality_mask = None
    file_paths = []

    try:
        if preprocessed_pt and preprocessed_pt.filename and preprocessed_pt.size and preprocessed_pt.size > 0:
            # === Обработка готового .pt тензора ===
            pt_path = upload_dir / "input.pt"
            with open(pt_path, "wb") as f:
                shutil.copyfileobj(preprocessed_pt.file, f)

            if pt_path.stat().st_size < 1000:
                raise ValueError("Загруженный .pt файл слишком маленький")

            # Загрузка тензора
            try:
                tensor = torch.load(pt_path, map_location="cpu", weights_only=False)
            except:
                tensor = torch.load(pt_path, map_location="cpu")

            # Поиск тензора внутри (на случай сложных структур)
            def find_tensor(obj, depth=0):
                if depth > 5:
                    return None
                if isinstance(obj, torch.Tensor) and obj.dim() >= 3:
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        res = find_tensor(v, depth + 1)
                        if res is not None:
                            return res
                if isinstance(obj, (list, tuple)):
                    for v in obj:
                        res = find_tensor(v, depth + 1)
                        if res is not None:
                            return res
                return None

            found = find_tensor(tensor)
            if found is not None:
                tensor = found

            # Приведение к нужной форме [1, 4, 160, 160, 160]
            if tensor.dim() == 5:
                tensor = tensor[0]
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(4, 1, 1, 1)
            elif tensor.dim() == 4 and tensor.shape[0] != 4:
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(4, 1, 1, 1)
                else:
                    raise ValueError(f"Неверная форма тензора: {tensor.shape}")

            if tensor.shape[0] != 4:
                raise ValueError(f"Ожидалось 4 модальности, получено {tensor.shape[0]}")

            mri_input = tensor.unsqueeze(0)           # → [1, 4, 160, 160, 160]
            modality_mask = torch.ones(1, 4)

        else:
            # === Обработка отдельных NIfTI файлов ===
            modality_files = [mri_t1, mri_t1ce, mri_t2, mri_flair]
            for i, f in enumerate(modality_files):
                if f and f.filename:
                    path = upload_dir / f"mod_{i}.nii.gz"
                    with open(path, "wb") as out:
                        shutil.copyfileobj(f.file, out)
                    file_paths.append(str(path))
                else:
                    file_paths.append(None)

        # === Вызов predict ===
        result = inference_engine.predict(
            mri_input=mri_input,
            file_paths=file_paths,
            clinical_data=clinical_data,
            modality_mask=modality_mask
        )

        result["input"] = {
            "clinical": clinical_data,
            "modalities_uploaded": 4 if mri_input is not None else len([p for p in file_paths if p])
        }

        RESULTS_CACHE[result["request_id"]] = result
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.get("/report/{request_id}")
async def get_report(request_id: str):
    """Скачивание PDF-отчёта"""
    if request_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Отчёт не найден")

    result = RESULTS_CACHE[request_id]
    pdf_path = inference_engine.generate_pdf_report(result, result.get("input", {}).get("clinical", {}))
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"NeuroOnco_Report_{request_id}.pdf")


@app.get("/reports")
async def list_reports(limit: int = Query(20, le=100)):
    """Список последних отчётов (из кэша)"""
    reports = []
    for rid, res in list(RESULTS_CACHE.items())[-limit:]:
        reports.append({
            "request_id": rid,
            "timestamp": res.get("timestamp"),
            "tumor_probability": res.get("tumor", {}).get("probability"),
            "diagnosis": res.get("diagnosis", {}).get("prediction"),
            "grade": res.get("grade", {}).get("prediction"),
        })
    return {"count": len(reports), "reports": reports}


@app.get("/model/info")
async def model_info():
    """Информация о модели (для мониторинга и документации)"""
    return {
        "model_version": "2.2.0",
        "architecture": "Bidirectional Hierarchical Multimodal Transformer",
        "ensemble_size": len(inference_engine.models) if hasattr(inference_engine, 'models') else 1,
        "clinical_dim": getattr(inference_engine, 'clinical_dim', 21),
        "num_diagnosis_classes": getattr(inference_engine, 'num_diagnosis_classes', 5),
        "is_real_model": inference_engine.is_real_model,
        "supported_modalities": ["T1", "T1CE", "T2", "FLAIR"],
        "tasks": ["tumor_detection", "who_grade", "diagnosis", "survival_prediction"]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": inference_engine.is_real_model,
        "version": "2.2.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/example")
async def example():
    """Демо-результат для тестирования"""
    clinical = {"age": 62, "sex": "M", "idh_status": "mutant", "mgmt_status": "methylated",
                "onep19q_status": "no", "timepoint": "baseline", "censored": 0}
    return inference_engine.predict(clinical_data=clinical)


if __name__ == "__main__":
    # Пути к SSL-сертификатам (самоподписанные для разработки)
    # Для production замените на реальные сертификаты (Let's Encrypt и т.п.)
    CERTS_DIR = BASE_DIR / "certs"
    SSL_KEYFILE = CERTS_DIR / "key.pem"
    SSL_CERTFILE = CERTS_DIR / "cert.pem"

    # Если certs рядом с main.py нет — пробуем родительскую папку / абсолютный путь
    if not SSL_KEYFILE.exists():
        SSL_KEYFILE = Path(__file__).resolve().parent.parent / "certs" / "key.pem"
        SSL_CERTFILE = Path(__file__).resolve().parent.parent / "certs" / "cert.pem"

    use_ssl = SSL_KEYFILE.exists() and SSL_CERTFILE.exists()

    print("🚀 NeuroOnco AI Web Service запущен")
    if use_ssl:
        print(f"   HTTPS: https://localhost:8000")
        print(f"   Документация API: https://localhost:8000/docs")
        print(f"   Сертификат: {SSL_CERTFILE}")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            ssl_keyfile=str(SSL_KEYFILE),
            ssl_certfile=str(SSL_CERTFILE),
        )
    else:
        print("   ⚠️  SSL-сертификаты не найдены — запуск по HTTP")
        print("   Документация API: http://localhost:8000/docs")
        print("   Создайте папку certs/ с key.pem и cert.pem рядом с main.py")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
