# Workflow-CI — M. Rohid Rivaldi

Repo ini berisi konfigurasi MLflow Project dan GitHub Actions CI/CD untuk proyek klasifikasi sampah (Garbage Classification) — bagian dari submission akhir kelas **Membangun Sistem Machine Learning (MSML)** di Dicoding.

## 📂 Struktur Folder

```
Workflow-CI_M._Rohid_Rivaldi/
├── MLProject/
│   ├── MLProject          # File konfigurasi MLflow Project
│   ├── conda.yaml         # Environment dependencies
│   ├── modelling.py       # Script training model
│   └── garbage_preprocessing/  # Dataset (generated saat CI jalan)
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI pipeline (17 steps)
└── README.md
```

## 🔄 CI/CD Pipeline

Workflow CI ini otomatis berjalan setiap kali ada **push ke branch `main`**. Langkah-langkahnya:

1. Checkout repository
2. Set up Python 3.12.7
3. Check environment
4. Install dependencies
5. Download & preprocessing dataset (dari Kaggle)
6. Run MLflow Project (training model)
7. Get latest MLflow run ID
8. Install Python dependencies
9. Upload artifacts ke GitHub
10. Build Docker image dari MLflow model
11. Login ke Docker Hub
12. Tag Docker image
13. Push Docker image ke Docker Hub
14. *(Post: Logout Docker Hub)*
15. *(Post: Python teardown)*
16. *(Post: Checkout cleanup)*
17. *(Complete job)*

## 🔐 GitHub Secrets yang Diperlukan

Isi di **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|--------|------------|
| `DAGSHUB_TOKEN` | Token DagsHub untuk MLflow tracking |
| `DAGSHUB_USERNAME` | Username DagsHub (`rohidrivaldi`) |
| `KAGGLE_USERNAME` | Username Kaggle |
| `KAGGLE_KEY` | API Key Kaggle |
| `DOCKER_HUB_TOKEN` | Access token Docker Hub |

## 📊 Integrasi DagsHub

Eksperimen MLflow disimpan di: [DagsHub — Workflow-CI_M._Rohid_Rivaldi](https://dagshub.com/rohidrivaldi/Workflow-CI_M._Rohid_Rivaldi)

## 🐳 Docker Hub

Model yang sudah di-build tersedia di: `docker pull rohidrivaldi/garbage-classification:latest`

## 👤 Author

**M. Rohid Rivaldi** — rezoku (Dicoding)  
MSML 2026

# CI-CD Pipeline Setup Complete
