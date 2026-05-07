<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/FAISS-Vektör%20Arama-00ADD8?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-FF6F00?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge" alt="Lisans">
</p>

<h1 align="center">📜 RAG Sözleşme Avukatı</h1>

<p align="center">
  <b>Yapay Zeka Destekli Sözleşme Analiz ve Sorgulama Sistemi</b><br>
  <i>PDF sözleşmelerinizi yükleyin, sorularınızı sorun — yapay zeka ile hukuki analiz yapın.</i>
</p>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Mimari ve Çalışma Prensibi](#-mimari-ve-çalışma-prensibi)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [RAG Nedir?](#-rag-nedir)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

**RAG Sözleşme Avukatı**, PDF formatındaki sözleşme belgelerini analiz edebilen ve kullanıcı sorularına yapay zeka destekli yanıtlar verebilen bir **Retrieval-Augmented Generation (RAG)** sistemidir.

Bu sistem sayesinde:
- 📄 PDF sözleşmelerinizi otomatik olarak işleyebilirsiniz
- 🔍 Sözleşme içeriğine yönelik sorular sorabilirsiniz
- 🤖 GPT-3.5 Turbo ile doğru ve kısa yanıtlar alabilirsiniz
- ⚡ FAISS vektör veritabanı ile hızlı benzerlik araması yapabilirsiniz

> **Örnek sorular:** *"Sözleşmenin tarafları kimlerdir?"*, *"Ücret ne kadar?"*, *"Sözleşme ne zaman sona eriyor?"*

---

## 🏗 Mimari ve Çalışma Prensibi

```
┌──────────────────────────────────────────────────────────────┐
│                     RAG Sözleşme Avukatı                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. VERİTABANI OLUŞTURMA (build_vector_database.py)         │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐  │
│  │  PDF     │───▶│  Metin   │───▶│ Chunking │───▶│ FAISS  │  │
│  │ Dosyası  │    │ Çıkarma  │    │ (500 kar)│    │  Index │  │
│  └─────────┘    └──────────┘    └──────────┘    └────────┘  │
│                                       │                      │
│                              ┌────────▼────────┐             │
│                              │   Embedding     │             │
│                              │  (BAAI/bge-m3)  │             │
│                              └─────────────────┘             │
│                                                              │
│  2. SORU-CEVAP (ask.question.py)                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐  │
│  │ Kullanıcı│───▶│ Embedding│───▶│  FAISS   │───▶│  GPT   │  │
│  │  Sorusu  │    │  Dönüşüm │    │  Arama   │    │  3.5   │  │
│  └─────────┘    └──────────┘    └──────────┘    └────────┘  │
│                                       │              │       │
│                              En yakın 3 chunk    Yanıt      │
│                              (context)           üretimi     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Açıklama | Versiyon |
|-----------|----------|----------|
| **Python** | Ana programlama dili | 3.10+ |
| **OpenAI GPT-3.5 Turbo** | Doğal dil ile yanıt üretimi | API |
| **FAISS** | Facebook AI Similarity Search — vektör benzerlik arama | 1.13.2 |
| **Sentence Transformers** | Metin embedding modeli (`BAAI/bge-m3`) | 5.4.1 |
| **PyMuPDF (fitz)** | PDF dosyalarından metin çıkarma | 1.27.2 |
| **python-dotenv** | `.env` dosyasından ortam değişkenlerini yükleme | 1.2.2 |
| **NumPy** | Sayısal hesaplamalar ve vektör işlemleri | 2.4.4 |

---

## 🚀 Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/Meszn/RAG_Sozlesme_Avukati.git
cd RAG_Sozlesme_Avukati
```

### 2. Sanal ortam oluşturun

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. Ortam değişkenlerini ayarlayın

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
OPENAI_API_KEY = "sk-proj-..."
```

> ⚠️ **Önemli:** API anahtarınızı asla paylaşmayın ve `.env` dosyasını Git'e yüklemeyin.

### 5. Sözleşme PDF'inizi ekleyin

`data/` klasörüne analiz etmek istediğiniz sözleşme PDF dosyasını ekleyin.

---

## 💡 Kullanım

### Adım 1: Vektör Veritabanını Oluşturun

PDF sözleşmenizi vektörel temsile dönüştürmek için:

```bash
python build_vector_database.py
```

Bu komut:
- PDF dosyasından metni çıkaracak
- Metni 500 karakterlik parçalara (chunk) bölecek
- Her parça için embedding vektörü oluşturacak
- FAISS indeksini ve chunk verilerini `data/` klasörüne kaydedecek

### Adım 2: Soru Sorun

```bash
python ask.question.py
```

Program çalıştığında interaktif bir soru-cevap oturumu başlar:

```
Sorunuzu giriniz: Sözleşmenin tarafları kimlerdir?

AI Assistant:
 Sözleşmenin tarafları ABC Teknoloji A.Ş. ve XYZ Yazılım Ltd. Şti.'dir.

Sorunuzu giriniz: Ücret ne kadar?

AI Assistant:
 Toplam proje ücreti 150.000 TL + KDV olarak belirlenmiştir.

Sorunuzu giriniz: q
Çıkış yapılıyor...
```

> Çıkmak için `exit`, `quit` veya `q` yazabilirsiniz.

---

## 📁 Proje Yapısı

```
RAG_Sozlesme_Avukati/
│
├── ask.question.py              # Ana soru-cevap motoru (RAG pipeline)
├── build_vector_database.py     # PDF → Vektör veritabanı oluşturucu
├── requirements.txt             # Python bağımlılıkları
├── .env                         # API anahtarları (Git'e yüklenmez)
├── .gitignore                   # Git tarafından yok sayılacak dosyalar
├── LICENSE                      # MIT Lisansı
├── README.md                    # Proje dokümantasyonu
│
└── data/
    ├── Kapsamli_Yazilim_Sozlesmesi.pdf   # Örnek sözleşme PDF dosyası
    ├── contracy_index.faiss              # FAISS vektör indeksi
    └── contracy_chunks.pkl               # Chunk verileri (pickle)
```

---

## 📚 RAG Nedir?

**RAG (Retrieval-Augmented Generation)**, büyük dil modellerinin (LLM) performansını artırmak için bilgi erişimi ile metin üretimini birleştiren bir tekniktir.

### RAG'ın 3 Temel Adımı:

| Adım | Açıklama |
|------|----------|
| **🔍 Retrieval (Bilgi Çekme)** | Kullanıcının sorusu embedding vektörüne dönüştürülür ve FAISS ile en benzer metin parçaları bulunur |
| **📎 Augmentation (Zenginleştirme)** | Bulunan metin parçaları, LLM'e bağlam (context) olarak sunulur |
| **✍️ Generation (Üretim)** | GPT-3.5, soruyu ve bağlamı kullanarak doğru ve anlamlı bir yanıt üretir |

### Neden RAG?

- ✅ **Güncel bilgi:** LLM'in eğitim verisinde olmayan bilgilere erişim sağlar
- ✅ **Doğruluk:** Hallüsinasyonu azaltır, gerçek verilere dayalı yanıtlar üretir
- ✅ **Özelleştirme:** Kendi belgeleriniz üzerinde çalışır
- ✅ **Maliyet:** Fine-tuning'e kıyasla çok daha ucuz ve hızlıdır

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

<p align="center">
  <b>⭐ Bu projeyi faydalı bulduysanız yıldız vermeyi unutmayın!</b>
</p>

<p align="center">
  Geliştirici: <a href="https://github.com/Meszn">Mustafa Sezen</a>
</p>
