# Deep Learning: Analisis Konseptual & Eksperimen Transfer Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Oxford--IIIT--Pet-blue)](https://www.robots.ox.ac.uk/~vgg/data/pets/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

Repositori ini memuat berkas tugas dan laporan laboratorium **GSLC Deep Learning** yang mencakup analisis teoritis arsitektur modern CNN serta benchmark eksperimen praktis klasifikasi citra pada subset dataset **Oxford-IIIT Pet** (10 kelas).

---

## Jalankan di Google Colab
Klik tombol di bawah ini untuk langsung menjalankan eksperimen di Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/b3Peee/Advanced-CNNs-and-Transfer-Learning/blob/main/GSLC_DeepLearning.ipynb)

---

## 📑 Daftar Isi
1. [Bagian A: Analisis Konseptual](#bagian-a-analisis-konseptual)
2. [Bagian C1: Tabel Konsolidasi Eksperimen](#bagian-c1-tabel-konsolidasi-eksperimen)
3. [Bagian C2: Pembahasan Analisis Komparatif](#bagian-c2-pembahasan-analisis-komparatif)
4. [Bagian C3: Failure Analysis (Analisis Misklasifikasi)](#bagian-c3-failure-analysis)
5. [Bagian C4: Rekomendasi Rekayasa & Limitasi](#bagian-c4-rekomendasi-rekayasa--batasan-eksperimen)
6. [Bagian C5: Visualisasi Fitur (Grad-CAM)](#bagian-c5-eksplorasi-tambahan-grad-cam)
7. [Pernyataan Penggunaan AI](#pernyataan-penggunaan-ai)

---

## Bagian A: Analisis Konseptual

### 1. Efek Regularisasi pada Batch Normalization
Batch Normalization menghitung mean dan varians pada setiap mini-batch saat pelatihan, lalu menormalisasi aktivasi. Ketergantungan aktivasi suatu sampel terhadap sampel lain dalam mini-batch menimbulkan efek stochastic noise yang berfungsi sebagai *mild regularizer*, sehingga pada banyak skenario meniadakan kebutuhan Dropout (*Ioffe & Szegedy, 2015*). Namun, efektivitas regulasi Batch Normalization menurun drastis pada:
* Ukuran mini-batch yang sangat kecil atau sampel tidak independen (*Ioffe, 2017*).
* Adanya disparitas perhitungan statistik antara fase training dan inference.
* Kasus jaringan sangat dalam yang overfit ekstrem (misal ResNet-1202 pada CIFAR-10) di mana regularisasi eksplisit seperti Dropout tetap diperlukan (*He et al., 2016*).

### 2. Degradasi Jaringan Dalam & Formulasi Residual
Jaringan yang lebih dalam teoritis tidak boleh memiliki training error yang lebih buruk dari jaringan dangkal karena layer tambahan cukup menjadi *identity mapping*. Faktanya, plain-34 memiliki error lebih tinggi dari plain-18 bukan karena overfitting, melainkan kesulitan solver mengoptimasi fungsi identitas melalui deretan transformasi non-linear (*He et al., 2016*). Residual learning mengatasi ini dengan merumuskan: F(x) := H(x) - x. Dengan menambahkan *skip connection* y = F(x) + x, gradien dapat mengalir langsung tanpa hambatan (*vanishing/exploding gradient* dapat dicegah), dan jika pemetaan identitas sudah optimal, solver hanya perlu menekan residual F(x) -> 0.

### 3. Feature Extraction vs. Fine-Tuning
* **Feature Extraction**: Membekukan (*freeze*) seluruh bobot *backbone pretrained* dan hanya melatih *classifier head* (linear layer akhir). Efisien komputasi, minim memori, serta ideal saat dataset target terbatas dan mirip dengan dataset pretrain.
* **Fine-Tuning**: Membuka sebagian atau seluruh bobot *backbone* untuk diperbarui dengan *learning rate* rendah. Diperlukan ketika target domain memiliki karakteristik khusus yang belum terwakili secara optimal oleh fitur umum ImageNet.

### 4. Kaidah Penerapan Data Augmentation
* Augmentasi **hanya diterapkan pada Train Set** untuk meningkatkan variasi generalisasi tanpa merusak integritas evaluasi (*PyTorch, 2025; Lones, 2024*).
* Menjalankan augmentasi sebelum split train-val/test memicu **Data Leakage**, menghasilkan metrik yang *overly optimistic*.
* Transformasi wajib bersifat **label-preserving** (*Shorten & Khoshgoftaar, 2019*). Contoh transformasi tidak layak: cropping atau zooming terlalu ekstrem pada klasifikasi ras hewan yang dapat memotong telinga atau moncong, menghilangkan ciri diagnostik ras tersebut.

### 5. Mitigasi Catastrophic Forgetting
*Catastrophic Forgetting* terjadi jika model di-fine-tune terlalu agresif hingga bobot pra-latih rusak (*Howard & Ruder, 2018*). Strategi pencegahannya:
1. **Differential Learning Rate**: Memberikan learning rate lebih rendah pada layer bawah dan lebih tinggi pada layer atas: η^(l-1) = η^l / 2.6.
2. **Gradual Unfreezing**: Membuka blok layer secara bertahap dari layer akhir menuju layer awal per epok.
3. **Slanted Triangular Learning Rate**: Penjadwalan learning rate yang naik cepat secara linier lalu melandai perlahan.

---

## Bagian C1: Tabel Konsolidasi Eksperimen

Pengujian dilakukan pada subset **10 kelas** dari dataset Oxford-IIIT Pet:

| Kode Eksperimen | Strategi / Model | Trainable Params | Learning Rate | Epoch Konvergen | Waktu Latih (s) | Akurasi Validasi | Macro F1 (Test) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **B1** | Scratch (ResNet) | 11.181.642 | `1e-3` | 12 | 106.04 | 35.57% | 0.4869 |
| **B2** | Feature Extraction (ResNet) | 5.130 | `1e-3` | 13 | 88.97 | 89.93% | 0.9394 |
| **B3** | Progressive Fine-Tuning | 8.398.858 | layer4: `1e-4`, fc: `1e-3` | 6 | 56.68 | **91.95%** | **0.9395** |
| **B4 (Tanpa Aug)** | Fine-Tuning Non-Aug | 8.398.858 | layer4: `1e-4`, fc: `1e-3` | 6 | 55.90 | **91.95%** | **0.9395** |
| **B4 (Dengan Aug)** | Fine-Tuning + Aug | 8.398.858 | layer4: `1e-4`, fc: `1e-3` | 3 | 51.13 | 90.60% | 0.9195 |
| **B5** | EfficientNet-B0 | **424.970** | features: `1e-4`, cls: `1e-3` | 14 | 128.76 | **91.95%** | 0.9067 |

---

## Bagian C2: Pembahasan Analisis Komparatif

1. **Disparitas Performa B1 vs. B2 & B3**:  
   Pelatihan dari nol (B1) hanya meraih akurasi 35.57%, sedangkan transfer learning melonjak ke >89%. Keterbatasan data latih menyebabkan B1 gagal membentuk hierarki representasi visual dasar, sementara bobot pretrained ImageNet menyediakan ekstraktor fitur yang kaya.
2. **Trade-off Performa vs. Stabilitas**:  
   Feature Extraction (B2) sangat stabil dan imun overfitting dengan hanya 5.130 parameter terlatih, namun terbentur batas ekspresi domain. Fine-tuning progresif (B3) mengaktifkan 8,3 juta parameter dengan *differential learning rate* untuk mencegah catastrophic forgetting, mencapai konvergensi tercepat (epoch 6) dan skor F1 tertinggi (0.9395).
3. **Peran Augmentasi dalam Meredam Overfitting**:  
   Pada B4 dengan augmentasi, *early stopping* terpicu lebih awal di epoch ke-3 (90.60%). Intensitas augmentasi awal yang agresif menambahkan variasi noise yang sedikit memperlambat pembentukan pola dasar, meski teoritis bermanfaat mempersempit gap train-val pada durasi training panjang.

---

## Bagian C3: Failure Analysis

Analisis kesalahan prediksi pada model terbaik (**B3**):

1. **American Bulldog -> Boxer** (*Confidence: 0.59*):  
   *Penyebab*: *Inter-class ambiguity* tinggi; kedua ras memiliki profil morfologi serupa (rahang persegi, moncong pendek, tekstur bulu).
2. **Beagle -> Basset Hound** (*Confidence: 0.56*):  
   *Penyebab*: Berbagi corak *tri-color* dan telinga panjang menggantung; crop wajah menghilangkan konteks proporsi postur tubuh.
3. **Bengal -> Birman** (*Confidence: 0.25*):  
   *Penyebab*: *Poor illumination / strong backlight* yang mereduksi kontras tekstur totol khas ras Bengal.
4. **American Bulldog -> Beagle** (*Confidence: 0.70*):  
   *Penyebab*: Sudut kamera ekstrem (*side-profile*) pada pencahayaan redup menyembunyikan geometri rahang bulldog.
5. **American Pit Bull Terrier -> Bombay** (*Confidence: 0.48*):  
   *Penyebab*: *Motion blur* dan warna hitam pekat homogen melenyapkan fitur anatomis pembeda antara anjing dan kucing.

---

## Bagian C4: Rekomendasi Rekayasa & Batasan Eksperimen

* **Rekomendasi**:  
  * Dataset terbatas -> Inisiasi dengan Feature Extraction.  
  * Target akurasi maksimal -> Transisi ke Progressive Fine-Tuning dengan Differential LR.  
  * Komputasi / Edge Devices -> Gunakan **EfficientNet-B0**, memangkas parameter drastis menjadi hanya **424.970** tanpa mengorbankan akurasi puncak (91.95%).
* **Limitasi Eksperimen**:  
  * *Single-seed testing* (disarankan k-fold cross validation untuk pengujian generalisasi lebih kuat).  
  * Terbatas pada subset 10 kelas dari dataset penuh.

---

## Bagian C5: Eksplorasi Tambahan (Grad-CAM)

Pemeriksaan representasi fitur menggunakan **Grad-CAM** pada blok layer4:
* **Sebelum Fine-Tuning (B2)**: Peta atensi berbasis ImageNet umum cenderung menyebar ke latar belakang dan kurang fokus pada fitur primer.
* **Sesudah Fine-Tuning (B3)**: Aktivasi model terkalibrasi tajam, mengabaikan noise latar belakang, dan berpusat langsung pada fitur diskriminatif utama (misal: area hidung dan moncong subjek).

---

## Pernyataan Penggunaan AI
Penggunaan kecerdasan buatan dalam pengerjaan tugas ini:
* **Tools**: Gemini, ChatGPT
* **Ruang Lingkup**:
  
  * Code Debugging.
  * Pendalaman pemahaman konsep materi.
  * Bantuan analisis hasil eksperimen.
