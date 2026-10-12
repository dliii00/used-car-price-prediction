# Prediksi Harga Mobil Bekas dengan Machine Learning

Project ini merupakan tugas praktikum **Machine Learning** yang berfokus pada analisis dan prediksi harga mobil bekas menggunakan pendekatan **Supervised Learning – Regression**.

Project mencakup seluruh tahapan dasar dalam proses Machine Learning, mulai dari **data loading, data understanding, data cleaning, penanganan missing value, deteksi outlier, standardisasi, encoding data kategorikal, Exploratory Data Analysis (EDA), train-test split, pembangunan model regresi, hingga evaluasi dan pemilihan model terbaik**.

---

## Tentang Project

Tujuan utama project ini adalah membangun model Machine Learning yang dapat memprediksi **harga mobil bekas (`price`)** berdasarkan berbagai karakteristik kendaraan.

Dataset yang digunakan merupakan dataset **Used Car Price Prediction** yang memiliki informasi mengenai karakteristik mobil, seperti merek, model, tahun kendaraan, jarak tempuh, jenis bahan bakar, mesin, transmisi, warna kendaraan, riwayat kecelakaan, dan clean title.

Pada tahap awal, dataset memiliki:
- **4.009 baris**
- **12 kolom**
- **11 fitur input**
- **1 variabel target**, yaitu `price`

Setelah dilakukan proses cleaning dan penanganan outlier, dataset yang digunakan untuk proses pemodelan berjumlah **3.555 data**.

---

## Tujuan

Project ini bertujuan untuk:
1. Memahami proses dasar pengolahan data dalam Machine Learning.
2. Melakukan data loading menggunakan Python dan Pandas.
3. Memeriksa dan membersihkan dataset.
4. Menangani missing value.
5. Mendeteksi dan menangani outlier menggunakan metode **Interquartile Range (IQR)**.
6. Melakukan standardisasi terhadap data numerik.
7. Mengubah data kategorikal menjadi data numerik.
8. Melakukan **Exploratory Data Analysis (EDA)**.
9. Menganalisis hubungan antarvariabel menggunakan korelasi.
10. Membagi dataset menjadi data training dan testing.
11. Membangun beberapa model regresi.
12. Membandingkan performa model menggunakan **MAE, MSE, dan R²**.
13. Menentukan model dengan performa terbaik untuk prediksi harga mobil bekas.

---

## Dataset

Dataset terdiri dari:

| Kolom | Keterangan |
|---|---|
| `brand` | Merek mobil |
| `model` | Model kendaraan |
| `model_year` | Tahun model kendaraan |
| `mileage` | Jarak tempuh kendaraan |
| `fuel_type` | Jenis bahan bakar |
| `engine` | Spesifikasi mesin |
| `transmission` | Jenis transmisi |
| `ext_col` | Warna eksterior |
| `int_col` | Warna interior |
| `accident` | Informasi riwayat kecelakaan |
| `clean_title` | Informasi clean title |
| `price` | Harga mobil yang menjadi target prediksi |

Target:

```text
price
```

---

# Workflow Machine Learning

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Understanding
   │
   ▼
Data Cleaning
   │
   ├── Missing Value
   ├── Data Format
   └── Duplicate Check
   │
   ▼
Outlier Detection
   │
   ▼
Standardisasi
   │
   ▼
Encoding Data Kategorikal
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Train-Test Split
   │
   ▼
Modeling
   │
   ├── LARS
   ├── Linear Regression
   └── Gradient Boosting Regressor
   │
   ▼
Model Evaluation
   │
   ▼
Pemilihan Model Terbaik
```

## 1. Data Loading

Dataset dibaca dengan Pandas `read_csv()`.

```python
import pandas as pd

df = pd.read_csv("data_mobil_bekas.csv")
```

Pemeriksaan awal:

```python
df.head()
df.info()
df.describe()
```

## 2. Data Cleaning

Tahapan pemeriksaan meliputi:
- Missing value
- Struktur dan tipe data
- Duplikat
- Format dataset
- Outlier

Penanganan missing value:
- Data numerik → median
- Data kategorikal → modus

Pemeriksaan duplikat:

```python
df.duplicated()
```

## 3. Penanganan Missing Value

Jumlah missing value diperiksa menggunakan:

```python
df.isnull().sum()
```

Data numerik diisi dengan nilai median, sedangkan data kategorikal diisi dengan modus. Setelah itu dilakukan pengecekan kembali untuk memastikan missing value telah ditangani.

## 4. Deteksi dan Penanganan Outlier

Variabel numerik yang diperiksa:
- `model_year`
- `mileage`
- `price`

Deteksi outlier menggunakan metode **Interquartile Range (IQR)**:

```text
IQR = Q3 - Q1

Batas bawah = Q1 - 1.5 × IQR
Batas atas  = Q3 + 1.5 × IQR
```

Jumlah data berubah dari:

```text
4.009 data
     ↓
3.555 data
```

## 5. Standardisasi

Standardisasi dilakukan menggunakan `StandardScaler` pada variabel:
- `model_year`
- `mileage`
- `price`

Standardisasi bertujuan untuk menyamakan skala data numerik sebelum digunakan dalam proses pemodelan.

## 6. Encoding Data Kategorikal

Dalam proses preprocessing digunakan beberapa pendekatan encoding:

- **One-Hot Encoding** menggunakan `pd.get_dummies()`
- **Label Encoding** menggunakan `LabelEncoder`

One-Hot Encoding menghasilkan jumlah kolom yang cukup banyak karena beberapa fitur seperti `model` dan `engine` memiliki banyak kategori unik.

Untuk proses pemodelan akhir digunakan dataframe hasil **Label Encoding**, yaitu `df_lencoder`.

## 7. Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik dataset dan hubungan antarvariabel.

Visualisasi yang digunakan:
- Histogram
- Boxplot
- Heatmap korelasi
- Analisis korelasi terhadap `price`

EDA membantu melihat distribusi data, outlier, serta hubungan antara fitur dengan target.

## 8. Train-Test Split

Variabel target:

```python
y = df_lencoder['price']
```

Fitur:

```python
X = df_lencoder.drop(columns=['price'])
```

Dataset kemudian dibagi menjadi:

| Data | Jumlah | Persentase |
|---|---:|---:|
| Training | 2.844 | 80% |
| Testing | 711 | 20% |
| Total | 3.555 | 100% |

---

# Model Machine Learning

Tiga model regresi dibandingkan dalam project ini.

## 1. LARS

```python
Lars(n_nonzero_coefs=1)
```

## 2. Linear Regression

```python
LinearRegression()
```

## 3. Gradient Boosting Regressor

```python
GradientBoostingRegressor(random_state=184)
```

---

# Evaluasi Model

Model dievaluasi menggunakan tiga metrik:

- **MAE (Mean Absolute Error)** → semakin kecil semakin baik.
- **MSE (Mean Squared Error)** → semakin kecil semakin baik.
- **R² (R-squared)** → semakin mendekati 1 semakin baik.

## Hasil Evaluasi

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| LARS | 0.678009 | 0.719743 | 0.262231 |
| Linear Regression | 0.426924 | 0.315625 | 0.676470 |
| **Gradient Boosting Regressor** | **0.316833** | **0.194855** | **0.800265** |

## Model Terbaik

Berdasarkan hasil evaluasi, **Gradient Boosting Regressor** menjadi model terbaik karena menghasilkan:

```text
MAE = 0.316833
MSE = 0.194855
R²  = 0.800265
```

Model ini memiliki **MAE dan MSE paling rendah**, serta **R² paling tinggi** dibandingkan dua model lainnya.

> **Catatan:** MAE dan MSE pada hasil evaluasi berada pada skala target yang telah distandardisasi karena `price` ikut diproses menggunakan standardisasi.

---

# Model Output

Model Gradient Boosting Regressor disimpan dalam dua format:

```text
gbr_model.joblib
gbr_model.pkl
```

File tersebut dapat digunakan untuk menyimpan model hasil training sehingga model tidak perlu dilatih ulang setiap kali ingin digunakan.

---

# Teknologi yang Digunakan

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Pickle**

---

# Struktur Repository

```text
used-car-price-prediction/
│
├── README.md
├── Tugas_laporan.ipynb
├── data_mobil_bekas.csv
├── gbr_model.joblib
└── gbr_model.pkl
```

---


#  Kendala yang Ditemui

Beberapa kendala yang ditemukan selama pengerjaan project antara lain:

1. **Format dataset**
   - Terdapat permasalahan format data yang menyebabkan beberapa nilai terbaca sebagai `NaN`.

2. **Missing value**
   - Dataset memiliki nilai kosong yang perlu ditangani sebelum proses modeling.

3. **Outlier**
   - Ditemukan outlier terutama pada variabel `mileage` dan `price`.

4. **Ketidaksesuaian kode awal**
   - Beberapa kode pada awal pengerjaan masih menggunakan nama kolom dari dataset lain seperti:
     - `OverallQual`
     - `SalePrice`
     - `YearBuilt`
     - `LotArea`
   - Kode kemudian disesuaikan dengan dataset mobil bekas dengan `price` sebagai target.

5. **Jumlah kolom setelah One-Hot Encoding**
   - One-Hot Encoding menghasilkan banyak kolom karena fitur `model` dan `engine` memiliki banyak kategori unik.

6. **Urutan preprocessing**
   - Tahapan preprocessing harus dilakukan secara hati-hati agar proses cleaning, standardisasi, encoding, dan modeling berjalan dengan benar.

---

#  Pengembangan Selanjutnya

Project ini masih dapat dikembangkan lebih lanjut dengan:

- Menggunakan **Random Forest Regressor**
- Mencoba **XGBoost**
- Melakukan **hyperparameter tuning**
- Menambah jumlah dan kualitas dataset
- Menggunakan metode encoding yang lebih sesuai untuk fitur dengan kategori sangat banyak
- Membuat aplikasi sederhana untuk prediksi harga mobil
- Membuat visualisasi yang lebih interaktif
- Membandingkan lebih banyak algoritma Machine Learning

---

#  Referensi

- McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter*. O'Reilly Media.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.).
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
- Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*.
- Waskom, M. L. (2021). *Seaborn: Statistical Data Visualization*.
- Kaggle — *Used Car Price Prediction Dataset*.

---

#  Author

**Fadli Hakiki**

**NIM:** 241011400779  
**Kelas:** 05TPLP019  
**Program Studi:** Teknik Informatika  
**Fakultas:** Fakultas Ilmu Komputer  
**Universitas:** Universitas Pamulang  
**Tahun Akademik:** 2026/2027

---

#  Ringkasan

Project ini menunjukkan bahwa proses Machine Learning tidak hanya berfokus pada pembuatan model, tetapi juga mencakup **data understanding, data cleaning, preprocessing, EDA, dan evaluasi model**.

Dari **4.009 data awal**, setelah proses preprocessing dan penanganan outlier diperoleh **3.555 data** yang digunakan untuk pemodelan.

Dari tiga model yang dibandingkan, yaitu **LARS, Linear Regression, dan Gradient Boosting Regressor**, model **Gradient Boosting Regressor** memberikan performa terbaik dengan:

```text
MAE = 0.316833
MSE = 0.194855
R²  = 0.800265
```

Dengan demikian, Gradient Boosting Regressor dipilih sebagai model terbaik pada project prediksi harga mobil bekas ini.
