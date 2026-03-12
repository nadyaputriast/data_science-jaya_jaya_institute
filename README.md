# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institute

- **Nama:** Putu Nadya Putri Astina
- **Email:** nadyaputriast@gmail.com
- **ID Dicoding:** nadyaputriast

---

## Business Understanding

Jaya Jaya Institute merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Saat ini, institusi menghadapi tantangan besar terkait tingginya rasio mahasiswa yang putus kuliah (*dropout*). Tingginya angka *dropout* ini tidak hanya berdampak buruk pada reputasi institusi, tetapi juga mengindikasikan adanya masalah sistemik dalam dukungan kesejahteraan dan proses belajar mengajar. Jaya Jaya Institute membutuhkan solusi berbasis data untuk mengidentifikasi akar masalah dan mendeteksi dini mahasiswa yang berisiko *dropout*.

### Permasalahan Bisnis

1. Faktor-faktor kuantitatif apa saja (dari sisi demografi, status finansial, dan rekam jejak akademik) yang memiliki korelasi tertinggi terhadap tingginya tingkat *dropout* mahasiswa?
2. Bagaimana tren metrik evaluasi dan nilai mahasiswa di semester 1 dan semester 2 dapat dijadikan sebagai indikator peringatan dini (*early warning indicator*) bahwa mahasiswa tersebut berpotensi putus kuliah?
3. Bagaimana cara memprediksi status akhir keberhasilan studi mahasiswa (*Dropout, Enrolled, Graduate*) secara akurat sejak dini menggunakan pemodelan *Machine Learning* berdasarkan data historis mereka?

### Cakupan Proyek

- **Eksplorasi & Persiapan Data:** Melakukan pembersihan data, menerjemahkan fitur (*mapping*) ke dalam Bahasa Indonesia agar relevan secara bisnis, menangani multikolinearitas (menghapus fitur dengan korelasi > 0.9), dan melakukan standardisasi data numerik.
- **Pembuatan Business Dashboard:** Membangun *dashboard* interaktif menggunakan Looker Studio. Dashboard dirancang secara visual (tanpa menggunakan tabel mentah) untuk memantau metrik krusial pendorong performa siswa (nilai akademik dan status finansial).
- **Machine Learning Modeling:** Mengembangkan dan melatih algoritma **XGBoost Classifier** untuk menangani masalah klasifikasi multikelas pada dataset yang tidak seimbang (*imbalanced data*).
- **Deployment & Explainable AI (XAI):** Membuat *prototype* sistem prediksi berbasis web interaktif menggunakan **Streamlit** yang dilengkapi dengan **SHAP (SHapley Additive exPlanations)** agar model tidak menjadi *black box*, melainkan dapat menjelaskan 3 faktor utama pendorong setiap prediksi.

### Persiapan

**Sumber Data:** Dataset historis performa mahasiswa (*Student Performance Dataset*) yang mencakup 36 fitur pendukung seperti data demografi, latar belakang orang tua, status finansial, kondisi makroekonomi, dan rekam jejak akademik pada semester 1 dan 2.

**Setup Environment:**

Sangat disarankan untuk menggunakan *Virtual Environment* agar dependensi *library* tidak saling berbenturan.

1. Buat *virtual environment* baru:
   ```bash
   python -m venv venv
   ```

2. Aktifkan *virtual environment*:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Buat file `requirements.txt` dan isi dengan *library* berikut:
   ```text
   streamlit==1.55.0
   pandas==2.3.3
   numpy==2.4.3
   scikit-learn==1.8.0
   xgboost==3.2.0
   joblib==1.5.2
   shap==0.51.0
   ```

4. Jalankan perintah instalasi:
   ```bash
   pip install -r requirements.txt
   ```

---

## Struktur File Proyek

```
submission/
├── app.py                  # Aplikasi utama Streamlit
├── expert_system.py        # Modul logika rekomendasi berbasis aturan
├── mappings.py             # Modul kamus mapping fitur & tooltip
├── notebook.ipynb          # Notebook eksplorasi data & pelatihan model
├── xgb_model.pkl           # Model XGBoost hasil training (tersimpan)
├── scaler.pkl              # StandardScaler hasil training (tersimpan)
├── encoder.pkl             # LabelEncoder untuk target kelas
├── requirements.txt        # Daftar dependensi library
└── README.md               # Dokumentasi proyek ini
```

---

## Pengembangan Model Machine Learning & Deployment

### Evaluasi Model Terbaik

Berdasarkan proses pelatihan dan pengujian berbagai algoritma, **XGBoost Classifier** terpilih sebagai model terbaik. XGBoost terbukti sangat tangguh dalam menangani data tabular yang kompleks dan secara bawaan mampu meminimalisir bias pada dataset yang tidak seimbang (*imbalanced target*) berkat mekanisme *boosting*. 

Dibandingkan *Random Forest* dan *Gradient Boosting*, XGBoost unggul terutama dalam mendeteksi kelas minoritas **Enrolled** dengan *recall* tertinggi sebesar **0.57** (vs. 0.39 dan 0.40) dan *F1-score* **0.50** (vs. 0.44 dan 0.43), metrik yang paling krusial dalam konteks sistem peringatan dini, karena kesalahan melewatkan mahasiswa berisiko (*false negative*) lebih berbahaya daripada *false positive*. Secara keseluruhan, XGBoost mencatatkan *weighted F1-score* **0.74** dengan *macro recall* **0.70**, mengungguli kedua model pembanding yang hanya mencapai *macro recall* 0.67.


### Fitur yang Digunakan Model (32 Fitur)

Setelah proses seleksi fitur untuk menghilangkan multikolinearitas (korelasi > 0.9), model menggunakan **32 fitur** berikut, perhatikan bahwa 4 kolom berikut **dibuang** dari dataset asli karena redundan:
- `Fathers_occupation` (korelasi tinggi dengan `Mothers_occupation`)
- `Curricular_units_2nd_sem_credited` (korelasi tinggi dengan kolom sem 1)
- `Curricular_units_2nd_sem_enrolled` (korelasi tinggi dengan kolom sem 1)
- `Curricular_units_2nd_sem_approved` (korelasi tinggi dengan kolom sem 1)

### Cara Menjalankan Aplikasi Secara Lokal

1. Arahkan terminal ke folder yang berisi `app.py` beserta semua file model (`.pkl`):
   ```bash
   cd path/to/submission/
   ```

2. Pastikan *virtual environment* aktif dan semua dependensi sudah terinstal.

3. Jalankan perintah berikut:
   ```bash
   streamlit run app.py
   ```

4. Aplikasi akan otomatis terbuka di browser pada alamat:
   ```
   http://localhost:8501
   ```

### Link Aplikasi (Streamlit Cloud)

> 🚀 **[Buka Aplikasi Prediksi - Jaya Jaya Institute](https://jaya-jaya-institute-nadyaputriast.streamlit.app/)**

---

## Business Dashboard

*Business Dashboard* telah dikembangkan menggunakan **Looker Studio** untuk membantu manajemen kampus memantau performa mahasiswa secara *real-time*. Dashboard ini **tidak menampilkan tabel mentah**, melainkan murni menggunakan visualisasi grafis (*100% Stacked Bar Chart, Grouped Bar Chart, Donut Chart, dan Scorecard*) agar mudah dipahami.

Dashboard secara spesifik memantau faktor-faktor terpenting yang menentukan performa mahasiswa:

1. **Overview KPI Makro**, Total mahasiswa, jumlah *dropout*, dan jumlah *graduate*.
2. **Monitoring Faktor Finansial**, Korelasi antara status menunggak SPP, kepemilikan beasiswa, dan beban utang terhadap rasio *dropout*.
3. **Monitoring Faktor Akademik**, Perbandingan rata-rata nilai dan metrik evaluasi semester 1 & 2 sebagai indikator keberhasilan belajar.

> 📊 **[Buka Dashboard Looker Studio](https://lookerstudio.google.com/reporting/fde1f490-53d0-4586-be22-f9a4ea9108ec)**

---

## Conclusion

Berdasarkan hasil analisis data eksploratif (*EDA*) dan visualisasi pada *dashboard*, permasalahan tingginya tingkat *dropout* di Jaya Jaya Institute berakar dari dua kendala utama:

1. **Krisis Finansial Menjadi Pemicu Terbesar:** Terdapat korelasi yang sangat kuat antara status ekonomi dan angka *dropout*. Kelompok mahasiswa yang **menunggak SPP**, **memiliki utang (debtor)**, dan **tidak memiliki beasiswa** mendominasi persentase mahasiswa yang putus kuliah.

2. **Kegagalan Adaptasi Akademik di Tahun Pertama:** Mahasiswa yang berakhir *dropout* rata-rata sudah menunjukkan rekam jejak yang sangat buruk sejak semester 1. Nilai mereka berada jauh di bawah kelompok mahasiswa yang berhasil lulus, dan performa tersebut tidak membaik di semester 2, diikuti dengan tingginya jumlah SKS yang tidak dievaluasi, mengindikasikan hilangnya motivasi sejak awal masa studi.

### Rekomendasi Action Items

1. **Implementasi Sistem Peringatan Dini (*Early Warning System*):** Gunakan aplikasi prediksi *Machine Learning* yang telah dibuat untuk mengevaluasi setiap mahasiswa baru di akhir semester 1. Fokuskan perhatian pada mahasiswa yang diprediksi masuk kategori *Dropout*.

2. **Program Restrukturisasi Finansial Proaktif:** Kampus harus lebih cepat mendeteksi mahasiswa yang mulai menunggak SPP di awal semester. Tawarkan program relaksasi pembayaran (cicilan) atau sediakan kuota *"Beasiswa Bantuan Darurat"* khusus bagi mahasiswa dari latar belakang ekonomi rentan yang memiliki nilai akademik mumpuni.

3. **Pendampingan Akademik (Mentoring) Terstruktur:** Mahasiswa yang gagal mencapai standar nilai minimal (di bawah rata-rata 10–12) pada semester 1 wajib diikutsertakan dalam program bimbingan akademik intensif atau tutor sebaya sebelum memasuki semester 2 untuk mencegah *burnout*.

4. **Penyediaan Fleksibilitas Waktu Studi:** Mempertimbangkan tingginya kebutuhan mahasiswa untuk bekerja demi menutupi beban biaya kuliah, institusi disarankan menawarkan opsi pemindahan jadwal ke kelas malam (*Evening Attendance*) sehingga mahasiswa tidak terpaksa melakukan *dropout*.