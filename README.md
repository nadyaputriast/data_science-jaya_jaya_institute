# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institute

- **Nama:** Putu Nadya Putri Astina
- **Email:** nadyaputriast@gmail.com
- **ID Dicoding:** nadyaputriast
- **[GitHub Repository](https://github.com/nadyaputriast/data_science-jaya_jaya_institute)**

---

## Business Understanding

Jaya Jaya Institute merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Saat ini, institusi menghadapi tantangan besar terkait tingginya rasio mahasiswa yang putus kuliah (*dropout*). Tingginya angka *dropout* ini tidak hanya berdampak buruk pada reputasi institusi, tetapi juga mengindikasikan adanya masalah sistemik dalam dukungan kesejahteraan dan proses belajar mengajar. Jaya Jaya Institute membutuhkan solusi berbasis data untuk mengidentifikasi akar masalah dan mendeteksi dini mahasiswa yang berisiko *dropout*.

### Permasalahan Bisnis

1. Faktor-faktor kuantitatif apa saja (dari sisi demografi, status finansial, dan rekam jejak akademik) yang memiliki korelasi tertinggi terhadap tingginya tingkat *dropout* mahasiswa?
2. Bagaimana tren metrik evaluasi dan nilai mahasiswa di semester 1 dan semester 2 dapat dijadikan sebagai indikator peringatan dini (*early warning indicator*) bahwa mahasiswa tersebut berpotensi putus kuliah?
3. Bagaimana cara memprediksi status akhir keberhasilan studi mahasiswa (*Dropout* atau *Graduate*) secara akurat sejak dini menggunakan pemodelan *Machine Learning* berdasarkan data historis mereka?

### Cakupan Proyek

- **Eksplorasi & Persiapan Data:** Melakukan pembersihan data, menerjemahkan fitur (*mapping*) ke dalam Bahasa Indonesia agar relevan secara bisnis, menangani multikolinearitas (menghapus fitur dengan korelasi > 0.85), dan melakukan standardisasi data numerik.
- **Pembuatan Business Dashboard:** Membangun *dashboard* interaktif menggunakan Looker Studio. Dashboard dirancang secara visual (tanpa menggunakan tabel mentah) untuk memantau metrik krusial pendorong performa siswa (nilai akademik dan status finansial).
- **Machine Learning Modeling:** Mengembangkan dan melatih algoritma **Random Forest Classifier** untuk klasifikasi biner (*Dropout* vs *Graduate*) menggunakan data mahasiswa yang sudah memiliki status akhir jelas. Data mahasiswa berstatus *Enrolled* tidak diikutsertakan dalam proses training karena status akhir mereka belum diketahui. Data ini dimanfaatkan pada tahap **inferensi** untuk memprediksi apakah mereka berpotensi *Dropout* atau *Graduate* di masa depan.
- **Deployment & Explainable AI (XAI):** Membuat *prototype* sistem prediksi berbasis web interaktif menggunakan **Streamlit** yang dilengkapi dengan **SHAP (SHapley Additive exPlanations)** agar model tidak menjadi *black box*, melainkan dapat menjelaskan 3 faktor utama pendorong setiap prediksi.

### Persiapan

**Sumber Data:** Dataset historis performa mahasiswa yang bersumber dari repositori UCI Machine Learning:
[Students' Performance Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

Dataset mencakup 36 fitur pendukung seperti data demografi, latar belakang orang tua, status finansial, kondisi makroekonomi, dan rekam jejak akademik pada semester 1 dan 2.

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
├── model/
│   ├── rf_model.pkl          # Model Random Forest hasil training (tersimpan)
│   ├── scaler.pkl            # StandardScaler hasil training (tersimpan)
│   └── encoder.pkl           # LabelEncoder untuk target kelas
├── app.py                    # Aplikasi utama Streamlit
├── expert_system.py          # Modul logika rekomendasi berbasis aturan
├── mappings.py               # Modul kamus mapping fitur & tooltip
├── notebook.ipynb            # Notebook eksplorasi data & pelatihan model
├── nadyaputriast-dashboard.jpg  # Screenshot dashboard
├── requirements.txt          # Daftar dependensi library
└── README.md                 # Dokumentasi proyek ini
```

---

## Pengembangan Model Machine Learning & Deployment

### Strategi Pemodelan

Model dilatih menggunakan **hanya data mahasiswa dengan status akhir yang sudah jelas**, yaitu *Dropout* (1.421 data) dan *Graduate* (2.209 data), total **3.630 data**. Data mahasiswa berstatus *Enrolled* (794 data) **tidak diikutsertakan dalam training** karena status akhir studi mereka belum dapat dikonfirmasi — data ini justru menjadi target utama **inferensi** untuk mendeteksi mahasiswa yang berisiko *dropout* di masa depan.

### Evaluasi Model Terbaik

Berdasarkan proses pelatihan dan pengujian berbagai algoritma, **Random Forest Classifier** terpilih sebagai model terbaik. Berikut ringkasan perbandingan performa ketiga model pada data uji:

| Model | Accuracy | Macro F1 | Macro Recall | Precision Dropout | Recall Dropout |
|---|---|---|---|---|---|
| **Random Forest** | **0.906** | **0.90** | **0.89** | **0.94** | 0.81 |
| Gradient Boosting | 0.902 | 0.89 | 0.89 | 0.93 | 0.81 |
| XGBoost | 0.895 | 0.89 | 0.88 | 0.92 | 0.80 |

**Random Forest terpilih** berdasarkan tiga pertimbangan utama:

1. **Macro F1-Score Tertinggi (0.90)** — paling seimbang dalam memprediksi kedua kelas tanpa bias ke kelas mayoritas.
2. **Precision Dropout Tertinggi (0.94)** — dari seluruh mahasiswa yang diprediksi *Dropout*, 94% benar-benar *Dropout*, sehingga intervensi kampus tidak terbuang sia-sia.
3. **Performa Konsisten** — unggul di seluruh metrik secara bersamaan, bukan hanya di satu aspek saja.

Untuk menangani ketidakseimbangan kelas (*Graduate* 60.9% vs *Dropout* 39.1%), diterapkan parameter `class_weight='balanced'` yang secara otomatis memberikan bobot lebih besar pada kelas *Dropout* agar model tidak bias ke kelas mayoritas.

### Fitur yang Digunakan Model (32 Fitur)

Setelah proses seleksi fitur untuk menghilangkan multikolinearitas (korelasi > 0.85), model menggunakan **32 fitur**. Empat kolom berikut **dibuang** dari dataset asli karena redundan:
- `Fathers_occupation` (korelasi tinggi dengan `Mothers_occupation`)
- `Curricular_units_2nd_sem_credited` (korelasi tinggi dengan kolom sem 1)
- `Curricular_units_2nd_sem_enrolled` (korelasi tinggi dengan kolom sem 1)
- `Curricular_units_2nd_sem_approved` (korelasi tinggi dengan kolom sem 1)

### Cara Menjalankan Aplikasi Secara Lokal

1. Pindahkan file `rf_model.pkl`, `scaler.pkl`, dan `encoder.pkl` dari folder `model/` ke direktori yang sama dengan `app.py`.

2. Arahkan terminal ke folder tersebut:
   ```bash
   cd path/to/submission/
   ```

3. Pastikan *virtual environment* aktif dan semua dependensi sudah terinstal.

4. Jalankan perintah berikut:
   ```bash
   streamlit run app.py
   ```

5. Aplikasi akan otomatis terbuka di browser pada alamat:
   ```
   http://localhost:8501
   ```

### Link Aplikasi (Streamlit Cloud)

> 🚀 **[Buka Aplikasi Prediksi — Jaya Jaya Institute](https://jaya-jaya-institute-nadyaputriast.streamlit.app/)**

---

## Business Dashboard

*Business Dashboard* telah dikembangkan menggunakan **Looker Studio** untuk membantu manajemen kampus memantau performa mahasiswa secara *real-time*. Dashboard ini **tidak menampilkan tabel mentah**, melainkan murni menggunakan visualisasi grafis (*100% Stacked Bar Chart, Grouped Bar Chart, Donut Chart, dan Scorecard*) agar mudah dipahami.

Dashboard secara spesifik memantau faktor-faktor terpenting yang menentukan performa mahasiswa:

1. **Overview KPI Makro** — Total mahasiswa, jumlah *dropout*, dan jumlah *graduate*.
2. **Monitoring Faktor Finansial** — Korelasi antara status menunggak SPP, kepemilikan beasiswa, dan beban utang terhadap rasio *dropout*.
3. **Monitoring Faktor Akademik** — Perbandingan rata-rata nilai dan metrik evaluasi semester 1 & 2 sebagai indikator keberhasilan belajar.

> 📊 **[Buka Dashboard Looker Studio](https://lookerstudio.google.com/reporting/fde1f490-53d0-4586-be22-f9a4ea9108ec)**

---

## Conclusion

Berdasarkan hasil analisis data eksploratif (*EDA*), visualisasi *dashboard*, dan pemodelan *machine learning*, permasalahan tingginya tingkat *dropout* di Jaya Jaya Institute berakar dari dua kendala utama:

1. **Krisis Finansial Menjadi Pemicu Terbesar:** Terdapat korelasi yang sangat kuat antara status ekonomi dan angka *dropout*. Kelompok mahasiswa yang **menunggak SPP**, **memiliki utang (debtor)**, dan **tidak memiliki beasiswa** mendominasi persentase mahasiswa yang putus kuliah. Temuan ini dikonfirmasi oleh model Random Forest yang secara konsisten menempatkan `Tuition_fees_up_to_date` dan `Debtor` sebagai fitur dengan nilai SHAP tertinggi dalam prediksi *dropout*.

2. **Kegagalan Adaptasi Akademik di Tahun Pertama:** Mahasiswa yang berakhir *dropout* rata-rata sudah menunjukkan rekam jejak yang sangat buruk sejak semester 1. Nilai mereka berada jauh di bawah kelompok mahasiswa yang berhasil lulus, dan performa tersebut tidak membaik di semester 2 — diikuti dengan tingginya jumlah SKS yang tidak dievaluasi, mengindikasikan hilangnya motivasi sejak awal masa studi.

**Performa Model:** Random Forest Classifier berhasil mencatatkan *macro recall* **0.89**, *macro F1-score* **0.90**, dan *precision Dropout* **0.94** pada data uji, mengungguli Gradient Boosting dan XGBoost di seluruh metrik utama. Selain itu, model ini juga berhasil memprediksi bahwa dari **794 mahasiswa yang masih berstatus *Enrolled***, sebanyak **327 orang (41.3%) berpotensi Dropout** dan **467 orang (58.7%) diprediksi akan Graduate** — hasil ini menjadi landasan konkret bagi kampus untuk segera melakukan intervensi tepat sasaran.

### Rekomendasi Action Items

1. **Implementasi Sistem Peringatan Dini (*Early Warning System*):** Gunakan aplikasi prediksi *Machine Learning* yang telah dibuat untuk mengevaluasi setiap mahasiswa baru di akhir semester 1. Fokuskan perhatian pada mahasiswa yang diprediksi masuk kategori *Dropout*, terutama dari hasil inferensi 327 mahasiswa *Enrolled* yang berisiko.

2. **Program Restrukturisasi Finansial Proaktif:** Kampus harus lebih cepat mendeteksi mahasiswa yang mulai menunggak SPP di awal semester. Tawarkan program relaksasi pembayaran (cicilan) atau sediakan kuota *"Beasiswa Bantuan Darurat"* khusus bagi mahasiswa dari latar belakang ekonomi rentan yang memiliki nilai akademik mumpuni.

3. **Pendampingan Akademik (Mentoring) Terstruktur:** Mahasiswa yang gagal mencapai standar nilai minimal (di bawah rata-rata 10–12) pada semester 1 wajib diikutsertakan dalam program bimbingan akademik intensif atau tutor sebaya sebelum memasuki semester 2 untuk mencegah *burnout*.

4. **Penyediaan Fleksibilitas Waktu Studi:** Mempertimbangkan tingginya kebutuhan mahasiswa untuk bekerja demi menutupi beban biaya kuliah, institusi disarankan menawarkan opsi pemindahan jadwal ke kelas malam (*Evening Attendance*) sehingga mahasiswa tidak terpaksa melakukan *dropout*.