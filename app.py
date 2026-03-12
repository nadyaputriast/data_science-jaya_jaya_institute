import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

# Import dari file terpisah yang baru dibuat
from expert_system import render_recommendation
from mappings import course_dict, nacionality_dict, app_mode_dict, prev_qual_dict, feature_names_dict, macro_data_dict, tooltips_dict, parents_edu_dict, parents_job_dict

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Jaya Jaya Institute", page_icon="🎓", layout="wide")

# 2. FUNGSI LOAD MODEL
@st.cache_resource
def load_models():
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    return xgb_model, scaler, encoder

model, scaler, encoder = load_models()

# 3. ANTARMUKA PENGGUNA (UI)
st.title("🎓 Sistem Prediksi Status Mahasiswa")
st.markdown("Prediksi potensi **Dropout**, **Enrolled (Aktif)**, atau **Graduate (Lulus)** dari data riwayat mahasiswa.")

tab1, tab2 = st.tabs(["📊 Dashboard Analisis Performa", "🔮 Sistem Prediksi Mahasiswa"])

with tab1:
    st.subheader("Dashboard Interaktif Jaya Jaya Institute")
    st.components.v1.html(
        """
        <iframe 
            width="100%" 
            height="850" 
            src="https://lookerstudio.google.com/embed/reporting/fde1f490-53d0-4586-be22-f9a4ea9108ec/page/p_ezgcd56n1d" 
            frameborder="0" 
            style="border:0;" 
            allowfullscreen>
        </iframe>
        """,
        height=850
    )

with tab2:
    # --- Kelompok 1: Demografi ---
    st.subheader("1. Data Demografi & Latar Belakang")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_at_enrollment = st.number_input("Usia saat mendaftar", 15, 60, 20, help=tooltips_dict['Age_at_enrollment'])
        gender = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan", help=tooltips_dict['Gender'])
        marital_status = st.selectbox("Status Pernikahan", [1,2,3,4,5,6], format_func=lambda x: {1:'Belum Menikah', 2:'Menikah', 3:'Duda/Janda', 4:'Bercerai', 5:'Kumpul Kebo', 6:'Berpisah Sah'}[x], help=tooltips_dict['Marital_status'])
    with col2:
        nacionality = st.selectbox("Kewarganegaraan", list(nacionality_dict.keys()), format_func=lambda x: nacionality_dict[x], help=tooltips_dict['Nacionality'])
        international = st.selectbox("Internasional?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", help=tooltips_dict['International'])
        displaced = st.selectbox("Status Pendatang?", [0, 1], format_func=lambda x: "Ya (Pendatang)" if x==1 else "Bukan Pendatang", help=tooltips_dict['Displaced'])
    with col3:
        mothers_qualification = st.selectbox("Pendidikan Ibu", list(parents_edu_dict.keys()), format_func=lambda x: parents_edu_dict[x], help=tooltips_dict['Mothers_qualification'])
        fathers_qualification = st.selectbox("Pendidikan Ayah", list(parents_edu_dict.keys()), format_func=lambda x: parents_edu_dict[x], help=tooltips_dict['Fathers_qualification'])
        mothers_occupation = st.selectbox("Pekerjaan Ibu", list(parents_job_dict.keys()), format_func=lambda x: parents_job_dict[x], help=tooltips_dict['Mothers_occupation'])
        # CATATAN: Fathers_occupation DIHAPUS dari input karena dibuang saat training

    # --- Kelompok 2: Akademik Sebelum & Pendaftaran ---
    st.subheader("2. Data Akademik Sebelumnya & Pendaftaran")
    col4, col5, col6 = st.columns(3)
    with col4:
        previous_qualification = st.selectbox("Pend. Sebelumnya", list(prev_qual_dict.keys()), format_func=lambda x: prev_qual_dict[x], help=tooltips_dict['Previous_qualification'])
        previous_qualification_grade = st.number_input("Nilai Pend. Sebelumnya", 0.0, 200.0, 130.0, help=tooltips_dict['Previous_qualification_grade'])
        admission_grade = st.number_input("Nilai Masuk Ujian", 0.0, 200.0, 125.0, help=tooltips_dict['Admission_grade'])
    with col5:
        application_mode = st.selectbox("Jalur Pendaftaran", list(app_mode_dict.keys()), format_func=lambda x: app_mode_dict[x], help=tooltips_dict['Application_mode'])
        application_order = st.number_input("Urutan Pilihan Prodi (0-9)", 0, 9, 1, help=tooltips_dict['Application_order'])
        course = st.selectbox("Program Studi", list(course_dict.keys()), format_func=lambda x: course_dict[x], help=tooltips_dict['Course'])
    with col6:
        daytime_evening_attendance = st.selectbox("Waktu Kuliah", [1, 0], format_func=lambda x: "Kelas Siang" if x==1 else "Kelas Malam", help=tooltips_dict['Daytime_evening_attendance'])
        educational_special_needs = st.selectbox("Kebutuhan Khusus?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", help=tooltips_dict['Educational_special_needs'])

    # --- Kelompok 3: Finansial & Makro ---
    st.subheader("3. Kondisi Finansial & Makroekonomi")
    col7, col8 = st.columns([1.5, 2])

    with col7:
        st.markdown("**Status Keuangan Mahasiswa:**")
        tuition_fees_up_to_date = st.selectbox("SPP Lancar?", [1, 0], format_func=lambda x: "Lancar" if x==1 else "Menunggak", help=tooltips_dict['Tuition_fees_up_to_date'])
        debtor = st.selectbox("Status Berutang?", [0, 1], format_func=lambda x: "Ya (Berutang)" if x==1 else "Tidak", help=tooltips_dict['Debtor'])
        scholarship_holder = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", help=tooltips_dict['Scholarship_holder'])

    with col8:
        st.markdown("**Kondisi Ekonomi Makro Saat Mendaftar:**")
        selected_year = st.selectbox("Pilih Tahun / Era Pendaftaran:", list(macro_data_dict.keys()), help=tooltips_dict['Macro_Preset'])

        auto_unemp = macro_data_dict[selected_year]['unemployment']
        auto_inf = macro_data_dict[selected_year]['inflation']
        auto_gdp = macro_data_dict[selected_year]['gdp']

        is_disabled = selected_year != 'Input Manual (Kustom)'

        col8_1, col8_2, col8_3 = st.columns(3)
        with col8_1:
            unemployment_rate = st.number_input("Pengangguran (%)", value=float(auto_unemp), disabled=is_disabled, key=f"unemp_{selected_year}")
        with col8_2:
            inflation_rate = st.number_input("Inflasi (%)", value=float(auto_inf), disabled=is_disabled, key=f"inf_{selected_year}")
        with col8_3:
            gdp = st.number_input("PDB / GDP (%)", value=float(auto_gdp), disabled=is_disabled, key=f"gdp_{selected_year}")

    # --- Kelompok 4: Akademik Kuliah ---
    st.subheader("4. Performa Akademik Kuliah")
    col10, col11, col12, col13 = st.columns(4)
    with col10:
        st.markdown("**Semester 1**")
        sem1_credited = st.number_input("SKS Diakui (Semester 1)", value=0, help=tooltips_dict['Curricular_units_credited'])
        sem1_enrolled = st.number_input("SKS Diambil (Semester 1)", value=6, help=tooltips_dict['Curricular_units_enrolled'])
        sem1_evaluations = st.number_input("Jml Evaluasi (Semester 1)", value=6, help=tooltips_dict['Curricular_units_evaluations'])
    with col11:
        st.markdown("**Semester 1 (Lanjut)**")
        sem1_approved = st.number_input("SKS Lulus (Semester 1)", value=6, help=tooltips_dict['Curricular_units_approved'])
        sem1_grade = st.number_input("Rata-rata Nilai (Semester 1)", 0.0, 20.0, 14.0, help=tooltips_dict['Curricular_units_grade'])
        sem1_without_eval = st.number_input("SKS Tanpa Evaluasi (Semester 1)", value=0, help=tooltips_dict['Curricular_units_without_evaluations'])
    with col12:
        st.markdown("**Semester 2**")
        # CATATAN: sem2_credited, sem2_enrolled, sem2_approved DIHAPUS dari UI
        # karena kolom tersebut dibuang saat training model
        sem2_evaluations = st.number_input("Jml Evaluasi (Semester 2)", value=6, help=tooltips_dict['Curricular_units_evaluations'])
        sem2_grade = st.number_input("Rata-rata Nilai (Semester 2)", 0.0, 20.0, 13.5, help=tooltips_dict['Curricular_units_grade'])
        sem2_without_eval = st.number_input("SKS Tanpa Evaluasi (Semester 2)", value=0, help=tooltips_dict['Curricular_units_without_evaluations'])
    with col13:
        st.markdown("**ℹ️ Info Semester 2**")
        st.info(
            "Kolom SKS Diakui, SKS Diambil, dan SKS Lulus Semester 2 "
            "tidak digunakan oleh model karena terdeteksi redundan "
            "dengan data Semester 1 saat proses training."
        )

    # ============================================================
    # 4. PREDIKSI & XAI
    # PENTING: Urutan kolom harus SAMA PERSIS dengan saat training!
    # Kolom yang dibuang saat training:
    #   - Fathers_occupation
    #   - Curricular_units_2nd_sem_credited
    #   - Curricular_units_2nd_sem_enrolled
    #   - Curricular_units_2nd_sem_approved
    # ============================================================
    input_data = pd.DataFrame({
        'Marital_status':                              [marital_status],
        'Application_mode':                            [application_mode],
        'Application_order':                           [application_order],
        'Course':                                      [course],
        'Daytime_evening_attendance':                  [daytime_evening_attendance],
        'Previous_qualification':                      [previous_qualification],
        'Previous_qualification_grade':                [previous_qualification_grade],
        'Nacionality':                                 [nacionality],
        'Mothers_qualification':                       [mothers_qualification],
        'Fathers_qualification':                       [fathers_qualification],
        'Mothers_occupation':                          [mothers_occupation],
        # Fathers_occupation → TIDAK ADA (dibuang saat training)
        'Admission_grade':                             [admission_grade],
        'Displaced':                                   [displaced],
        'Educational_special_needs':                   [educational_special_needs],
        'Debtor':                                      [debtor],
        'Tuition_fees_up_to_date':                     [tuition_fees_up_to_date],
        'Gender':                                      [gender],
        'Scholarship_holder':                          [scholarship_holder],
        'Age_at_enrollment':                           [age_at_enrollment],
        'International':                               [international],
        'Curricular_units_1st_sem_credited':           [sem1_credited],
        'Curricular_units_1st_sem_enrolled':           [sem1_enrolled],
        'Curricular_units_1st_sem_evaluations':        [sem1_evaluations],
        'Curricular_units_1st_sem_approved':           [sem1_approved],
        'Curricular_units_1st_sem_grade':              [sem1_grade],
        'Curricular_units_1st_sem_without_evaluations':[sem1_without_eval],
        # Curricular_units_2nd_sem_credited  → TIDAK ADA (dibuang saat training)
        # Curricular_units_2nd_sem_enrolled  → TIDAK ADA (dibuang saat training)
        'Curricular_units_2nd_sem_evaluations':        [sem2_evaluations],
        # Curricular_units_2nd_sem_approved  → TIDAK ADA (dibuang saat training)
        'Curricular_units_2nd_sem_grade':              [sem2_grade],
        'Curricular_units_2nd_sem_without_evaluations':[sem2_without_eval],
        'Unemployment_rate':                           [unemployment_rate],
        'Inflation_rate':                              [inflation_rate],
        'GDP':                                         [gdp],
    })

    # Validasi jumlah kolom sebelum prediksi
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    if expected_features is not None:
        missing = set(expected_features) - set(input_data.columns)
        extra = set(input_data.columns) - set(expected_features)
        if missing or extra:
            st.error(f"❌ Mismatch kolom! Kurang: {missing} | Lebih: {extra}")
            st.stop()
        # Pastikan urutan kolom sama dengan saat training
        input_data = input_data[expected_features]

    # Prediksi XGBoost        
    input_scaled = scaler.transform(input_data)
    prediction_encoded = model.predict(input_scaled)
    prediction_label = encoder.inverse_transform(prediction_encoded)[0]

    # XAI (SHAP)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    pred_idx = prediction_encoded[0]

    class_shap_values = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0, :, pred_idx]
    top_3_indices = np.argsort(np.abs(class_shap_values))[::-1][:3]
    features_list = input_data.columns.tolist()
    top_features = [features_list[idx] for idx in top_3_indices]

    # UI Hasil
    st.markdown("---")
    st.subheader("💡 Hasil Prediksi (Live AI Update):")

    if prediction_label == 'Dropout':
        st.error(f"⚠️ Risiko Mahasiswa: **{prediction_label.upper()}**")
    elif prediction_label == 'Graduate':
        st.success(f"🎓 Status Mahasiswa: **{prediction_label.upper()}**")
    else:
        st.warning(f"⏳ Status Mahasiswa: **{prediction_label.upper()}**")

    st.markdown(f"**🧠 Kenapa AI mengklasifikasikan mahasiswa ini sebagai {prediction_label}?**")
    st.write("Berdasarkan analisis algoritma XAI (SHAP), berikut adalah **3 faktor utama** pendorong keputusan tersebut:")

    for i, idx in enumerate(top_3_indices):
        raw_feature_name = features_list[idx]
        clean_feature_name = feature_names_dict.get(raw_feature_name, raw_feature_name)
        st.markdown(f"{i+1}. **{clean_feature_name}**")

    # Panggil fungsi rekomendasi dari file expert_system.py
    render_recommendation(prediction_label, top_features, input_data)