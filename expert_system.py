import streamlit as st

def render_recommendation(prediction_label, top_features, input_data):
    # Ekstrak data aktual yang diinputkan di web
    data = input_data.iloc[0]
    
    # Diagnosa Kondisi Aktual Mahasiswa
    is_tuition_late = data['Tuition_fees_up_to_date'] == 0
    is_debtor = data['Debtor'] == 1
    no_scholarship = data['Scholarship_holder'] == 0
    
    # Cek apakah nilai akademiknya memang jelek (di bawah 10) atau ada SKS tidak lulus
    academic_struggle = data['Curricular_units_1st_sem_grade'] < 10 or data['Curricular_units_2nd_sem_grade'] < 10 or (data['Curricular_units_1st_sem_approved'] < data['Curricular_units_1st_sem_enrolled'])

    # Cek kategori dari Top 3 XAI
    has_financial = any(f in ['Tuition_fees_up_to_date', 'Debtor', 'Scholarship_holder'] for f in top_features)
    has_academic = any('Curricular_units' in f for f in top_features) or any(f in ['Admission_grade', 'Previous_qualification_grade'] for f in top_features)
    has_demographic = any(f in ['Age_at_enrollment', 'Gender', 'Marital_status', 'Displaced', 'Daytime_evening_attendance', 'Application_mode', 'Nacionality', 'International'] for f in top_features)

    st.markdown("---")
    st.markdown("**📋 Rekomendasi Tindakan (Action Items):**")
    
    # JIKA PREDIKSI: DROPOUT
    if prediction_label == 'Dropout':
        if has_financial and has_academic:
            st.error("🚨 **Krisis Ganda (Finansial & Akademik):** Mahasiswa terdeteksi berisiko tinggi karena kombinasi faktor nilai dan keuangan.")
            
            # Saran dinamis sesuai data aslinya!
            fin_advice = "Tawarkan skema cicilan SPP / penyelesaian utang." if (is_tuition_late or is_debtor) else "Karena SPP-nya lancar, tawarkan beasiswa untuk meringankan bebannya."
            acad_advice = "Wajibkan kelas remedial/tutor sebaya untuk memperbaiki nilai yang tertinggal." if academic_struggle else "Lakukan konseling akademik untuk mencegah hilangnya motivasi."
            
            st.info(f"💡 **Saran:** {fin_advice} Selain itu, {acad_advice}")
            
        elif has_financial:
            st.error("💸 **Faktor Finansial Mendominasi:** Masalah keuangan menjadi pendorong utama potensi dropout.")
            if is_tuition_late or is_debtor:
                st.info("💡 **Saran:** Segera panggil mahasiswa untuk diskusi restrukturisasi biaya (cicilan/penundaan bayar).")
            elif no_scholarship:
                st.info("💡 **Saran:** Walaupun SPP lunas, ketiadaan beasiswa membebani mahasiswa ini. Prioritaskan dalam program 'Beasiswa Darurat'.")
            else:
                st.info("💡 **Saran:** Lakukan wawancara finansial 1-on-1 untuk memastikan kondisi ekonominya stabil.")
                
        elif has_academic:
            st.error("📚 **Krisis Akademik Tunggal:** Mahasiswa sangat kesulitan mengikuti perkuliahan.")
            if academic_struggle:
                st.info("💡 **Saran:** Peringatan Dini Akademik! Berikan surat peringatan dan pendampingan belajar ekstra.")
            else:
                st.info("💡 **Saran:** Nilainya tampak wajar, namun sistem mendeteksi risiko akademik. Lakukan konseling agar dia tidak burnout.")
                
        elif has_demographic:
            st.error("👤 **Kendala Personal / Demografi:** Risiko murni dipicu kuat oleh faktor usia, jalur pendaftaran, atau domisili.")
            st.info("💡 **Saran:** Lakukan sesi *Stay Interview* untuk menggali hambatan personal mahasiswa (misal: kendala adaptasi, *homesick*, atau beban kerja luar).")
        else:
            st.info("💡 **Saran:** Lakukan konseling mendalam dengan fokus pada 3 faktor XAI yang disebutkan di atas.")

    # JIKA PREDIKSI: GRADUATE
    elif prediction_label == 'Graduate':
        if has_financial and has_academic:
            st.success("🌟 **Prestasi & Finansial Stabil:** Nilai prima ditunjang kuat oleh kelancaran biaya.")
            st.info("💡 **Saran:** Berikan apresiasi dan pastikan kelancaran administrasinya hingga lulus.")
        elif has_academic:
            st.success("📚 **Kekuatan Akademik Dominan:** Performa belajar yang konsisten menjadi motor kelulusan.")
            st.info("💡 **Saran:** Pertimbangkan untuk merekrut mahasiswa ini menjadi asisten laboratorium/dosen.")
        elif has_financial:
            st.success("💰 **Dukungan Finansial Terjamin:** Kondisi finansial yang sehat membantu mahasiswa fokus berkuliah.")
            st.info("💡 **Saran:** Jaga agar proses daftar ulang di semester-semester berikutnya tidak merepotkan mahasiswa.")
        else:
            st.success("👤 **Profil Kuat:** Mahasiswa memiliki rekam jejak yang sangat stabil.")
            st.info("💡 **Saran:** Pertahankan fasilitas dan dukungan kampus untuk mahasiswa ini.")