"""Build V2 corpus: mixed Indonesian + code for multi-purpose training.

Steps:
1. Read existing Indonesian corpus
2. Fetch Wikipedia articles (govt, finance, economy, business, tech)
3. Add hardcoded domain texts
4. Add Python code snippets with Indonesian comments
5. Train BPE tokenizer on mixed corpus
6. Tokenize and save
"""

import sys, os, json, pickle, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'indonesian_corpus_v2')
os.makedirs(DATA_DIR, exist_ok=True)

from anggira.bpe_tokenizer import BPETokenizer

# ── 1. Read existing Indonesian corpus ──
print("═" * 50)
print("📚 Building V2 Mixed Corpus")
print("═" * 50)

print("\n1️⃣  Reading existing Indonesian corpus...")
v1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       'data', 'indonesian_corpus', 'indonesian_corpus.txt')
existing_text = ""
if os.path.exists(v1_path):
    with open(v1_path, 'r', encoding='utf-8') as f:
        existing_text = f.read()
    print(f"   ✅ {len(existing_text):,} chars loaded from V1 corpus")
else:
    print(f"   ⚠️  V1 corpus not found at {v1_path}")

# ── 2. Fetch current Wikipedia articles ──
print("\n2️⃣  Fetching Wikipedia articles...")
import urllib.request, urllib.error

def wiki_extract(title):
    url = (f"https://id.wikipedia.org/w/api.php?action=query&prop=extracts"
           f"&explaintext=&titles={urllib.request.quote(title)}&format=json&formatversion=2")
    req = urllib.request.Request(url, headers={'User-Agent': 'AnggiraAI/2.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        pages = data['query']['pages']
        return pages[0].get('extract', '') if pages else ''
    except Exception as e:
        print(f"  ⚠️  Error {title}: {e}")
        return ''

wiki_topics = [
    "Ekonomi_Indonesia", "Prabowo_Subianto", "Gibran_Rakabuming_Raka",
    "Kabinet_Merah_Putih", "Perbankan_di_Indonesia", "Internet_di_Indonesia",
    "Energi_terbarukan_di_Indonesia", "Pertanian_di_Indonesia",
    "Pariwisata_di_Indonesia", "Transportasi_di_Indonesia",
    "Pendidikan_di_Indonesia", "Kesehatan_di_Indonesia",
    "Industri_di_Indonesia", "Telekomunikasi_di_Indonesia",
    "Olahraga_di_Indonesia", "Budaya_Indonesia",
    "Sains_dan_teknologi_di_Indonesia", "Jakarta",
    "Jawa_Barat", "Jawa_Timur", "Bali",
    "Sejarah_Indonesia", "Indonesia",
]

wiki_texts = []
for topic in wiki_topics:
    text = wiki_extract(topic)
    if text and len(text) > 200:
        wiki_texts.append(f"\n=== {topic.replace('_', ' ')} ===\n{text}")
        print(f"   ✅ {topic}: {len(text):,} chars")
    else:
        print(f"   ⚠️  {topic}: too short ({len(text)} chars) or empty")

total_wiki = sum(len(t) for t in wiki_texts)
print(f"   Total wiki: {total_wiki:,} chars")

# ── 3. Hardcoded domain texts ──
print("\n3️⃣  Adding hardcoded domain texts...")

hardcoded = []

# Finance
hardcoded.append("""
=== Sistem Keuangan dan Perbankan Indonesia ===
Otoritas Jasa Keuangan (OJK) adalah lembaga independen yang mengawasi sektor jasa keuangan di Indonesia. OJK didirikan pada tahun 2011 berdasarkan Undang-Undang Nomor 21 Tahun 2011 tentang Otoritas Jasa Keuangan. Tugas OJK meliputi pengaturan dan pengawasan terhadap kegiatan jasa keuangan di sektor perbankan, pasar modal, perasuransian, dana pensiun, lembaga pembiayaan, dan lembaga jasa keuangan lainnya.
Bank Indonesia adalah bank sentral Republik Indonesia yang bertanggung jawab menjaga stabilitas moneter dan sistem keuangan. Tujuan utama Bank Indonesia adalah mencapai dan memelihara stabilitas nilai rupiah. Bank Indonesia memiliki wewenang dalam menetapkan suku bunga acuan yang dikenal sebagai BI Rate, yang kemudian berkembang menjadi BI 7-Day Reverse Repo Rate.
Perbankan Indonesia terdiri dari bank umum, bank syariah, dan bank perkreditan rakyat. Bank-bank besar di Indonesia antara lain Bank Mandiri, Bank Central Asia (BCA), Bank Negara Indonesia (BNI), Bank Rakyat Indonesia (BRI), dan Bank Tabungan Negara (BTN). Selain itu, terdapat pula bank-bank swasta seperti Bank Danamon, Bank Permata, dan Bank CIMB Niaga.
Perbankan syariah di Indonesia berkembang pesat sejak diberlakukannya Undang-Undang Perbankan Syariah. Bank syariah terbesar di Indonesia adalah Bank Syariah Indonesia (BSI) yang merupakan hasil merger tiga bank syariah milik negara. Prinsip perbankan syariah didasarkan pada hukum Islam, mengharamkan bunga (riba), dan menggunakan sistem bagi hasil (mudharabah) dan jual-beli (murabahah).
Pasar modal Indonesia diatur dan diawasi oleh OJK. Bursa Efek Indonesia (BEI) adalah bursa saham utama di Indonesia. Indeks harga saham gabungan (IHSG) adalah indeks yang mengukur kinerja seluruh saham yang tercatat di BEI. Selain itu, terdapat LQ45 yang mengukur kinerja 45 saham paling likuid dan berkapitalisasi besar.
Industri fintech di Indonesia tumbuh pesat didukung oleh penetrasi internet dan smartphone yang semakin tinggi. Perusahaan fintech terkemuka di Indonesia antara lain GoPay, OVO, Dana, dan LinkAja. Selain itu, terdapat platform pinjaman online seperti Kredivo, Akulaku, dan fintech syariah. Regulasi fintech di Indonesia diatur oleh OJK dan Bank Indonesia melalui berbagai peraturan tentang layanan pinjaman berbasis teknologi informasi dan uang elektronik.

=== Perekonomian Indonesia ===
Indonesia merupakan negara dengan perekonomian terbesar di Asia Tenggara dan terbesar ke-16 di dunia berdasarkan produk domestik bruto (PDB). PDB Indonesia pada tahun 2024 mencapai sekitar 21.000 triliun rupiah atau sekitar 1,37 triliun dolar AS. Pertumbuhan ekonomi Indonesia berkisar antara 4,5 hingga 5,3 persen per tahun pada periode 2022 hingga 2025 pasca-pandemi COVID-19.
Struktur perekonomian Indonesia didominasi oleh sektor industri pengolahan, perdagangan, pertanian, pertambangan, dan konstruksi. Sektor industri pengolahan menyumbang sekitar 19 persen dari PDB, menjadikannya sektor terbesar. Industri manufaktur utama Indonesia meliputi industri makanan dan minuman, industri kimia, farmasi, industri logam, elektronik, dan industri otomotif.
Indonesia adalah negara penghasil kelapa sawit terbesar di dunia, dengan produksi mencapai lebih dari 45 juta ton per tahun. Selain itu, Indonesia juga merupakan produsen utama batu bara, nikel, tembaga, timah, dan karet alam. Sektor pertambangan menyumbang sekitar 8 persen dari PDB dan merupakan sumber utama pendapatan ekspor.
Kebijakan fiskal Indonesia dikelola oleh Kementerian Keuangan. Anggaran Pendapatan dan Belanja Negara (APBN) adalah instrumen kebijakan fiskal utama. Pendapatan negara bersumber dari penerimaan pajak, penerimaan negara bukan pajak (PNBP), dan hibah. Pajak penghasilan (PPh), pajak pertambahan nilai (PPN), dan pajak bumi dan bangunan (PBB) merupakan sumber penerimaan pajak utama.
Investasi di Indonesia diatur oleh Badan Koordinasi Penanaman Modal (BKPM). Pemerintah Indonesia terus berupaya meningkatkan iklim investasi melalui berbagai kebijakan, termasuk Undang-Undang Cipta Kerja yang disahkan pada tahun 2020. Tujuan UU Cipta Kerja adalah menyederhanakan regulasi dan mendorong investasi serta penciptaan lapangan kerja.

=== Bisnis dan Industri di Indonesia ===
Konglomerasi bisnis di Indonesia dikuasai oleh beberapa kelompok usaha besar. Kelompok usaha terbesar di Indonesia antara lain Grup Astra, Grup Salim, Grup Lippo, Grup Djarum, Grup Sinar Mas, Grup Bakrie, Grup MNC, dan Grup CT Corp. Grup Astra adalah salah satu konglomerat terbesar di Indonesia dengan bisnis di otomotif, alat berat, pertambangan, agribisnis, infrastruktur, dan jasa keuangan.
E-commerce di Indonesia mengalami pertumbuhan yang sangat pesat. Platform e-commerce terbesar di Indonesia adalah Shopee, Tokopedia, Lazada, Bukalapak, dan Blibli. Nilai pasar e-commerce Indonesia diperkirakan mencapai lebih dari 100 miliar dolar AS pada tahun 2025. Pertumbuhan e-commerce didorong oleh meningkatnya penetrasi internet, penggunaan ponsel pintar, dan perubahan perilaku belanja masyarakat.
UMKM atau Usaha Mikro, Kecil, dan Menengah merupakan tulang punggung perekonomian Indonesia. UMKM menyumbang sekitar 60 persen dari PDB dan menyerap sekitar 97 persen tenaga kerja. Pemerintah Indonesia mendukung UMKM melalui program Kredit Usaha Rakyat (KUR), pelatihan kewirausahaan, dan digitalisasi usaha.
Persaingan bisnis di Indonesia diatur oleh Komisi Pengawas Persaingan Usaha (KPPU). KPPU bertugas mengawasi agar praktik monopoli dan persaingan usaha tidak sehat tidak terjadi. Undang-Undang Nomor 5 Tahun 1999 tentang Larangan Praktik Monopoli dan Persaingan Usaha Tidak Sehat menjadi dasar hukum KPPU.

=== Teknologi Digital di Indonesia ===
Revolusi digital di Indonesia ditandai dengan penetrasi internet yang mencapai lebih dari 79 persen populasi pada tahun 2024 atau sekitar 221 juta pengguna. Penggunaan ponsel pintar mencapai lebih dari 190 juta pengguna. Indonesia adalah salah satu pasar digital terbesar di Asia Tenggara dengan ekonomi digital diperkirakan mencapai 130 miliar dolar AS pada tahun 2025.
Ekonomi digital Indonesia didominasi oleh sektor e-commerce, transportasi daring, layanan keuangan digital, dan media digital. Perusahaan teknologi Indonesia yang menjadi unicorn antara lain Gojek, Tokopedia, Traveloka, Bukalapak, OVO, dan Ajaib. Decacorn Indonesia adalah GoTo (hasil merger Gojek dan Tokopedia) yang terdaftar di Bursa Efek Indonesia.
Pemerintah Indonesia mendorong transformasi digital melalui berbagai inisiatif seperti Gerakan Nasional 1000 Startup Digital, program Digital Talent Scholarship, dan pengembangan infrastruktur telekomunikasi seperti Palapa Ring. Palapa Ring adalah proyek serat optik nasional yang menghubungkan seluruh wilayah Indonesia termasuk daerah terpencil.
Perkembangan kecerdasan buatan (AI) di Indonesia semakin pesat. Berbagai universitas dan perusahaan mengembangkan riset dan aplikasi AI untuk sektor kesehatan, pertanian, pendidikan, dan pelayanan publik. Pemerintah Indonesia melalui Kementerian Komunikasi dan Informatika mengeluarkan strategi nasional AI untuk mendorong adopsi AI di berbagai sektor.
Teknologi 5G mulai diimplementasikan di Indonesia pada tahun 2021. Operator seluler seperti Telkomsel, Indosat Ooredoo Hutchison, dan XL Axiata telah meluncurkan layanan 5G di kota-kota besar. Teknologi 5G diharapkan dapat mendorong perkembangan Internet of Things (IoT), smart city, dan industri 4.0 di Indonesia.
Startup dan ekosistem venture capital di Indonesia terus berkembang. Modal ventura asing dan domestik aktif berinvestasi di startup Indonesia. Sektor yang paling menarik bagi investor antara lain fintech, logistik, edtech, healthtech, dan agritech. Beberapa startup Indonesia telah menjadi perusahaan publik melalui pencatatan saham di BEI.

=== Berita Terkini Indonesia 2024-2025 ===
Pemilihan umum Indonesia 2024 dilaksanakan pada tanggal 14 Februari 2024. Pasangan Prabowo Subianto dan Gibran Rakabuming Raka memenangkan pilpres dengan perolehan suara sekitar 58,6 persen, mengalahkan pasangan Anies Baswedan-Muhaimin Iskandar dan Ganjar Pranowo-Mahfud MD. Prabowo Subianto dilantik sebagai Presiden Indonesia kedelapan pada 20 Oktober 2024, menggantikan Joko Widodo yang telah menjabat selama dua periode.
Kabinet Merah Putih diumumkan oleh Presiden Prabowo pada Oktober 2024. Kabinet ini terdiri dari 48 kementerian dan lembaga, termasuk beberapa kementerian baru seperti Kementerian Ekonomi Kreatif, Kementerian Teknologi dan Inovasi, serta Kementerian Hilirisasi. Wakil Presiden Gibran Rakabuming Raka adalah putra sulung dari mantan Presiden Joko Widodo.
Program prioritas pemerintahan Prabowo-Gibran meliputi program makan bergizi gratis untuk anak sekolah, pembangunan Ibu Kota Nusantara (IKN), hilirisasi sumber daya alam, swasembada pangan dan energi, serta digitalisasi layanan publik. Program makan bergizi gratis merupakan program unggulan dengan target menjangkau lebih dari 80 juta penerima.
Pembangunan Ibu Kota Nusantara (IKN) di Kalimantan Timur terus berlanjut. IKN dirancang sebagai kota pintar (smart city) berbasis hutan dan ramah lingkungan. Pemindahan ibu negara dari Jakarta ke IKN direncanakan bertahap dengan target tahap awal pada tahun 2028. Pembangunan IKN menuai pro dan kontra, dengan kekhawatiran utama terkait pendanaan dan dampak lingkungan.
Inflasi Indonesia pada periode 2024-2025 berada dalam kisaran 2,5 hingga 4 persen, relatif terkendali dibandingkan negara lain. Bank Indonesia mempertahankan suku bunga acuan di level 5,75 hingga 6,25 persen untuk menjaga stabilitas nilai tukar rupiah. Nilai tukar rupiah terhadap dolar AS bergerak di kisaran 15.500 hingga 16.000 rupiah per dolar.
Perekonomian Indonesia menghadapi tantangan dari perlambatan ekonomi global, penurunan harga komoditas, dan ketidakpastian geopolitik. Namun, konsumsi domestik yang kuat dan investasi yang terus mengalir menjadi pendorong utama pertumbuhan. Pemerintah menargetkan pertumbuhan ekonomi 5,2 persen pada tahun 2025.

=== Infrastruktur Indonesia ===
Pembangunan infrastruktur menjadi prioritas utama pemerintahan Indonesia. Jalan tol Trans-Jawa telah tersambung dari Merak hingga Banyuwangi dengan total panjang sekitar 1.200 kilometer. Jalan tol Trans-Sumatra juga terus dibangun dan akan menghubungkan seluruh provinsi di Pulau Sumatera. Pembangunan infrastruktur jalan tol bertujuan untuk memperlancar distribusi logistik dan konektivitas antar daerah.
Transportasi umum massal di Indonesia terus dikembangkan. Moda Raya Terpadu (MRT) Jakarta telah beroperasi sejak 2019 dan terus diperpanjang jaringannya. Lintas Raya Terpadu (LRT) beroperasi di Jakarta dan Palembang. Kereta Cepat Jakarta-Bandung (Whoosh) mulai beroperasi pada Oktober 2023 dan merupakan kereta cepat pertama di Asia Tenggara.
Proyek strategis nasional (PSN) mencakup berbagai sektor infrastruktur termasuk bendungan, irigasi, pelabuhan, bandara, dan kawasan industri. Pembangunan bendungan baru bertujuan untuk meningkatkan kapasitas irigasi dan penyediaan air baku. Pengembangan pelabuhan seperti Pelabuhan Patimban dan Pelabuhan Kuala Tanjung bertujuan untuk meningkatkan daya saing ekspor Indonesia.

=== Energi dan Sumber Daya Alam Indonesia ===
Indonesia kaya akan sumber daya alam termasuk minyak bumi, gas alam, batu bara, nikel, tembaga, emas, dan timah. Indonesia adalah produsen nikel terbesar di dunia, yang menjadi kunci dalam produksi baterai kendaraan listrik. Hilirisasi nikel menjadi kebijakan strategis pemerintah untuk meningkatkan nilai tambah di dalam negeri.
Transisi energi menjadi isu penting di Indonesia. Pemerintah menargetkan mencapai Net Zero Emission pada tahun 2060. Pengembangan energi terbarukan meliputi tenaga surya, angin, panas bumi, dan hidro. Indonesia memiliki potensi energi panas bumi terbesar di dunia, namun pemanfaatannya masih terbatas.
Konflik agraria dan lingkungan terkait sumber daya alam masih terjadi di berbagai daerah. Isu deforestasi dan kerusakan lingkungan akibat pertambangan dan perkebunan kelapa sawit menjadi perhatian. Pemerintah berupaya menyeimbangkan pembangunan ekonomi dengan kelestarian lingkungan melalui program-program keberlanjutan.

=== Ketenagakerjaan Indonesia ===
Jumlah angkatan kerja Indonesia sekitar 140 juta orang. Tingkat pengangguran terbuka pada tahun 2024 berkisar antara 4,8 hingga 5,2 persen. Sektor informal masih menyerap sekitar 56 persen tenaga kerja. Upah minimum provinsi (UMP) ditetapkan setiap tahun oleh pemerintah daerah dengan mempertimbangkan inflasi dan pertumbuhan ekonomi.
Undang-Undang Cipta Kerja yang disahkan pada 2020 dan diperbaharui pada 2023 melalui Perppu Cipta Kerja bertujuan menciptakan lapangan kerja dan menyederhanakan regulasi ketenagakerjaan. Perubahan dalam UU ini meliputi pengaturan upah, pesangon, dan jam kerja yang lebih fleksibel. UU ini menuai kontroversi dan protes dari serikat pekerja yang khawatir akan hak-hak buruh.
""")

hardcoded.append("""
=== Python untuk Data Science ===
# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membaca data dari file CSV
data = pd.read_csv('data_penjualan.csv')
print(f"Dataset memiliki {data.shape[0]} baris dan {data.shape[1]} kolom")

# Membersihkan data dengan menghapus nilai yang hilang
data = data.dropna()

# Memisahkan fitur dan target
X = data[['harga', 'promosi', 'lokasi']].values
y = data['penjualan'].values

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi data testing
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Menampilkan koefisien model
for i, col in enumerate(['harga', 'promosi', 'lokasi']):
    print(f"Koefisien {col}: {model.coef_[i]:.4f}")

=== Python Dasar untuk Pemula ===
# Program pertama: Halo Dunia
print("Halo Dunia!")

# Variabel dan tipe data
nama = "Budi"
umur = 25
tinggi = 170.5
is_student = True

print(f"Nama: {nama}, Umur: {umur}, Tinggi: {tinggi} cm")

# List dan perulangan
buah_buahan = ["apel", "pisang", "jeruk", "mangga", "durian"]
for buah in buah_buahan:
    print(f"Saya suka {buah}")

# Dictionary
siswa = {
    "nama": "Ani",
    "kelas": "10A",
    "nilai": {
        "matematika": 85,
        "bahasa_indonesia": 90,
        "ipa": 88
    }
}
print(f"Siswa {siswa['nama']} mendapat nilai matematika {siswa['nilai']['matematika']}")

# Fungsi
def hitung_rata_rata(nilai_list):
    total = sum(nilai_list)
    return total / len(nilai_list)

nilai_ani = [85, 90, 88, 92, 78]
rata = hitung_rata_rata(nilai_ani)
print(f"Rata-rata nilai: {rata:.2f}")

=== Machine Learning dengan Python ===
# Klasifikasi sederhana dengan K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Melatih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluasi
accuracy = knn.score(X_test, y_test)
print(f"Akurasi model KNN: {accuracy:.2%}")

# Memprediksi sampel baru
sampel_baru = [[5.1, 3.5, 1.4, 0.2]]
sampel_scaled = scaler.transform(sampel_baru)
prediksi = knn.predict(sampel_scaled)
print(f"Kelas prediksi: {iris.target_names[prediksi[0]]}")

=== Neural Network Sederhana dengan NumPy ===
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

# Inisialisasi bobot
input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Training
lr = 0.5
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss
    loss = np.mean((a2 - y) ** 2)

    # Backpropagation
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update bobot
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

# Testing
print("\nHasil prediksi XOR:")
for i in range(4):
    pred = sigmoid(np.dot(sigmoid(np.dot(X[i], W1) + b1), W2) + b2)
    print(f"  {X[i]} -> {pred[0][0]:.4f} (target: {y[i][0]})")

=== Pengolahan Data dengan Pandas ===
import pandas as pd
import numpy as np

# Membuat dataframe dari dictionary
data_siswa = {
    'nama': ['Budi', 'Ani', 'Citra', 'Dedi', 'Eka'],
    'matematika': [85, 92, 78, 88, 95],
    'fisika': [80, 88, 82, 90, 85],
    'kimia': [78, 85, 80, 86, 90],
    'jurusan': ['IPA', 'IPA', 'IPS', 'IPA', 'IPS']
}
df = pd.DataFrame(data_siswa)
print("Data Siswa:")
print(df)

# Analisis deskriptif
print("\nStatistik Deskriptif:")
print(df[['matematika', 'fisika', 'kimia']].describe())

# Filter data
print("\nSiswa dengan nilai matematika di atas 85:")
print(df[df['matematika'] > 85][['nama', 'matematika']])

# Group by
print("\nRata-rata nilai per jurusan:")
print(df.groupby('jurusan')[['matematika', 'fisika', 'kimia']].mean())

# Menambah kolom baru
df['rata_rata'] = df[['matematika', 'fisika', 'kimia']].mean(axis=1)
df['status'] = df['rata_rata'].apply(lambda x: 'Lulus' if x >= 80 else 'Remidi')

print("\nData dengan rata-rata dan status:")
print(df)

# Simpan ke CSV
df.to_csv('data_siswa_output.csv', index=False)
print("\nData disimpan ke data_siswa_output.csv")

=== Pemrograman Web dengan Flask ===
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Route dasar
@app.route('/')
def home():
    return render_template('index.html', title='Halaman Utama')

@app.route('/api/hello', methods=['GET'])
def hello_api():
    nama = request.args.get('nama', 'Dunia')
    return jsonify({'pesan': f'Halo {nama}!', 'status': 'sukses'})

@app.route('/hitung', methods=['POST'])
def hitung():
    data = request.json
    angka1 = data.get('angka1', 0)
    angka2 = data.get('angka2', 0)
    operasi = data.get('operasi', 'tambah')

    if operasi == 'tambah':
        hasil = angka1 + angka2
    elif operasi == 'kurang':
        hasil = angka1 - angka2
    elif operasi == 'kali':
        hasil = angka1 * angka2
    else:
        hasil = 'operasi tidak dikenal'

    return jsonify({'hasil': hasil})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

=== Algoritma dan Struktur Data ===
# Bubble Sort dalam Python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Binary Search
def binary_search(arr, target):
    kiri = 0
    kanan = len(arr) - 1
    while kiri <= kanan:
        tengah = (kiri + kanan) // 2
        if arr[tengah] == target:
            return tengah
        elif arr[tengah] < target:
            kiri = tengah + 1
        else:
            kanan = tengah - 1
    return -1

# Contoh penggunaan
data = [64, 34, 25, 12, 22, 11, 90]
print("Data awal:", data)
data_terurut = bubble_sort(data.copy())
print("Data terurut:", data_terurut)
cari = 22
hasil = binary_search(data_terurut, cari)
print(f"Mencari {cari}: ditemukan di indeks {hasil}")

# Queue menggunakan collections.deque
from collections import deque
antrian = deque()
antrian.append('Pelanggan 1')
antrian.append('Pelanggan 2')
antrian.append('Pelanggan 3')
print(f"Antrian sekarang: {list(antrian)}")
dilayani = antrian.popleft()
print(f"{dilayani} sedang dilayani")
print(f"Sisa antrian: {list(antrian)}")

=== Fungsi dan Modularitas Python ===
# Modul utilitas untuk analisis data
import math

def hitung_persentase(nilai, total):
    if total == 0:
        return 0
    return (nilai / total) * 100

def klasifikasi_nilai(nilai):
    if nilai >= 90:
        return 'A'
    elif nilai >= 80:
        return 'B'
    elif nilai >= 70:
        return 'C'
    elif nilai >= 60:
        return 'D'
    else:
        return 'E'

def hitung_diskon(harga, persen_diskon):
    diskon = harga * (persen_diskon / 100)
    return harga - diskon

def fibonacci(n):
    a, b = 0, 1
    hasil = []
    for _ in range(n):
        hasil.append(a)
        a, b = b, a + b
    return hasil

# Contoh penggunaan
print(f"20% dari 85: {hitung_persentase(20, 85):.1f}%")
print(f"Nilai 85: {klasifikasi_nilai(85)}")
print(f"Harga setelah diskon 25% dari Rp100.000: Rp{hitung_diskon(100000, 25):,.0f}")
print(f"Fibonacci 10 angka: {fibonacci(10)}")

=== Object Oriented Programming Python ===
class Hewan:
    def __init__(self, nama, umur):
        self.nama = nama
        self.umur = umur

    def bersuara(self):
        return f"{self.nama} membuat suara"

    def info(self):
        return f"{self.nama}, {self.umur} tahun"

class Kucing(Hewan):
    def __init__(self, nama, umur, warna):
        super().__init__(nama, umur)
        self.warna = warna
        self.energi = 100

    def bersuara(self):
        return f"{self.nama} mengeong: Meow!"

    def tidur(self, jam):
        self.energi += jam * 10
        if self.energi > 100:
            self.energi = 100
        return f"{self.nama} tidur selama {jam} jam. Energi sekarang: {self.energi}"

    def bermain(self, durasi):
        if self.energi >= durasi * 5:
            self.energi -= durasi * 5
            return f"{self.nama} bermain selama {durasi} menit. Energi: {self.energi}"
        else:
            return f"{self.nama} terlalu lelah untuk bermain!"

class Anjing(Hewan):
    def __init__(self, nama, umur, ras):
        super().__init__(nama, umur)
        self.ras = ras

    def bersuara(self):
        return f"{self.nama} menggonggong: Guk Guk!"

    def menjaga(self):
        return f"{self.nama} menjaga rumah!"

# Contoh penggunaan
kucing = Kucing("Mimi", 2, "Oranye")
print(kucing.bersuara())
print(kucing.bermain(10))
print(kucing.tidur(2))

anjing = Anjing("Rex", 3, "Labrador")
print(anjing.bersuara())
print(anjing.menjaga())

=== Database dengan Python dan SQLite ===
import sqlite3

# Membuat koneksi database
conn = sqlite3.connect('toko_buku.db')
cursor = conn.cursor()

# Membuat tabel
cursor.execute('''
CREATE TABLE IF NOT EXISTS buku (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    judul TEXT NOT NULL,
    penulis TEXT NOT NULL,
    tahun_terbit INTEGER,
    harga REAL,
    stok INTEGER DEFAULT 0
)
''')

# Menambahkan data
buku_baru = [
    ('Pemrograman Python', 'Budi Santoso', 2023, 85000, 10),
    ('Belajar Machine Learning', 'Ani Wijaya', 2024, 120000, 5),
    ('Data Science untuk Pemula', 'Citra Dewi', 2023, 95000, 8)
]

cursor.executemany(
    'INSERT INTO buku (judul, penulis, tahun_terbit, harga, stok) VALUES (?, ?, ?, ?, ?)',
    buku_baru
)
conn.commit()

# Query data
cursor.execute('SELECT * FROM buku')
semua_buku = cursor.fetchall()
for buku in semua_buku:
    print(f"ID: {buku[0]}, Judul: {buku[1]}, Penulis: {buku[2]}, Harga: Rp{buku[4]:,.0f}")

# Update stok
cursor.execute('UPDATE buku SET stok = stok - 1 WHERE id = 1')
conn.commit()

conn.close()
""")

total_hardcoded = sum(len(t) for t in hardcoded)
print(f"   Hardcoded texts: {total_hardcoded:,} chars")

# ── 4. English texts (bilingual capability) ──
print("\n4️⃣  Adding English texts...")

english_texts = []

english_texts.append("""
=== Introduction to Artificial Intelligence ===
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. The field of AI was formally founded in 1956 at the Dartmouth Conference, where researchers gathered to explore the possibility of creating machines that could think like humans.
Machine Learning (ML) is a subset of AI that focuses on algorithms that improve automatically through experience. Instead of being explicitly programmed for every task, machine learning algorithms learn patterns from data. The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds hidden patterns in unlabeled data, and reinforcement learning uses rewards and punishments to guide an agent's behavior.
Deep Learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks have achieved remarkable results in image recognition, natural language processing, speech recognition, and game playing. Convolutional Neural Networks (CNNs) are particularly effective for image tasks, while Recurrent Neural Networks (RNNs) and Transformers excel at sequence data like text and speech.
Natural Language Processing (NLP) is a key application of AI that deals with the interaction between computers and human language. Modern NLP systems can translate between languages, summarize documents, answer questions, generate human-like text, and analyze sentiment. The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, revolutionized NLP and forms the basis of models like GPT, BERT, and T5.
Large Language Models (LLMs) like GPT-4, Claude, and Llama are trained on vast amounts of text data and can generate coherent and contextually relevant responses. These models use the transformer architecture with billions of parameters. The key innovation behind LLMs is scaling: larger models trained on more data with more compute consistently perform better, following what researchers call scaling laws.
=== Python for Machine Learning ===
import numpy as np
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if epoch % 100 == 0:
                loss = mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: MSE = {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

=== Data Structures and Algorithms ===
A data structure is a particular way of organizing data in a computer so that it can be used effectively. The choice of data structure depends on the type of operations that need to be performed. Common data structures include arrays, linked lists, stacks, queues, trees, graphs, hash tables, and heaps.
Arrays are the simplest data structure, storing elements in contiguous memory locations. Arrays provide O(1) access by index but O(n) for insertion and deletion. Linked lists consist of nodes where each node contains data and a pointer to the next node, allowing O(1) insertion and deletion but O(n) access.
Stacks follow the Last-In-First-Out (LIFO) principle, like a stack of plates. The main operations are push (add to top) and pop (remove from top). Stacks are used in function call management, undo operations, and expression evaluation. Queues follow the First-In-First-Out (FIFO) principle, like a line of people waiting. They are used in task scheduling, breadth-first search, and buffering.
Trees are hierarchical data structures consisting of nodes connected by edges. Binary trees are the most common type, where each node has at most two children. Binary Search Trees (BST) maintain the property that left child values are less than the parent and right child values are greater, enabling O(log n) search, insertion, and deletion on average.
Hash tables store key-value pairs and provide O(1) average time complexity for lookups. A hash function maps keys to array indices. Collisions occur when two keys hash to the same index and are resolved through chaining (storing multiple items in a linked list) or open addressing (finding the next empty slot).
=== Sorting and Searching Algorithms ===
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print("Original:", arr)
print("Sorted (quicksort):", quicksort(arr.copy()))
print("Sorted (mergesort):", mergesort(arr.copy()))
print(f"Index of 5: {binary_search(sorted(arr), 5)}")
=== Programming Fundamentals ===
Variables are containers for storing data values. In Python, variables are dynamically typed, meaning you don't need to declare their type explicitly. Common data types include integers, floats, strings, booleans, lists, tuples, sets, and dictionaries. Each type supports specific operations and methods.
Control flow statements allow you to execute different code paths based on conditions. The if-elif-else statement evaluates boolean expressions and runs the corresponding block. Loops let you repeat code: for loops iterate over sequences, while loops continue until a condition becomes False. The break statement exits a loop early, and continue skips to the next iteration.
Functions are reusable blocks of code that perform a specific task. They take input parameters and can return values. Functions promote code reuse, modularity, and readability. Lambda functions are anonymous one-line functions useful for simple operations. Higher-order functions like map, filter, and reduce take other functions as arguments.
Object-Oriented Programming (OOP) organizes code around objects rather than functions. A class is a blueprint for creating objects, defining attributes (data) and methods (functions). The four pillars of OOP are encapsulation (hiding internal state), inheritance (creating class hierarchies), polymorphism (different classes responding to the same interface), and abstraction (hiding complex implementation details).
=== Computer Networking ===
Computer networking allows computers to communicate and share resources. Networks are classified by their scale: PAN (Personal Area Network), LAN (Local Area Network), MAN (Metropolitan Area Network), and WAN (Wide Area Network). The Internet is the largest WAN, connecting billions of devices worldwide.
The OSI model has seven layers: Physical, Data Link, Network, Transport, Session, Presentation, and Application. Each layer handles specific aspects of communication. The TCP/IP model has four layers: Network Interface, Internet, Transport, and Application. IP (Internet Protocol) handles addressing and routing, while TCP (Transmission Control Protocol) ensures reliable, ordered delivery of data.
HTTP (Hypertext Transfer Protocol) is the foundation of data communication on the web. HTTP/1.1 introduced persistent connections and chunked transfer encoding. HTTP/2 added multiplexing and server push. HTTP/3 uses QUIC over UDP for reduced latency. HTTPS encrypts HTTP traffic using TLS (Transport Layer Security), protecting against eavesdropping and tampering.
=== Mathematics for Machine Learning ===
Linear algebra is fundamental to machine learning. Vectors are ordered collections of numbers that represent points in space. Matrices are rectangular arrays of numbers that represent linear transformations. Key operations include dot products, matrix multiplication, transposition, and finding determinants and eigenvalues. These operations form the basis of how neural networks process data.
Calculus enables optimization in machine learning. The derivative measures how a function changes with respect to its input. The gradient is a vector of partial derivatives pointing in the direction of steepest ascent. Gradient descent uses the negative gradient to find function minima, which is how neural networks learn. The chain rule allows us to compute gradients through multiple layers (backpropagation).
Probability theory deals with uncertainty and randomness. A probability distribution describes the likelihood of different outcomes. The normal (Gaussian) distribution is the most common, characterized by its mean and standard deviation. Bayes' theorem describes how to update probabilities given new evidence, forming the basis of Bayesian machine learning.
Statistics provides tools for drawing conclusions from data. Descriptive statistics summarize data using measures like mean, median, mode, variance, and standard deviation. Inferential statistics allows us to make predictions or generalizations about a population based on a sample. Hypothesis testing determines whether observed effects are statistically significant.
=== The Unix Operating System ===
Unix is a powerful, multi-user operating system family that originated at Bell Labs in the 1970s. Linux is a Unix-like operating system kernel created by Linus Torvalds in 1991. The core philosophy of Unix is that each program should do one thing well, and complex tasks are accomplished by combining simple tools through pipes and redirection.
The Unix filesystem is a hierarchical tree structure rooted at /. Key directories include /bin (essential user binaries), /etc (configuration files), /home (user home directories), /var (variable data like logs), /tmp (temporary files), and /usr (user utilities and applications). Everything in Unix is a file, including devices, sockets, and pipes.
Common Unix commands include ls (list files), cd (change directory), cp (copy), mv (move/rename), rm (remove), mkdir (create directory), grep (search text), find (locate files), chmod (change permissions), ps (process status), and kill (terminate processes). The pipe operator | connects commands, sending the output of one command as input to the next.
Shell scripting allows automating repetitive tasks. A shell script is a text file containing commands. Variables, conditionals, loops, and functions make shell scripts powerful automation tools. Common shells include Bash (Bourne Again Shell), Zsh (Z Shell), and Fish (Friendly Interactive Shell).
=== Introduction to Databases ===
A database is an organized collection of structured data. Database Management Systems (DBMS) provide interfaces for creating, reading, updating, and deleting data (CRUD operations). Relational databases store data in tables with rows and columns, using SQL (Structured Query Language) for queries. Examples include PostgreSQL, MySQL, and SQLite.
SQL is a declarative language for managing relational databases. Key SQL commands include SELECT (retrieve data), INSERT (add new rows), UPDATE (modify existing data), DELETE (remove rows), CREATE TABLE (define new tables), and JOIN (combine data from multiple tables). A primary key uniquely identifies each row, while foreign keys establish relationships between tables.
NoSQL databases provide flexible schemas for unstructured or semi-structured data. Document stores like MongoDB store JSON-like documents. Key-value stores like Redis are extremely fast for simple lookups. Column-family stores like Cassandra excel at write-heavy workloads. Graph databases like Neo4j are designed for highly connected data with complex relationships.
ACID properties ensure reliable database transactions: Atomicity (all or nothing), Consistency (data remains valid), Isolation (concurrent transactions don't interfere), and Durability (committed data persists). The CAP theorem states that distributed databases can only guarantee two of three properties: Consistency, Availability, and Partition tolerance.
=== Version Control with Git ===
Git is a distributed version control system created by Linus Torvalds in 2005. It tracks changes to files, enables collaboration, and maintains a complete history of a project. Unlike centralized systems, every developer has a full copy of the repository, enabling offline work and resilience.
A Git repository consists of a working directory, staging area, and commit history. The basic workflow is: modify files in the working directory, stage changes with git add, and commit them with git commit. The git log command shows the commit history. The git diff command shows uncommitted changes. The git status command shows the current state of the repository.
Branching is a powerful Git feature that allows parallel development. A branch is a movable pointer to a specific commit. The main branch (often called main or master) typically represents the stable version. Feature branches are created for new development. Merging combines changes from different branches using either fast-forward or three-way merge strategies.
Remote repositories enable collaboration. The git clone command creates a local copy of a remote repository. The git push command uploads local commits, and git pull downloads and integrates remote changes. Pull requests or merge requests are the standard mechanism for proposing changes in collaborative development.
=== English Technical Vocabulary for Indonesian Programmers ===
When programming, many technical terms are in English. The Python programming language is widely used for data science, web development, and automation. Common terms include function, variable, class, object, method, attribute, parameter, argument, return value, loop, condition, exception, and module.
Version control with Git involves terms like repository, commit, branch, merge, pull request, conflict, clone, fork, remote, and tag. Understanding these terms is essential for collaborative software development.
Web development terms include client, server, request, response, endpoint, API, REST, JSON, HTML, CSS, JavaScript, frontend, backend, full-stack, framework, and middleware. These terms describe the architecture of web applications.
Database terms include query, index, migration, schema, table, row, column, foreign key, join, aggregation, transaction, and stored procedure. These concepts are used when working with both SQL and NoSQL databases.
Machine learning terms include model, training, inference, feature, label, epoch, batch, gradient, loss function, accuracy, precision, recall, overfitting, underfitting, validation, and hyperparameter. Understanding these terms is essential for anyone working with AI and machine learning systems.
""")

english_texts.append("""
=== Probability and Statistics ===
Probability is the measure of the likelihood that an event will occur. It ranges from 0 (impossible) to 1 (certain). The probability of event A is written as P(A). For mutually exclusive events, P(A or B) = P(A) + P(B). For independent events, P(A and B) = P(A) * P(B). Conditional probability P(A|B) is the probability of A given that B has occurred.

Bayes' Theorem is a fundamental principle in probability: P(A|B) = P(B|A) * P(A) / P(B). This theorem describes how to update beliefs based on new evidence. In machine learning, Bayes' theorem forms the basis of naive Bayes classifiers and Bayesian inference.

The normal distribution (also called Gaussian distribution) is a continuous probability distribution characterized by its bell-shaped curve. It is defined by two parameters: the mean (mu) and standard deviation (sigma). The 68-95-99.7 rule states that approximately 68% of data falls within one standard deviation of the mean, 95% within two, and 99.7% within three standard deviations.

Descriptive statistics summarize data using measures of central tendency (mean, median, mode) and measures of dispersion (range, variance, standard deviation, interquartile range). The mean is the arithmetic average, the median is the middle value, and the mode is the most frequent value. Standard deviation measures how spread out the data points are from the mean.
""")

english_texts.append("""
=== English Code Examples ===
# A simple web scraper using requests and BeautifulSoup
import requests
from bs4 import BeautifulSoup

def scrape_headlines(url):
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []
    
    for headline in soup.find_all(['h1', 'h2', 'h3']):
        text = headline.get_text(strip=True)
        if text and len(text) > 10:
            headlines.append(text)
    
    return headlines[:10]

# A recursive descent parser for simple arithmetic expressions
# Grammar: expr -> term (+ term)*, term -> factor (* factor)*, factor -> number | (expr)
import re

class Parser:
    def __init__(self, text):
        self.tokens = re.findall(r'\d+|[+*()]', text)
        self.pos = 0
    
    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def consume(self, expected=None):
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Expected {expected}, got {tok}")
        self.pos += 1
        return tok
    
    def parse_expr(self):
        result = self.parse_term()
        while self.peek() == '+':
            self.consume('+')
            result += self.parse_term()
        return result
    
    def parse_term(self):
        result = self.parse_factor()
        while self.peek() == '*':
            self.consume('*')
            result *= self.parse_factor()
        return result
    
    def parse_factor(self):
        tok = self.peek()
        if tok == '(':
            self.consume('(')
            result = self.parse_expr()
            self.consume(')')
            return result
        else:
            self.consume()
            return int(tok)
    
    def parse(self):
        result = self.parse_expr()
        if self.peek() is not None:
            raise SyntaxError(f"Unexpected token: {self.peek()}")
        return result

# Test the parser
print(Parser("2+3*4").parse())  # 14
print(Parser("(2+3)*4").parse())  # 20
print(Parser("1+2+3+4+5").parse())  # 15
""")

english_texts.append("""
=== Mathematical Reasoning ===
Let's reason through some mathematical concepts. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The first few primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29. The number 2 is the only even prime number. There are infinitely many prime numbers, as proven by Euclid around 300 BC.

A function f from set X to set Y is a relation that assigns each element x in X exactly one element y in Y. The domain is the set of all possible inputs, and the codomain is the set of all possible outputs. The range is the set of actual outputs. A function is injective (one-to-one) if distinct inputs map to distinct outputs. A function is surjective (onto) if every element in the codomain is mapped to by some element in the domain.

The Fibonacci sequence is defined recursively: F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n > 1. This sequence appears in many natural phenomena, from the arrangement of leaves on a stem to the spiral of a seashell. The ratio of consecutive Fibonacci numbers approaches the golden ratio phi = (1 + sqrt(5)) / 2 approximately equal to 1.618.

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Test
for n in [5, 10, 15]:
    print(f"fib({n}) = {fibonacci(n)}")
for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    print(f"is_prime({n}) = {is_prime(n)}")
print(f"factorial(5) = {factorial(5)}")
print(f"factorial(10) = {factorial(10)}")

=== Graph Theory ===
A graph G = (V, E) consists of a set of vertices V and a set of edges E. Graphs can be directed (edges have a direction) or undirected (edges are bidirectional). Graphs are used to model many real-world systems: social networks, road networks, the internet, molecular structures, and dependency relationships in software.

Breadth-First Search (BFS) traverses a graph level by level, visiting all neighbors of a vertex before moving to the next level. BFS uses a queue and finds the shortest path in unweighted graphs. Depth-First Search (DFS) explores as far as possible along each branch before backtracking. DFS uses a stack or recursion and is useful for topological sorting, cycle detection, and maze solving.

Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph with non-negative edge weights. It uses a priority queue and greedily selects the vertex with the smallest distance. The Bellman-Ford algorithm handles negative edge weights and can detect negative cycles. A* is a heuristic search algorithm that often finds the shortest path more efficiently than Dijkstra's algorithm.

from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

def dfs(graph, start):
    visited = set()
    result = []
    def dfs_recursive(vertex):
        visited.add(vertex)
        result.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs_recursive(neighbor)
    dfs_recursive(start)
    return result

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print(f"BFS from A: {bfs(graph, 'A')}")
print(f"DFS from A: {dfs(graph, 'A')}")
""")

total_english = sum(len(t) for t in english_texts)
print(f"   English texts: {total_english:,} chars")

# ── 5. Combine all text ──
print(f"\n5️⃣  Combining all texts...")
all_texts = [existing_text] + wiki_texts + hardcoded + english_texts
combined_corpus = '\n'.join(all_texts)
print(f"   Total corpus: {len(combined_corpus):,} chars")
print(f"   Lines: {combined_corpus.count(chr(10)):,}")

# Save raw text
corpus_path = os.path.join(DATA_DIR, 'indonesian_corpus.txt')
with open(corpus_path, 'w', encoding='utf-8') as f:
    f.write(combined_corpus)
print(f"   Saved: {corpus_path}")

# ── 5. Train BPE tokenizer ──
print("\n5️⃣  Training BPE tokenizer...")
# Split into lines for training (BPE benefits from seeing many short sequences)
train_texts = combined_corpus.split('\n')
train_texts = [t for t in train_texts if len(t.strip()) > 0]
print(f"   Training on {len(train_texts):,} lines")

tok = BPETokenizer()
VOCAB_TARGET = 5000
tok.train(train_texts, vocab_size=VOCAB_TARGET, verbose=True)

# Save tokenizer
tok_path = os.path.join(DATA_DIR, 'tokenizer.pkl')
tok.save(tok_path)
print(f"   Tokenizer saved: {tok_path}")

# ── 6. Tokenize corpus ──
print("\n6️⃣  Tokenizing corpus...")
all_ids = []
doc_count = 0
for line in combined_corpus.split('\n'):
    line = line.strip()
    if line:
        ids = tok.encode(line, bos=True, eos=True)
        all_ids.extend(ids)
        doc_count += 1

tokens_arr = np.array(all_ids, dtype=np.int32)
token_path = os.path.join(DATA_DIR, 'tokens.npy')
np.save(token_path, tokens_arr)
print(f"   Tokens: {len(tokens_arr):,}")
print(f"   Documents: {doc_count:,}")
print(f"   Unique vocab used: {len(np.unique(tokens_arr)):,}")

# ── 7. Test encode/decode ──
print("\n7️⃣  Testing tokenization...")
test_texts = [
    "Indonesia adalah negara demokrasi terbesar ketiga di dunia",
    "def hello_world():\n    print('Halo Dunia')",
    "class NeuralNetwork:\n    def __init__(self):\n        self.layers = []",
    "Prabowo Subianto adalah Presiden Indonesia kedelapan",
    "import numpy as np\nimport pandas as pd",
]

for t in test_texts:
    ids = tok.encode(t, bos=True, eos=True)
    decoded = tok.decode(ids)
    # Remove special tokens for display
    decoded_clean = decoded
    print(f"\n  In:  {t[:60]}")
    print(f"  IDs:  [{ids[0]}, {ids[1]}, ... {ids[-1]}] ({len(ids)} tokens)")
    print(f"  Out: {decoded_clean[:60]}")

# Print coverage stats
print("\n\n═" * 50)
vocab_file = os.path.join(DATA_DIR, 'vocab_info.txt')
with open(vocab_file, 'w', encoding='utf-8') as f:
    f.write(f"Corpus: {len(combined_corpus):,} chars\n")
    f.write(f"Tokens: {len(tokens_arr):,}\n")
    f.write(f"Documents: {doc_count:,}\n")
    f.write(f"Vocab size: {tok.vocab_size}\n")
    f.write(f"Merges: {len(tok.merges)}\n")
    f.write(f"UNK: {np.sum(tokens_arr == 1):,} ({(np.sum(tokens_arr == 1)/len(tokens_arr)*100):.1f}%)\n")

print(f"✅ V2 corpus build complete!")
print(f"   Corpus: {corpus_path}")
print(f"   Tokens: {DATA_DIR}/tokens.npy ({len(tokens_arr):,} tokens)")
print(f"   Tokenizer: {tok_path} (vocab={tok.vocab_size})")
print(f"   Vocab info: {vocab_file}")
print(f"\n   Next: run bin/burst_train_v2.py")
