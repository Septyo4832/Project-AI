TOLONG DIBACA AGAR TIDAK ADA MASALAH

dikarenakan ada masalah pada depedency dan codenya, maka versi python yang disarankan untuk digunakan adalah Python 3.10

setelah clone repository dan install Python 3.10, buat virtual environment agar package yang diinstal nanti tidak mengganggu projek lain kalian. cara buatnya:
py -3.10 -m venv .venv  #ketikkan ini di terminal vscode untuk repo projek
setelah itu harus diaktifkan tapi biasanya sudah aktif otomatis ketika menggunakan terminal baru, tapi kalau otomatis. maka harus diaktikan seperti ini:
.venv\Scripts\activate #ketikkan di terminal

setelah membuat vn, selanjutnya menginstal semua package dari requierement.txt, caranya:
pip install -r requirements.txt #ketikan di terminal
