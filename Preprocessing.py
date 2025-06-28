import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CSV dosyasını oku
veriler = pd.read_csv("veriler.csv")
print(veriler)  # Ham veriyi yazdır

# Sadece "boy" sütununu al (örnek amaçlı)
boy = veriler[["boy"]]
print(boy)

# Eksik verileri doldurmak için SimpleImputer kullan (boy, kilo, yas sütunları)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Bu sütunlar üzerinde çalışılacak: boy, kilo, yas
Yas = veriler.iloc[:, 1:4].values
print(Yas)

# Eksik değerleri ortalama ile doldur
imputer = imputer.fit(Yas)
Yas = imputer.transform(Yas)
print(Yas)

# Ülke sütunu kategorik olduğu için OneHotEncoder ile sayısal hale getir
ulke = veriler.iloc[:, 0:1].values  # 2D array olarak al
print(ulke)

# OneHotEncoder ile ülke sütununu kodla
ohe = preprocessing.OneHotEncoder(sparse_output=False)  # sklearn >= 1.2
ulke = ohe.fit_transform(ulke)
print(ulke)

# Kodlanmış ülke verilerini DataFrame'e çevir
index = range(len(veriler))
sonuc = pd.DataFrame(data=ulke, index=index, columns=["fr", "tr", "us"])
print(sonuc)

# Eksik değerleri doldurulmuş boy, kilo, yas verilerini DataFrame'e çevir
sonuc2 = pd.DataFrame(data=Yas, index=index, columns=['boy', 'kilo', 'yas'])
print(sonuc2)

# Ülke + sayısal verileri birleştir
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

# Cinsiyet sütununu al (bağımlı değişken)
cinsiyet = veriler.iloc[:, -1:].values  # 2D olarak alınır
sonuc3 = pd.DataFrame(data=cinsiyet, index=index, columns=['cinsiyet'])
print(sonuc3)

# Son veri seti: ülke + sayısal veriler + cinsiyet
s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

# Veriyi eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# Veriyi ölçeklendir (standardizasyon)
sc = StandardScaler()

# Eğitim ve test verisini ayrı ayrı dönüştür (fit sadece eğitim verisine uygulanır)
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
