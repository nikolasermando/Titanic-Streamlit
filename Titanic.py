# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:04:55 2022

@author: Nikolas Ermando
"""
# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Header
st.write("""
# Titanic Machine Learning dengan Logistic Regression
##### Dibuat oleh Nikolas Ermando
Aplikasi ini memprediksi status keselamatan penumpang Titanic berdasarkan berapa faktor.
Project ini terinspirasi dari [course Udemy Data Science and Machine Learning](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) yang diajarkan oleh Jose Portila.
Data dipreroleh dari [Kaggle](https://www.kaggle.com/c/titanic).
""")
st.write("Libraries yang dipakai : streamlit, pandas, numpy, scikit-learn, dan plotly")
# Sidebar
st.sidebar.header('Input Nilai disini')
st.sidebar.subheader('Faktor faktor dibawah ini dinilai memberikan dampak keselamatan pada penumpang Titanic')

def user_input_features():
    PassengerId = st.sidebar.slider('ID Penumpang',1,891,1)
    Pclass = st.sidebar.slider("Kelas Penumpang",1,3,1)
    Age = st.sidebar.slider("Umur Penumpang",0,80,26)
    SibSp = st.sidebar.slider("Jumlah Saudara Kandung / Sepupu",0,8,2)
    Parch = st.sidebar.slider("Jumlah Orang tua kandung / Anak Kandung",0,6,2)
    Fare = st.sidebar.slider("Tarif Penumpang",0,513,200)
    male = st.sidebar.selectbox("Jenis Kelamin (0:Perempuan, 1:Pria)",(0,1,))
    st.sidebar.write("Jika Tujuan ke Queenstown dan Southhampton adalah 0, Maka tujuan ke Cherbourg")
    Q = st.sidebar.selectbox("Tujuan ke Queenstown", (0,1,))
    S = st.sidebar.selectbox("Tujuan ke Southhampton", (0,1,))
    data = {"PassengerId":PassengerId,
            "Pclass":Pclass,
            "Age":Age,
            "SibSp":SibSp,
            "Parch":Parch,
            "Fare":Fare,
            "male":male,
            "Q":Q,
            "S":S}
    features = pd.DataFrame(data,index=[0])
    return features
input_df = user_input_features()

# Membaca dataset
train = pd.read_csv('titanic_train.csv')

#Eksplorasi data
st.subheader("1. Mengeksplor Data")
st.write("Pertama-tama, mengimport dataset yang diperoleh dari Kaggle")
st.write('Data Dimension: ' + str(len(train)) + ' rows and ' + str(len(train.columns)) + ' columns.')
st.dataframe(train)

# Mencari data yang hilang
st.subheader("2. Mengidentifikasi Missing Value")
st.write("Disini, penulis mengidentifikasi data yang hilang dengan menggunakan visualisasi heatmap untuk mendapatkan gambaran dan pola dari data yang hilang atau null value")
st.write("Berikut ini adalah visualisasi Heatmap menggunakan library plotly")

trainnullarr = train.isnull().to_numpy()
fignull = px.imshow(trainnullarr, aspect="auto", x = train.columns, color_continuous_scale='Agsunset')
st.plotly_chart(fignull)

st.write("Sekitar 20 persen dari data Age hilang. Proporsi Usia yang hilang mungkin saja digantikan dengan nilai berdasarkan analisis eksplorasi data, Jika didapatkan hubungan antara variabel yang memungkinan, kemungkinan besar variabel yang hilang dari usia penumpang bisa diganti.")
st.write("Melihat kolom Kabin, sepertinya kehilangan terlalu banyak data untuk melakukan sesuatu yang berguna pada proses machine learning sehingga kolom Kabin sebaiknya dihapus saja.")

# Visualisasi hubungan pclass dan usia
figpclassage = px.box(train, x="Pclass", y="Age")
st.plotly_chart(figpclassage)

st.write("Dapat dilihat bahwa kelas yang lebih tinggi memiliki usia yang lebih tua, sehingga data yang hilang pada kolom Age akan digantikan berdasarkan kelas penumpang yang ada. Nilai yang digunakan adalah nilai median dari boxplot. Kelas 1 memiliki median 37, kelas 2 memiliki median 29, dan kelas 3 memiliki median 24.")
st.write("Setelah menganti nilai hilang pada kolom umur, maka heatmap dataset nilai yang hilang akan ditampilkan sebagai berikut")

# Data Cleaning kolom Age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train1 = train
train1['Age'] = train1[['Age','Pclass']].apply(impute_age, axis=1)

# Visualisasi heatmap data cleaning pertama
trainnullarr1 = train1.isnull().to_numpy()
fignull1 = px.imshow(trainnullarr1, aspect="auto", x = train.columns, color_continuous_scale='Agsunset')
st.plotly_chart(fignull1)

st.write("Kolom Kabin memiliki missing value yang terlalu banyak, sehingga lebih baik dihapus saja.")

# Visualisasi heatmap data cleaning kedua
train2 = train1
train2.drop('Cabin',axis=1,inplace=True)

trainnullarr2 = train2.isnull().to_numpy()
fignull2 = px.imshow(trainnullarr2, aspect="auto", x = train.columns, color_continuous_scale='Agsunset')
st.plotly_chart(fignull2)

st.write("Untuk kolom Embark, missing valuenya sangat sedikit. Sehingga dihapus saja baris yang memiliki nilai hilang pada kolom Embarknya")

# Visualisasi heatmap data cleaning ketiga
train3 = train2
train3.dropna(inplace=True)

trainnullarr3 = train3.isnull().to_numpy()
fignull3 = px.imshow(trainnullarr3, aspect="auto", x = train.columns, color_continuous_scale='Agsunset')
st.plotly_chart(fignull3)

# Model data
st.subheader("3. Memodelkan data")
st.write("Karena dataset sudah bersih, maka dimulai pemodelan data dengan libraries scikit-learn python. Jenis model yang digunakan adalah Logistic Regression karena data yang akan kita prediksi merupakan status keselamatan yang mana nilainya terdiri dari 0 (tidak selamat) dan 1 (selamat).")
st.write("Dataset dibagi menjadi 70% data training dan 30% data testing. Hal ini bertujuan agar 70% data tersebut digunakan untuk membangun model Logistic Reggresion-nya, sedangkan 30% data digunakan untuk mengevaluasi model yang dibangun.")

# Simplify data
sex = pd.get_dummies(train3['Sex'],drop_first=True)
embark = pd.get_dummies(train3['Embarked'],drop_first=True)
train3.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train3 = pd.concat([train3,sex,embark],axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(train3.drop('Survived',axis=1), 
                                                    train3['Survived'], test_size=0.30, 
                                                    random_state=12453000)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Memprediksi Model
predictions = logmodel.predict(X_test)

# Evaluasi
st.write("Pada train test split, Random Number yang dimasukin 12453000.")
st.write("Model Report :")
st.text(classification_report(y_test,predictions))
# Prediksi
st.subheader("4. Let's Try Predict Our Model!")

# Mengabungkan dataset dengan nilai yang kita masukin ke sidebar
train4 = train3.drop(columns = ['Survived'])
df = pd.concat([input_df,train4],axis=0)
df = df[:1]

# Prediksi
prediction1 = logmodel.predict(df)
prediction_proba = logmodel.predict_proba(df)

# Menampilkan Prediksi Sidebar
arr = np.array(['Tidak Selamat','Selamat'])

st.write("Masukin nilai pada sidebar yang sudah disediakan")
st.write("Hasil Prediksi Status Keselamatan")
st.write(arr[prediction1])

st.write("Probabilitas masing-masing status 0 (Tidak Selamat) dan 1 (Selamat)")
st.write(prediction_proba)

# Kesimpulan
st.subheader("5. Kesimpulan")
st.write("""
         * Jenis Kelamin Perempuan memiliki tingkat keselamatan yang lebih tinggi dibanding Pria, yang berarti proses evakuasi memprioritaskan wanita dibanding pria
         * Semakin besar tarif penumpang yang dibayar semakin terjamin keselamatan ketika terjadi bencana, yang berarti proses evakuasi lebih memprioritas penumpang yang membayar lebih banyak
         * ID Penumpang, Kelas Penumpang, Umur Penumpang, Jumlah Saudara, Jumlah Orang tua dan Anak, dan Destinasi tidak menjadi pengaruh yang signifikan terhadap status keselamatan penumpang""")

