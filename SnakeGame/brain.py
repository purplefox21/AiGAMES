# Keras kütüphanelerini içe aktar
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Beyin sınıfını oluşturma
class Brain():
     
     # Başlatıcı fonksiyon, temel parametreleri ayarlar
     def __init__(self, inputShape, lr = 0.005):
          self.inputShape = inputShape  # Giriş şekli, modelin alacağı görsel verinin boyutlarını tanımlar
          self.learningRate = lr  # Öğrenme hızı, modelin öğrenme sürecinde ne kadar hızlı ayarlamalar yapacağını belirler
          self.numOutputs = 4  # Hareket edilebilecek 4 yön (yukarı, aşağı, sağ, sol)
          
          # Sinir ağı modelini oluşturma
          self.model = Sequential()  # Katmanlar sıralı bir şekilde eklenecek
          
          # Birinci evrişimli katmanı ekleme
          self.model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = self.inputShape))  # 32 filtre kullanarak 3x3 boyutunda evrişim yapılır; aktivasyon fonksiyonu olarak ReLU kullanılır
          
          # Max pooling katmanı
          self.model.add(MaxPooling2D((2, 2)))  # Öznitelik haritasını küçültmek için 2x2 boyutunda max pooling uygulanır
          
          # İkinci evrişimli katmanı ekleme
          self.model.add(Conv2D(64, (2,2), activation = 'relu'))  # 64 filtre kullanarak 2x2 boyutunda evrişim yapılır; aktivasyon fonksiyonu olarak ReLU kullanılır
          
          # Düzleştirme katmanı
          self.model.add(Flatten())  # Çok boyutlu öznitelik haritalarını tek boyutlu bir vektöre dönüştürür
          
          # Yoğun katman
          self.model.add(Dense(256, activation = 'relu'))  # 256 nöronlu yoğun katman; aktivasyon fonksiyonu olarak ReLU kullanılır
          
          # Çıkış katmanı
          self.model.add(Dense(self.numOutputs))  # Çıkış katmanı, hareket edilebilecek 4 yön için çıktı üretir
          
          # Modeli derleme
          self.model.compile(optimizer = Adam(lr = self.learningRate), loss = 'mean_squared_error')  # Modeli Adam optimizasyonu ve ortalama kare hata kaybı ile derler
      
     # Kaydedilmiş bir modeli yükleme metodu
     def loadModel(self, filepath):
          self.model = load_model(filepath)  # Belirtilen yoldaki modeli yükler
          return self.model
