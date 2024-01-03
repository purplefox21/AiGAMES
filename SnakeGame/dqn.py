# Deneyim Tekrarı Hafızasını başlatma
import numpy as np

class Dqn():
     
     # Başlatıcı fonksiyon, temel parametreleri ayarlar
     def __init__(self, maxMemory, discount):
          self.maxMemory = maxMemory  # Maksimum hafıza boyutu, hafızada saklanacak maksimum deneyim sayısını belirler
          self.discount = discount  # İndirim faktörü, gelecekteki ödüllerin şimdiki değerini hesaplarken kullanılır
          self.memory = list()  # Deneyimleri saklamak için boş bir liste oluşturulur
          
     # Yeni deneyimi hatırlama fonksiyonu
     def remember(self, transition, gameOver):
          self.memory.append([transition, gameOver])  # Hafızaya yeni bir deneyim eklenir; her deneyim bir durum geçişi ve oyunun bitip bitmediği bilgisini içerir
          if len(self.memory) > self.maxMemory:  # Eğer hafıza maksimum boyutu aşarsa, en eski deneyim silinir
               del self.memory[0]
     
     # Girdi ve hedef değerlerin toplu olarak alınması
     def getBatch(self, model, batchSize):
          lenMemory = len(self.memory)  # Hafızadaki toplam deneyim sayısı
          numOutputs = model.output_shape[-1]  # Modelin çıkış katmanındaki nöron sayısı (eylem sayısı)
          
          # Girdi ve hedef dizilerini başlatma
          inputs = np.zeros((min(batchSize, lenMemory), self.memory[0][0][0].shape[1], self.memory[0][0][0].shape[2], self.memory[0][0][0].shape[3]))
          targets = np.zeros((min(batchSize, lenMemory), numOutputs))
          
          # Rastgele deneyimlerden geçişleri çıkarma
          for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(batchSize, lenMemory))):
               currentState, action, reward, nextState = self.memory[inx][0]  # Mevcut durum, eylem, ödül ve sonraki durumu içeren deneyim
               gameOver = self.memory[inx][1]  # Oyunun bu deneyimde bitip bitmediği
               
               # Girdi ve hedef değerleri güncelleme
               inputs[i] = currentState  # Girdi olarak mevcut durum
               targets[i] = model.predict(currentState)[0]  # Hedef değerler, modelin tahminleri ile başlatılır
               if gameOver:
                    targets[i][action] = reward  # Eğer oyun bittiyse, hedef değeri doğrudan elde edilen ödül olur
               else:
                    # Oyun devam ediyorsa, hedef değer, elde edilen ödül ve gelecekteki en iyi ödülün indirimli değeri olur
                    targets[i][action] = reward + self.discount * np.max(model.predict(nextState)[0])
          
          return inputs, targets  # İşlenmiş girdi ve hedef değerlerini döndürür
