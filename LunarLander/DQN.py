# Gerekli kütüphaneler ve modüller içe aktarılıyor.
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
from collections import deque

# DQN sınıfı, Deep Q-Learning algoritmasının bir uygulamasıdır.
class DQN:
    """ Implementation of deep q learning algorithm """
    def __init__(self, action_space, state_space):
        # Aksiyon ve durum uzayının boyutları tanımlanıyor.
        self.action_space = action_space
        self.state_space = state_space

        # Eğitim sürecinde kullanılacak parametreler tanımlanıyor.
        self.epsilon = 1.0  # Keşif oranı
        self.gamma = .99  # İndirim faktörü
        self.batch_size = 64  # Mini parti boyutu
        self.epsilon_min = .01  # Epsilon'un minimum değeri
        self.lr = 0.001  # Öğrenme oranı
        self.epsilon_decay = .996  # Epsilon azalma oranı

        # Deneyim tekrarı (experience replay) için bir bellek tanımlanıyor.
        self.memory = deque(maxlen=1000000)

        # Sinir ağı modeli oluşturuluyor.
        self.model = self.build_model()

    # Sinir ağı modelini oluşturan fonksiyon.
    def build_model(self):
        model = Sequential()
        model.add(Dense(250, input_dim=self.state_space, activation=relu))  # İlk katman
        model.add(Dense(150, activation=relu))  # İkinci katman
        model.add(Dense(self.action_space, activation=linear))  # Çıkış katmanı
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))  # Model derleniyor
        return model

    # Oynanan oyunun durumunu belleğe kaydeden fonksiyon.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # Bir aksiyon seçen fonksiyon. Rastgele (keşif) veya model tahmini (kullanım) arasında karar verir.
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Eğer rastgele bir sayı epsilon'dan küçükse, rastgele bir aksiyon seçilir (keşif).
            return random.randrange(self.action_space)
        else:
            # Aksi takdirde, modelin tahmin ettiği en iyi aksiyon seçilir (kullanım).
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    # Bellekteki deneyimleri kullanarak modeli güncelleyen fonksiyon.
    def replay(self):
        # Eğer bellekte yeterli sayıda deneyim yoksa, fonksiyon erken sonlanır.
        if len(self.memory) < self.batch_size:
            return

        # Bellekten rastgele bir mini parti seçilir.
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Durumlar ve sonraki durumlar sıkıştırılır (squeeze) - gereksiz boyutlar çıkarılır.
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Hedef Q değerleri hesaplanır.
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        # Seçilen aksiyonlar için hedefler güncellenir.
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        # Model, yeni hedeflerle güncellenir.
        self.model.fit(states, targets_full, epochs=1, verbose=0)

        # Epsilon değeri azaltılır (keşif oranı düşürülür).
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
