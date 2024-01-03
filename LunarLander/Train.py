# Gerekli kütüphaneler ve modüller içe aktarılıyor.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQN  # DQN sınıfınızın bulunduğu yer

# get_env ve get_initial_state fonksiyonlarınızın tanımları burada olmalı

def get_env():
    env = gym.make("LunarLander-v2", render_mode="human")
    
    return env

def get_initial_state(env):
    initial_state = env.reset()
    state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
    return np.reshape(state, (1, 8))

# DQN ajanını eğiten fonksiyon.
def train_dqn(episode):
    env = get_env()  # Ortam yaratılıyor
    agent = DQN(env.action_space.n, env.observation_space.shape[0])  # DQN ajanı oluşturuluyor
    scores = []  # Her bölümün skorlarını saklamak için liste
    epsilon_values = []  # Her bölümdeki epsilon değerlerini saklamak için liste

    for e in range(episode):
        state = get_initial_state(env)  # Başlangıç durumu alınıyor
        score = 0  # Bu bölüm için skor
        

        max_steps = 1000  # Maksimum adım sayısı
        for i in range(max_steps):
            action = agent.act(state)  # Ajan bir aksiyon seçiyor
            env.render()  # Ortamın görselleştirilmesi
            result = env.step(action)  # Seçilen aksiyon ortama uygulanıyor
            next_state, reward, done, _ = result[:4]  # Sonraki durum, ödül, ve oyunun bitip bitmediği alınıyor
            next_state = np.reshape(next_state, (1, 8))  # Sonraki durum yeniden şekillendiriliyor

            # Ajanın deneyimini belleğe kaydediyoruz
            agent.remember(state, action, reward, next_state, done)
            state = next_state  # Mevcut durumu güncelliyoruz
            agent.replay()  # Ajanın modelini güncelliyoruz
            
            

            score += reward # Skoru güncelle
            if done:
                break  # Eğer oyun bitti ise döngüyü sonlandır

        scores.append(score)  # Skoru listeye ekle
        epsilon_values.append(agent.epsilon)  # Epsilon değerini listeye ekle
        print(f"Episode: {e+1}/{episode}, Score: {score}, Epsilon: {agent.epsilon}")

    return scores, epsilon_values, agent  # Skorlar, epsilon değerleri ve ajanı döndürüyoruz

# Ana program bloğu
if __name__ == '__main__':
    episodes = 25
    scores, epsilon_values, agent = train_dqn(episodes)  # Ajanı eğitiyoruz

    # Görselleştirme
    plt.figure(figsize=(12, 6))
    
    # Eğitim Sürecindeki Skorların Zaman Serisi Grafiği
    plt.subplot(2, 1, 1)
    plt.plot(scores)
    plt.title("Her Bölümdeki Skorlar")
    plt.xlabel("Bölüm")
    plt.ylabel("Skor")

    # Epsilon Değerinin Değişimi
    plt.subplot(2, 1, 2)
    plt.plot(epsilon_values)
    plt.title("Epsilon Değerinin Zaman İçindeki Değişimi")
    plt.xlabel("Bölüm")
    plt.ylabel("Epsilon Değeri")

    plt.tight_layout()
    plt.show()

    # Modeli kaydetme
    model_name = "25bolum_1000.h5"  # İstediğiniz dosya adını burada belirleyin
    agent.model.save(model_name)
    print(f"Model saved as {model_name}")
