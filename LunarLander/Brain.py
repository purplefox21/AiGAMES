# Brain.py
import gymnasium as gym  # gymnasium kütüphanesini içe aktarır. Bu kütüphane, çeşitli simülasyon ortamlarını sağlar.
import numpy as np  # NumPy kütüphanesini içe aktarır. Bu kütüphane, bilimsel hesaplamalar için kullanılır.
from DQN import DQN

# get_env fonksiyonu, "Lunar Lander" oyun ortamını oluşturur ve döndürür.
def get_env():
    # 'LunarLander-v2' ortamını yaratır ve bu ortamı 'human' görsel modunda başlatır.
    env = gym.make("LunarLander-v2", render_mode="human")
    return env  # Oluşturulan ortamı döndürür.

# get_initial_state fonksiyonu, ortamın başlangıç durumunu alır ve formatlar.
def get_initial_state(env):
    # Ortamın başlangıç durumunu sıfırlar ve bu durumu alır.
    initial_state = env.reset()

    # Eğer başlangıç durumu bir tuple ise, bu tuple'ın ilk elemanını alır.
    # Bazı ortamlar ekstra bilgilerle tuple döndürebilir. Burada sadece durum bilgisine ihtiyacımız var.
    state = initial_state[0] if isinstance(initial_state, tuple) else initial_state

    # Durum bilgisini yeniden şekillendirir (reshape). Bu, sinir ağının girdi formatına uygun olmalıdır.
    return np.reshape(state, (1, 8))





"""  

Action Space == Discrete(4)

Observation Space == Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32)


Her bir özellik, oyunun fiziksel ortamı ve uzay aracının durumu hakkında bilgi verir. İşte bu 8 boyutlu vektördeki özellikler ve bunların muhtemel anlamları:

X Pozisyonu: Uzay aracının yatay (X ekseninde) konumu. Genellikle -1.5 ile 1.5 arasında bir değer alır.
Y Pozisyonu: Uzay aracının dikey (Y ekseninde) konumu. Genellikle -1.5 ile 1.5 arasında bir değer alır.
X Hızı: Uzay aracının yatay eksen boyunca hızı. Genellikle -5 ile 5 arasında bir değer alır.
Y Hızı: Uzay aracının dikey eksen boyunca hızı. Genellikle -5 ile 5 arasında bir değer alır.

Açı: Uzay aracının açısı. Bu, aracın yatay eksene göre ne kadar eğimli olduğunu gösterir ve genellikle -π ile π arasında bir değer alır (-3.1415927 ile 3.1415927).
Açısal Hız: Uzay aracının açısal hızı. Bu, aracın ne kadar hızlı döndüğünü gösterir ve genellikle -5 ile 5 arasında bir değer alır.

Sol Bacak Durumu: Uzay aracının sol bacağının yerle temas edip etmediğini gösterir. 0, temas yok anlamına gelirken 1, temas olduğunu gösterir.
Sağ Bacak Durumu: Uzay aracının sağ bacağının yerle temas edip etmediğini gösterir. 0, temas yok anlamına gelirken 1, temas olduğunu gösterir.

"""