# AE_train.py

from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import numpy as np

if __name__ == "__main__":
    print("Hi. I am an Auto Encoder Trainer.")
    batch_size = 32
    num_epochs = 5

    data_loader = MNISTData()
    data_loader.load_data()
    
    x_train = data_loader.x_train  # (N, 784) 형태, 0~1 사이값

    # ────────────────────────────────────────────────────────────
    # Noise Adder: 각 픽셀을 0으로 만들 확률 p_zero = 0.5
    # mask 값이 1일 때는 원본 유지, 0일 때는 0으로
    p_zero = 0.5
    mask = np.random.binomial(1, 1 - p_zero, size=x_train.shape)
    x_train_noisy = x_train * mask
    # ────────────────────────────────────────────────────────────

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    auto_encoder.fit(
        x = x_train_noisy,  # 노이즈가 추가된 입력
        y = x_train,        # 원본을 복원하도록 학습
        batch_size = batch_size,
        epochs = num_epochs
    )

    save_path = "./model/ae_model.weights.h5"
    auto_encoder.save_weights(save_path)
    print("Saved model weights to %s" % save_path)
