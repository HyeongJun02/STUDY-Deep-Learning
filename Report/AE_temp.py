import numpy as np
import tensorflow as tf
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 노이즈 추가 함수
def add_noise(x):
    noise_mask = np.random.binomial(1, 0.5, x.shape)
    return x * noise_mask

if __name__ == "__main__":
    print("Denoising AutoEncoder Test 코드")

    # MNIST 데이터 로드
    data_loader = MNISTData()
    data_loader.load_data()

    # 학습된 모델 로드
    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    load_path = "./model/denoising_ae_model.weights.h5"
    auto_encoder.load_weights(load_path)

    print("\n2. 노이즈 이미지 복원")

    num_test_items = 56
    x_test = data_loader.x_test[:num_test_items]
    y_test = data_loader.y_test[:num_test_items]
    x_test_noised = add_noise(x_test)

    reconst_data = auto_encoder.en_decoder.predict(x_test_noised)
    reconst_data = tf.math.sigmoid(reconst_data).numpy()
    x_noised_imgs = x_test_noised.reshape(num_test_items, 28, 28)
    recon_imgs = reconst_data.reshape(num_test_items, 28, 28)
    MNISTData.print_56_pair_images(x_noised_imgs, recon_imgs, y_test)

    print("\n3. 평균 code로 이미지 생성")
    
    # 1000개의 이미지 선별
    x_1000 = data_loader.x_test[:1000]
    y_1000 = data_loader.y_test[:1000]
    codes = auto_encoder.encoder.predict(x_1000)
    
    # 숫자별 평균 저장할 벡터, 숫자 나온 횟수 저장
    avg_code = np.zeros((10, auto_encoder.code_dim))
    count = np.zeros(10)

    # 생성된 code vector의 값을 숫자별 평균 구하기
    for i in range(len(codes)):
        label = y_1000[i]
        avg_code[label] += codes[i]
        count[label] += 1

    for i in range(10):
        if count[i] > 0:
            avg_code[i] /= count[i]
    
    # 평균을 decoder에 넣어서 복원
    decoded_imgs = auto_encoder.decoder.predict(avg_code)
    decoded_imgs = tf.math.sigmoid(decoded_imgs).numpy().reshape(10, 28, 28)
    MNISTData.print_10_images(decoded_imgs, list(range(10)))

    print("\n4. 평균 + 표준편차 code로 이미지 생성")

    # 표준편차 저장할 벡터터
    std_code = np.zeros((10, auto_encoder.code_dim))

    # 표준편차 계산
    for i in range(len(codes)):
        label = y_1000[i]
        std_code[label] += np.square(codes[i] - avg_code[label])
    for i in range(10):
        if count[i] > 0:
            std_code[i] = np.sqrt(std_code[i] / count[i])

    new_codes = []
    for i in range(10):
        for _ in range(5):
            rand_vec = np.random.uniform(low=-1.0, high=1.0, size=(auto_encoder.code_dim,))
            new_code = avg_code[i] + std_code[i] * rand_vec
            new_codes.append(new_code)
    
    new_codes = np.array(new_codes)
    new_dcoded = auto_encoder.decoder.predict(new_codes)
    new_dcoded = tf.math.sigmoid(new_dcoded).numpy().reshape(50, 28, 28)

    label_list = [i for i in range(10) for _ in range(5)] # 5*10 행렬 생성
    MNISTData.print_50_images(new_dcoded, label_list)

    print("\n5. 평균 code t-SNE 시각화")

    # 평균 벡터(10개, 32차원)를 t-SNE 2D로 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    avg_code_2d = tsne.fit_transform(avg_code)

    # 시각화
    plt.figure(figsize=(8,6))
    plt.title("5. 각 숫자별 평균 code t_SNE 시각화", fontsize=14)

    for i in range(10):
        plt.scatter(avg_code_2d[i, 0], avg_code_2d[i, 1], label=str(i), s=100)
        plt.text(avg_code_2d[i, 0]+0.5, avg_code_2d[i, 1], str(i), fontsize=12)

    plt.legend()
    plt.grid(True)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()