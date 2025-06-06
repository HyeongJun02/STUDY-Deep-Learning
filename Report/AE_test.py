import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from MNISTData import MNISTData
from AutoEncoder import AutoEncoder

from sklearn.manifold import TSNE
import matplotlib.cm as cm


def load_denoising_ae(model_path: str):
    data_loader = MNISTData()
    data_loader.load_data()

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {model_path}")
    auto_encoder.load_weights(model_path)

    return data_loader, auto_encoder


def q2_noisy_reconstruct(data_loader: MNISTData, auto_encoder: AutoEncoder):
    print("===== [Q2] Denoising 성능 확인 =====")
    # 2-1) 테스트 56개 선택
    num_test_items = 56
    x_test = data_loader.x_test[:num_test_items]
    y_test = data_loader.y_test[:num_test_items]

    # 2-2) Noise Adder: 픽셀당 0.5 확률로 0 마스킹
    p_zero = 0.5
    mask = np.random.binomial(n=1, p=1 - p_zero, size=x_test.shape)
    x_test_noisy = x_test * mask

    # 2-3) 재구성 수행: en_decoder -> sigmoid -> (56, 784) -> reshape (56,28,28)
    reconst_flat = auto_encoder.en_decoder.predict(x_test_noisy)
    reconst_flat = tf.math.sigmoid(reconst_flat).numpy()
    reconst_imgs = reconst_flat.reshape((num_test_items,
                                         data_loader.width,
                                         data_loader.height))

    # 2-4) 시각화: 56개 노이즈 -> 재구성 페어
    MNISTData.print_56_pair_images(
        x_test_noisy.reshape((num_test_items, data_loader.width, data_loader.height)),
        reconst_imgs,
        y_test.tolist()
    )
    print("===== [Q2] 완료 =====")


def q3_class_mean(data_loader: MNISTData, auto_encoder: AutoEncoder):
    print("===== [Q3] 각 숫자(class)별 code 평균값으로 이미지 생성 =====")
    # 3-1) 테스트 1000개 선택
    num_samples = 1000
    x_sub = data_loader.x_test[:num_samples]
    y_sub = data_loader.y_test[:num_samples]

    # 3-2) Encoder로 1000개 코드 생성 (shape = (1000, code_dim))
    codes = auto_encoder.encoder.predict(x_sub, batch_size=128, verbose=0)

    # 3-3) 클래스별 합계 및 개수 누적
    num_classes = 10
    code_dim = auto_encoder.code_dim
    sum_codes = np.zeros((num_classes, code_dim), dtype=np.float32)
    count_codes = np.zeros((num_classes,), dtype=np.float32)

    for i, lbl in enumerate(y_sub):
        sum_codes[lbl] += codes[i]
        count_codes[lbl] += 1.0

    # 3-4) 클래스별 평균 코드 계산
    avg_codes = np.zeros((num_classes, code_dim), dtype=np.float32)
    for c in range(num_classes):
        if count_codes[c] > 0:
            avg_codes[c] = sum_codes[c] / count_codes[c]

    # 3-5) 평균 코드 -> 디코더 복원 -> (10, 784) -> sigmoid -> (10, 28, 28)
    reconst_flat = auto_encoder.decoder.predict(avg_codes, batch_size=num_classes)
    reconst_flat = tf.math.sigmoid(reconst_flat).numpy()
    reconst_imgs = reconst_flat.reshape((num_classes,
                                         data_loader.width,
                                         data_loader.height))

    # 3-6) 시각화: 클래스 0~9 평균 코드 이미지 10개
    MNISTData.print_10_images(
        reconst_imgs,
        list(range(num_classes))
    )
    print("===== [Q3] 완료 =====")

    return avg_codes, codes, y_sub


def q4_class_variation(data_loader: MNISTData, auto_encoder: AutoEncoder,
                       avg_codes: np.ndarray, codes: np.ndarray, y_sub: np.ndarray):
    print("===== [Q4] 각 숫자(class)별 code 평균값 및 표준편차를 활용하여 새로운 숫자 생성 =====")
    num_classes = 10
    code_dim = auto_encoder.code_dim
    num_variations = 5

    # 4-1) 클래스별 코드 벡터 리스트 생성 -> 표준편차 계산
    codes_by_class = {c: [] for c in range(num_classes)}
    for i, lbl in enumerate(y_sub):
        codes_by_class[lbl].append(codes[i])
    std_codes = np.zeros((num_classes, code_dim), dtype=np.float32)
    for c in range(num_classes):
        class_codes = np.stack(codes_by_class[c], axis=0)
        std_codes[c] = np.std(class_codes, axis=0)

    # 4-2) rand 벡터 생성: 클래스별 5개, 각 원소 ∈ [-1,1]
    rand_vectors = {
        c: np.random.uniform(low=-1.0, high=1.0, size=(num_variations, code_dim))
        for c in range(num_classes)
    }

    # 4-3) avg + std × rand -> variation 코드 (shape = (10, 5, 32))
    codes_var = np.zeros((num_classes, num_variations, code_dim), dtype=np.float32)
    for c in range(num_classes):
        for j in range(num_variations):
            codes_var[c, j] = avg_codes[c] + std_codes[c] * rand_vectors[c][j]

    # 4-4) variation 코드 -> 디코더 복원 -> (50, 784) -> sigmoid -> (50, 28, 28)
    codes_var_flat = codes_var.reshape((num_classes * num_variations, code_dim))
    reconst_flat = auto_encoder.decoder.predict(codes_var_flat, batch_size=50)
    reconst_flat = tf.math.sigmoid(reconst_flat).numpy()
    reconst_imgs = reconst_flat.reshape((num_classes * num_variations,
                                         data_loader.width,
                                         data_loader.height))

    # 4-5) 시각화: 10행×5열 그리드
    fig, axes = plt.subplots(nrows=num_classes, ncols=num_variations,
                             figsize=(num_variations * 1.5, num_classes * 1.5))

    for c in range(num_classes):
        for j in range(num_variations):
            ax = axes[c, j]
            img = reconst_imgs[c * num_variations + j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(str(c), fontsize=12, rotation=0, labelpad=10, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("===== [Q4] 완료 =====")


def q5_tsne_visualization(avg_codes: np.ndarray):
    print("===== [Q5] 각 숫자(class) 별 code 평균값 위치관계 확인 =====")
    num_classes = avg_codes.shape[0]
    labels = list(range(num_classes))

    # 5-1) t-SNE로 2차원 축소 (perplexity, learning_rate 등은 필요에 따라 조정 가능)
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, random_state=42)
    codes_2d = tsne.fit_transform(avg_codes)

    # 5-2) 산점도 그리기
    plt.figure(figsize=(8, 6))
    colors = cm.get_cmap('tab10', num_classes)
    for i in range(num_classes):
        x, y = codes_2d[i]
        plt.scatter(x, y, color=colors(i), label=str(i), s=100)

        # 각 점 옆에 레이블 쓰기
        plt.text(x + 1.0, y + 1.0, str(i), fontsize=12, weight='bold')

    plt.title("t-SNE Visualization of Avg Codes (per digit)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Digit", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("===== [Q5] 완료 =====")


def main():
    model_path = "./model/ae_model.weights.h5"
    data_loader, auto_encoder = load_denoising_ae(model_path)

    # [Q2]
    q2_noisy_reconstruct(data_loader, auto_encoder)

    # [Q3]
    avg_codes, codes_1000, y_sub_1000 = q3_class_mean(data_loader, auto_encoder)

    # [Q4]
    q4_class_variation(data_loader, auto_encoder, avg_codes, codes_1000, y_sub_1000)

    # [Q5]
    q5_tsne_visualization(avg_codes)

    print("=== AE_test.py (문제 2~5) 모든 단계 완료 ===")


if __name__ == "__main__":
    main()
