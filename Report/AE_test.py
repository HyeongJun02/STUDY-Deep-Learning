# AE_test.py

import tensorflow as tf
import numpy as np
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder

if __name__ == "__main__":
    # ───────────────────────────────────────────────
    # 0) 모델 로드 (이미 학습된 상태라고 가정)
    # ───────────────────────────────────────────────
    data_loader = MNISTData()
    data_loader.load_data()

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    load_path = "./model/ae_model.weights.h5"   # AE_train.py에서 저장된 가중치 경로
    print(f"Loading weights from {load_path} ...")
    auto_encoder.load_weights(load_path)
    print("Weights loaded.\n")

    # ───────────────────────────────────────────────
    # 1) 테스트 데이터 중 1000개 선택
    # ───────────────────────────────────────────────
    num_samples = 1000
    x_test_full = data_loader.x_test      # (10000, 784)
    y_test_full = data_loader.y_test      # (10000,)

    x_subset = x_test_full[:num_samples]  # (1000, 784)
    y_subset = y_test_full[:num_samples]  # (1000,)

    # ───────────────────────────────────────────────
    # 2) Encoder를 통해 1000개에 대한 코드(code) 벡터 생성
    # ───────────────────────────────────────────────
    #    - encoder.predict(x_subset)는 shape=(1000, code_dim) 반환
    codes = auto_encoder.encoder.predict(x_subset)  # (1000, 32)

    # ───────────────────────────────────────────────
    # 3) 클래스별 평균 코드 계산
    # ───────────────────────────────────────────────
    num_classes = 10
    code_dim = auto_encoder.code_dim  # 32

    # 클래스별 코드 합계와 카운트 초기화
    sum_codes = np.zeros((num_classes, code_dim), dtype=np.float32)
    count_codes = np.zeros((num_classes,), dtype=np.float32)

    # 각 샘플마다 (label → codes[i])를 누적
    for i, lbl in enumerate(y_subset):
        sum_codes[lbl] += codes[i]
        count_codes[lbl] += 1.0

    # 클래스별 평균 코드
    avg_codes = np.zeros((num_classes, code_dim), dtype=np.float32)
    for c in range(num_classes):
        if count_codes[c] > 0:
            avg_codes[c] = sum_codes[c] / count_codes[c]
        else:
            # 만약 해당 클래스 샘플이 없으면 0벡터로 남겨둠
            avg_codes[c] = np.zeros((code_dim,), dtype=np.float32)

    # ───────────────────────────────────────────────
    # 4) 평균 코드를 Decoder 입력으로 주어 이미지 생성
    # ───────────────────────────────────────────────
    #    - decoder.predict(avg_codes)는 shape=(10, 784) 반환
    reconst_flat = auto_encoder.decoder.predict(avg_codes)  
    # 디코더 출력에 sigmoid 적용 → [0,1] 범위 값으로 변환
    reconst_flat = tf.math.sigmoid(reconst_flat).numpy()  # (10, 784)

    # 28×28 형태로 reshape
    reconst_images = reconst_flat.reshape((num_classes, data_loader.width, data_loader.height))  # (10,28,28)

    # ───────────────────────────────────────────────
    # 5) 클래스별 평균 코드로 생성된 이미지를 10장 시각화
    # ───────────────────────────────────────────────
    label_list = list(range(num_classes))  # [0,1,2,3,4,5,6,7,8,9]
    MNISTData.print_10_images(reconst_images, label_list)
    print("Finished generating and plotting 10 class-mean images.")
