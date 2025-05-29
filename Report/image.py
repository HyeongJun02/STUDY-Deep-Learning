auto_encoder.fit(
    x = x_train_noisy,  # Noise Adder 적용된 입력
    y = x_train,        # 원본 복원을 목표로 함
    batch_size = 32,
    epochs     = num_epochs
)
