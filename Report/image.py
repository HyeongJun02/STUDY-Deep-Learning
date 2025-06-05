auto_encoder.fit(                       # type: ignore
    x = x_train_noisy,                      # type: ignore
    y = x_train,                      # type: ignore
    batch_size = 32,
    epochs     = num_epochs                      # type: ignore
)
