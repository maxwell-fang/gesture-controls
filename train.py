from pose_estimation import load_model, train_HanCo
if __name__ == '__main__':
    model = load_model('./pose_estimation/models/79.pt')
    train_HanCo(batch_size=32, epochs=100, model=model, start_epoch=100, learning_rate=0.0001, load_weights=False, patience=20)