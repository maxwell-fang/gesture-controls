from pose_estimation import load_model, train_HanCo
if __name__ == '__main__':
    model = load_model('./pose_estimation/models/12.pt')
    train_HanCo(batch_size=32, epochs=88, model=model, start_epoch=12, learning_rate=0.001, load_weights=False, patience=20)