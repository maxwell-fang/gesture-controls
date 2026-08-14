from pose_estimation import load_model, train_HanCo
if __name__ == '__main__':
    model = load_model('./pose_estimation/models/48.pt')
    train_HanCo(batch_size=48, epochs=52, model=model, start_epoch=48, learning_rate=0.0005, load_weights=False, patience=20)