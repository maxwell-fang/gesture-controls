from pose_estimation import train_merged, load_model, train_HanCo
if __name__ == '__main__':
    model = load_model('./pose_estimation/models/74.pt')
    train_HanCo(batch_size=48, epochs=26, model=model, start_epoch=74, learning_rate=0.0001, load_weights=False, patience=20)