from pose_estimation import load_model, train_HanCo
if __name__ == '__main__':
    # model = load_model('./pose_estimation/models/74.pt')
    train_HanCo(batch_size=48, epochs=100, model=None, start_epoch=0, learning_rate=0.001, load_weights=False, patience=20)