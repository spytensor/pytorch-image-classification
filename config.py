class DefaultConfigs(object):
    #1.string parameters
    train_data = "../data/train/"
    test_data = ""
    val_data = "../data/val/"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"

    #2.numeric parameters
    epochs = 40
    batch_size = 4
    img_height = 224
    img_weight = 224
    num_classes = 62
    seed = 888
    lr = 1e-3
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
