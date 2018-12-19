class DefaultConfigs(object):
    #1.string parameters
    train_data = "/home/user/zcj/tutorials/data/train/"
    test_data = ""
    val_data = "/home/user/zcj/tutorials/data/val/"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"

    #2.numeric parameters
    epochs = 40
    batch_size = 16
    img_height = 300
    img_weight = 300
    num_classes = 62
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
