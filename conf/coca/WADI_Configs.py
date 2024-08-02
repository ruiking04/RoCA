class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'WADI'
        # model configs
        self.input_channels = 127
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32
        self.hidden_size = 64
        self.num_layers = 3
        self.project_channels = 64

        self.dropout = 0.45
        self.window_size = 32
        self.time_step = 16

        # training configs
        self.num_epoch = 50
        self.change_center_epoch = 10
        self.center_eps = 0.01
        self.omega = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # training strategy
        self.warmup = 10
        # Specify train type ("original", "soft_boundary", "refine", "loe_ts", "loe_soft")
        self.train_method = 'loe_ts'
        self.nu = 0.001
        # loe_ts parameters
        self.mu = 0.85
        # soft_boundary parameters
        self.phi = 0.01
        self.freeze_length_epoch = 2

        # Anomaly Detection parameters
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0005
        # Methods for determining thresholds ("fix","floating","one_anomaly")
        self.threshold_determine = 'floating'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.scale_ratio = 0.8
        self.jitter_ratio = 0.2

