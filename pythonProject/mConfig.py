class Config1(object):
    def __init__(self):
        super().__init__()
        self.NUM_NET=2
        self.INNER_FEATURE_D=16
        self.INNER_FEATURE_E=16
        self.INNER_FEATURE_A=16
        self.INNER_FEATURE_MA=64 #4(head_num)*INNER_FEATURE_A
        self.FINAL_CLASSES=10

        self.BATCH_SIZE=1
        self.EPOCH=1

        self.LEARNING_RATE=0.001
        self.MOMENTUM=0.5
