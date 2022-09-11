import os

dimensions = {'Pybullet': {'obs': 18,
                           'obs_extra_info': 18,
                           'acts': 7,
                           'achieved_goals': 11,
                           'achieved_goals_extra_info': 11,
                           'shoulder_img_hw': 200,
                           'hz': 25}}
class GlobalConfig:
    # Data
    min_window_size = 20
    max_window_size = 40
    use_image = True
    root = '../data1'
    train_data = ['../data1/UR5', '../data1/UR5_high_transition']
    validation_data = ['../data1/UR5_validate']

    # Dimension
    obs_dim = 18
    act_dim = 7


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
