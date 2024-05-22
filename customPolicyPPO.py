from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * observation_space.shape[1] * observation_space.shape[2], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.cnn(observations)

# Custom Policy that uses the CNN
class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_class=CustomCNNFeaturesExtractor, **kwargs):
        super(CustomCNNPolicy, self).__init__(observation_space, action_space, lr_schedule, features_extractor_class=features_extractor_class, **kwargs)