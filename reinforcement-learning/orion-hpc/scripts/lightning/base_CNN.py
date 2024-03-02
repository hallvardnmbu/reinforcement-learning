import torch


# disgusting placeholder
class BaseCNN(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.add_layers()

    def add_layers(self):
        self.layers.append(torch.nn.Conv2d(1, 32, kernel_size=8, stride=4))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Flatten())
        self.layers.append(torch.nn.Linear(3136, 512))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(512, 5))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Example usage
if __name__ == "__main__":
    # Assuming some dummy input for demonstration
    dummy_input = torch.randn(1, 1, 84, 84)  # Example input tensor
    model = BaseCNN()
    output = model(dummy_input)
    print(output.shape)  # Should match the expected output shape based on the network configuration
