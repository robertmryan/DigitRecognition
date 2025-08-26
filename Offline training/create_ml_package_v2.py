import torch, torch.nn.functional as F, coremltools as ct
from mnist_cnn_mps import SmallCNN

model = SmallCNN().eval()
model.load_state_dict(torch.load("artifacts/mnist_smallcnn_mps.pt", map_location="cpu"))

class ProbHead(torch.nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, x):
        logits = self.core(x)
        return F.softmax(logits, dim=1)   # <-- tensor only

wrapped = ProbHead(model).eval()
example = torch.randn(1,1,28,28)
traced = torch.jit.trace(wrapped, example)

mean, std = 0.1307, 0.3081
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image", shape=(1,1,28,28), color_layout="G",
        scale=1.0/(255*std), bias=[-mean/std]
    )],
    # Name the single output so ClassifierConfig can find it:
    outputs=[ct.TensorType(name="probabilities")],
    classifier_config=ct.ClassifierConfig(
        class_labels=[str(i) for i in range(10)],
        predicted_feature_name="classLabel",
        predicted_probabilities_output="probabilities",
    ),
)
mlmodel.save("artifacts/MNISTSmallCNN_v2.mlpackage")