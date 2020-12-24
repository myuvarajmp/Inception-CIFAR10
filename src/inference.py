from .model import getModel
from pathlib import Path
import PIL
import torch
from torchvision import  transforms


class Inference:
    #print(modelno)
    print('Inference hello')
    def __init__(self, modelnumber, save_model_filename="saved_weights.pt"):
        self.modelnumber = modelnumber
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
        self.model = getModel(training=False, modelno = self.modelnumber, num_classes=len(self.classes))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict1 = {'inception' : 1, 'inceptionv3' : 2} 
        if dict1['inception'] == self.modelnumber:
            print('Inception Hello')
            self.model.load_state_dict(torch.load(f"./src/saved_weights/inception.pt",map_location=device))
        if dict1['inceptionv3'] == self.modelnumber:
            print('Inception v3 Hello')
            self.model.load_state_dict(torch.load(f"./src/saved_weights/inceptionv3.pt",map_location=device))
        return None

    def __call__(self,image):
        if isinstance(image,(Path,str)):
            image = PIL.Image.open(image).convert("RGB")
        elif not isinstance(image,PIL.JpegImagePlugin.JpegImageFile): 
            raise Exception("must be PIL image or path ")
        image_input = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(image_input).squeeze(0)
            prob = torch.argmax(out).item()
            return self.classes[prob]

