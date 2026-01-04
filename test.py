

import torch

from torchvision import transforms

from PIL import Image

from SimpleConvolutionalNetwork import SimpleConvolutionalNetwork


transform = transforms.Compose([

    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)

])



def testando_imagens(caminho_imagem):

    image = Image.open(caminho_imagem).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)


    with torch.no_grad():

        output = model(image)

        _, predicted = torch.max(output,1)

    
    return predicted.item()





model = SimpleConvolutionalNetwork()

model.load_state_dict(torch.load("modelo_frutas.pth", weights_only= True))

model.eval()


classes = {

    0 : "maça",

    1 : "banana"
}


testando= testando_imagens("Test_images/bananinha2.jpg")


print("A fruta na imagem é uma:", classes[testando])
