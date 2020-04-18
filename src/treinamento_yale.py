import  cv2
import  os
import  numpy as np
from  PIL import  Image

eigeface = cv2.face.EigenFaceRecognizer_create(40 , 80000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 20000)
Lbph = cv2.face.LBPHFaceRecognizer_create(2, 2,7 ,7,80)

def getImageCommit():
    caminhos = [os.path.join('yalefaces/treinamento' , f ) for f in os.listdir('yalefaces/treinamento')]
    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imageFace = Image.open(caminhoImagem).convert('L')
        imagemNp = np.array(imageFace , 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split('.')[0].replace("subject" , ""))
        ids.append(id)
        faces.append(imagemNp)


    return  np.array(ids) , faces


id , faces = getImageCommit()
print("Treinando")

eigeface.train(faces , id)
eigeface.write('classificadorEigenYale.yml')


fisherface.train(faces , id)
fisherface.write('classificadorFisherYale.yml')

Lbph.train(faces , id)
Lbph.write('classificadorLBPHYale.yml')

print("Trainamento ja realizado")
