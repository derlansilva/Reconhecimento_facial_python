import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
#reconhecedor = cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read("classificadorEigenYale.yml")

reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorFisherYale.yml")

#reconhecedor = cv2.face.LBPHFaceRecognizer_create()
#reconhecedor.read("classificadorLBPHYale.yml")

totalAcerto = 0
percentualAcertos = 0.0
totalConfianca = 0.0

caminhos = [os.path.join("yalefaces/teste" , f) for f in os.listdir("yalefaces/teste")]

for caminhoImagem in caminhos:

    imageFace = Image.open(caminhoImagem).convert('L')

    imagemFaceNP = np.array(imageFace , 'uint8')
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
    for x , y , l , a in facesDetectadas:
        idPrevisto  , confianca = reconhecedor.predict(imagemFaceNP)
        idAtual = int(os.path.split(caminhoImagem)[1].split('.')[0].replace("subject" , ""))
        print(str(idAtual)+ "Foi classificado como" + str(idPrevisto)+ "-" + str(confianca))

        if idPrevisto == idAtual:
            totalAcerto+= 1
            totalConfianca = confianca

percentualAcertos = (totalAcerto/30)*100
totalConfianca = totalConfianca/totalAcerto

print("Percentual de acerto : "+ str(percentualAcertos))
print("Total confiança " + str(totalConfianca))
