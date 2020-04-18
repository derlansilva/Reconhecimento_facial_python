import cv2
import numpy as np 

classificador = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_eye.xml")

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostra = 25
id= input("insira um identificador")
largura , altura = 220 , 220
print("Capiturando as faces")

while True:
    connectado , imagem = camera.read() 
    imagemCinza = cv2.cvtColor(imagem , cv2.COLOR_BGR2GRAY)
    facesdetectadas = classificador.detectMultiScale(imagemCinza , scaleFactor=1.1 , minSize=(150 , 150))

    for x , y , l , a in facesdetectadas:
        cv2.rectangle(imagem , (x , y) , (x+l , y+a) , (0,0,255), 2)
        regiao =imagem[y:y+a , x:x+l]

        regiaoCinzaOlho = cv2.cvtColor(regiao , cv2.COLOR_BGR2GRAY)
        olhosdetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
        for ox , oy , ol , oa in olhosdetectados:
            cv2.rectangle(regiao , (ox,oy) , (ox+ol , oy +oa) , (0,255 , 0) , 2)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                if np.average(imagemCinza)> 110:
                    imagemface = cv2.resize(imagemCinza[y:y+a , x:x+l] , (largura , altura))
                    cv2.imwrite("fotos/pessoa."+str(id) + "." + str(amostra)+ ".jpg" , imagemface)
                    print("[foto " + str(amostra )+ "Capiturada")
                    amostra += 1

            elif cv2.waitKey(1) & 0xFF == ord("d"):
                break

    cv2.imshow("face" , imagem)
    if amostra >= numeroAmostra+1:
        break

print("Faces capituradas com sucesso")
camera.release()
cv2.destroyAllWindows()
