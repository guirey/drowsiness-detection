#Algoritmo que realiza a localização da face atraves 
#de treinamento via import de XML
#Desenvolvido por Fernando M. Wittmann 06.06.2020
#Adaptado por Guilherme Rey e Giovanni Bovolato 31.10.2023 

#importação de bibliotecas
import cv2, os

#Funcao que realiza a busca de arquivos nos diretorios do computador
def find(name, path):
    for root, dirs, files in os.walk(path):
        if (name in files) or (name in dirs):
            print("O arquivo {} esta em {}".format(name, root))
            return os.path.join(root, name)
    #No caso de nao encontrar o arquivo, recursao para diretorios anteriores
    return find(name, os.path.dirname(path))

#Importa o arquivo XML encontrado
cv2path = os.path.dirname(cv2.__file__)
haar_path = find('haarcascades', cv2path)
xml_name = 'haarcascade_frontalface_alt2.xml'
xml_path = os.path.join(haar_path, xml_name)

clf = cv2.CascadeClassifier(xml_path)               #Inicializa o classificador
cap = cv2.VideoCapture(0)                           #Inicializa a webcam

#Looping para leitura do conteúdo
while(not cv2.waitKey(20) & 0xFF == ord('s')):
        ret, frame = cap.read()                       #Captura o proximo frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Converte a imagem para tons de cinza
        rostos = clf.detectMultiScale(gray)         #Classifica
        for x, y, w, h in rostos:                   #Desenha um retangulo na face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Detector de Face',frame)        #Plota a imagem 

#Desliga a webcam
cap.release()

#Fecha a janela de video
cv2.destroyAllWindows()
cv2.waitKey(1)