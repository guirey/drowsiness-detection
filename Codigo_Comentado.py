# Rede que faz a junção entre 3 redes, sendo uma de detecção da face
# outra de detecção dos olhos e a última uma CNN que faz a classificaçao
# do estado do olho, sendo aberto ou fechado
# Desenvolvido por Christian Haller 27.06.2020
# Adaptado por Guilherme Rey e Giovanni Bovolato 02.11.2023


#importação das bibliotecas a serem utilizadas

import cv2, os, time, datetime
import numpy as np
from keras.models import load_model
from pygame import mixer

# Configuração da definição do sistema de alarme sonoro do dispositivo

mixer.init()
alarme = mixer.Sound(f'{os.getcwd()}\\Data\\drowsiness_data\\bip.wav')
alarme.play()

#importação dos XML utilizados para detecção de rosto e olhos

rosto = cv2.CascadeClassifier(f'{os.getcwd()}\\Data\\drowsiness_data\\haarcascade_frontalface_alt.xml')
olhoesq = cv2.CascadeClassifier(f'{os.getcwd()}\\Data\\drowsiness_data\\haarcascade_lefteye_2splits.xml')
olhodir = cv2.CascadeClassifier(f'{os.getcwd()}\\Data\\drowsiness_data\\haarcascade_righteye_2splits.xml')

#carregando a rede treinada

rede = load_model(f'{os.getcwd()}<inserir o caminho do arquivo .h5 da rede treinada>')
path = os.getcwd()

# obtendo o video através de uma camera IP
cap = cv2.VideoCapture(0)
if cap.isOpened() == True:                      #verifica se foi possivel abrir a tela de video com sucesso
    print("Captura de video aberta com sucesso.")
else:                                           #retorna mensagem de erro caso nao seja possivel abrir a tela de video com sucesso
    print("Problemas ao abrir o video.")

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL           #define a fonte a ser utilizada

# define o valor inicial para as variaveis de inferencia de olhos fechados
valor = 0
limite = 6
espessura = 2
dirpred = [99]
esqpred = [99]

# inicia o ciclo de predição:

# looping infinito de captura de imagem, inferencia e pontuação
while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]         #ALTURA E LARGURA

    # converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica o HaarCascade para imagens monocromaticas
    rostos = rosto.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    olho_esquerdo = olhoesq.detectMultiScale(gray)
    olho_direito = olhodir.detectMultiScale(gray)

    # desenha black bars em cima e embaixo
    cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), thickness=cv2.FILLED)

    # desenha o retangulo no rosto
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #ALTERA A COR DO RETANGULO

    # com o olho direito detectado, faz o pré processamento e realiza a predição com a CNN
    for (x, y, w, h) in olho_direito:
        olho_d = frame[y:y + h, x:x + w]
        olho_d = cv2.cvtColor(olho_d, cv2.COLOR_BGR2GRAY)
        olho_d = cv2.resize(olho_d, (24, 24))
        olho_d = olho_d / 255
        olho_d = olho_d.reshape(24, 24, -1)
        olho_d = np.expand_dims(olho_d, axis=0)
        dirpred = np.argmax(rede.predict(olho_d), axis=1)
        break

    # com o olho esquerdo detectado, faz o pré processamento e realiza a predição com a CNN
    for (x, y, w, h) in olho_esquerdo:
        olho_e = frame[y:y + h, x:x + w]
        olho_e = cv2.cvtColor(olho_e, cv2.COLOR_BGR2GRAY)
        olho_e = cv2.resize(olho_e, (24, 24))
        olho_e = olho_e / 255
        olho_e = olho_e.reshape(24, 24, -1)
        olho_e = np.expand_dims(olho_e, axis=0)
        esqpred = np.argmax(rede.predict(olho_e), axis=1)
        break

    #Se ambos os olhos se encontram fechados, muda para FECHADO na tela e adiciona pontos ao contador de fadiga
    if (dirpred[0] == 0 and esqpred[0] == 0):
        valor += 1
        cv2.putText(frame, "FECHADO", (10, height - 20), fonte, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #Caso o valor estoure o limite, assume o limite como o valor
        if valor > limite + 1:
            valor = limite
    else:
        valor -= 1
        cv2.putText(frame, "ABERTO", (10, height - 20), fonte, 1, (255, 255, 255), 1, cv2.LINE_AA)

    #Escreve o valor atual do nivel de fadiga na tela
    if (valor < 0):
        valor = 0
    cv2.putText(frame, 'Nivel de Fadiga:' + str(valor), (400, height - 20), fonte, 1, (255, 255, 255), 1, cv2.LINE_AA)

    #Caso o valor atinja o limite definido, dispara os alarmes sonoros e visuais e salva o frame em que isso ocorreu
    if (valor > limite):
        ts = time.time()                  #alteracoes novas linhas 112 a 115
        st = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y__%H-%M-%S')
        nome_arquivo = str("Olho_Fechado") + " " + str(st)
        cv2.imwrite(os.path.join(path, + str(file_name) + 'frame.jpg'), frame)
        #cv2.imwrite(os.path.join(path, 'closed_eyes_screencap.jpg'), frame)
        try:
            alarme.play()
        except:  
            pass

        #Adiciona um retangulo vermelho como sinal de alerta na tela
        if (espessura < 16):
            espessura += 2
        #Afina mais o retangula para uma impressao de piscante
        else:
            espessura -= 2
            if (espessura < 2):
                espessura = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness=espessura)

    #plota o frame final com todos os elementos na tela
    cv2.imshow('frame', frame)

    #sai do looping infinito pressionando a tecla S
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

#fecha a captura de video e as janelas
cap.release()
cv2.destroyAllWindows()
