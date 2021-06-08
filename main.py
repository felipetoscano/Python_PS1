import cv2
import dlib
import numpy

def shapeToNp(shape):
    coords = numpy.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def coordinatesEyes(shape):
    margin_width  = 15
    margin_height = 10

    x = (shape[37][0]) - margin_width
    y = (shape[37][1]) - margin_height

    x1 = (shape[46][0]) + margin_width
    y1 = (shape[46][1]) + margin_height

    return x, y, x1, y1

def maskEyes(mask, side, shape):
    points = [shape[i] for i in side]
    points = numpy.array(points, dtype=numpy.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def circleEye(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1)
    except:
        return 0, 0
    return cx, cy

def averagePoints(p1, p2):
    return (p1 + p2) / 2

#Inicializando detector de face
detector = dlib.get_frontal_face_detector()

#Carregando shape
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Pontos de shape do olho esquerdo
leftEye = [36, 37, 38, 39, 40, 41] 

#Pontos de shape do olho direito
rightEye = [42, 43, 44, 45, 46, 47] 

#Kernel
kernel = numpy.ones((9, 9), numpy.uint8)

#Dimensões da imagem de direção dos olhos
widhtImgEyesDirecion = 800
heightImgEyesDirecion = 600

videoCapture = cv2.VideoCapture(0)

#Vídeo de entrada de exemplo
#videoCapture = cv2.VideoCapture("video.mp4")

if videoCapture.isOpened():
    isOpened, frame = videoCapture.read()
else:
    isOpened = False

while isOpened:
    #Resize do vídeo de entrada de exemplo
    #heightFrame, widthFrame = frame.shape[:2]
    #frame = cv2.resize(frame, (int(widthFrame / 3), int(heightFrame / 3)))

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Imagem usada para desenhar a região dos olhos
    imgBoundingBoxes = frame.copy()

    #Imagem da região dos olhos
    imgEyes = frame.copy()

    #Imagem da direção dos olhos
    imgEyesDirection = numpy.zeros([heightImgEyesDirecion, widhtImgEyesDirecion, 3])
    imgEyesDirection[::] = (255, 255, 255)

    faces = detector(frame)
    for face in faces:
        shape = predictor(imgGray, face)

        #Transforma o shape para uma lista mais fácil de manipular
        shape = shapeToNp(shape)

        #Dimensões do frame apenas dos olhos
        x, y, x1, y1 = coordinatesEyes(shape)

        #Criando fundo branco ao redor dos pontos dos olhos
        mask = numpy.zeros(imgEyes.shape[:2], dtype=numpy.uint8)
        mask = maskEyes(mask, leftEye, shape)
        mask = maskEyes(mask, rightEye, shape)

        #Filtrando a íris
        mask   = cv2.dilate(mask, kernel, 5)
        eyes   = cv2.bitwise_and(imgEyes, imgEyes, mask=mask)
        mask   = (eyes == [0, 0, 0]).all(axis=2)

        eyes[mask] = [255, 255, 255] 
        eyesGray  = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        _, processedImage  = cv2.threshold(eyesGray, 80, 255, cv2.THRESH_BINARY)
        processedImage = cv2.medianBlur(processedImage, 7)
        processedImage = cv2.dilate(processedImage, None, iterations=5)
        processedImage = cv2.erode(processedImage, None, iterations=4)
        processedImage = cv2.bitwise_not(processedImage)

        mid = (shape[42][0] + shape[39][0]) // 2
        cx1, cy1 = circleEye(processedImage[:, 0:mid], mid, imgEyes)
        cx2, cy2 = circleEye(processedImage[:, mid:], mid, imgEyes, True)

        #Desenho da imagem da região dos olhos
        cv2.rectangle(imgBoundingBoxes, (x, y), (x1, y1), (0, 0, 255), 1)
        count = 36
        for i in shape[36:48]:
            cv2.circle(imgBoundingBoxes, (i[0], i[1]), 2, (0, 0, 255), -1)
            cv2.putText(imgBoundingBoxes, str(count), (i[0], i[1]), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 0, 255))
            count = count + 1

        #Recortando região dos olhos
        imgEyes = imgEyes[y:y1, x:x1]

        #Media de coordenadas de direções dos olhos
        averageLeft   = averagePoints(shape[36][0], shape[42][0])
        averageRight  = averagePoints(shape[39][0], shape[45][0])
        averageTop    = averagePoints(shape[37][1], shape[43][1])
        averageBottom = averagePoints(shape[41][1], shape[47][1])

        #Média de coordenadas que os olhos estão olhando (no frame completo)
        averageEyesLookingX  = averagePoints(cx1, cx2)
        averageEyesLookingY  = averagePoints(cy1, cy2)

        #Distancia dos olhos de cada ponto
        eyesDistanceLeft   = averageEyesLookingX - averageLeft
        eyesDistanceRight  = averageRight - averageEyesLookingX
        eyesDistanceBottom = averageBottom - averageEyesLookingY
        eyesDistanceTop    = averageEyesLookingY - averageTop 

        direction = ""
        if(eyesDistanceLeft < eyesDistanceRight and eyesDistanceLeft <= 20):
            direction = "esquerda"
            averageEyesLookingX = averageEyesLookingX / 1.5
        elif(eyesDistanceRight < eyesDistanceLeft and eyesDistanceRight <= 20):
            direction = "direita"
            averageEyesLookingX = averageEyesLookingX * 1.5
        elif((averageBottom - averageTop) <= 12):
            direction = "baixo"
            averageEyesLookingY = averageEyesLookingY * 1.5
        elif(eyesDistanceTop < eyesDistanceBottom and eyesDistanceTop <= 5):
            direction = "cima"
            averageEyesLookingY = averageEyesLookingY / 1.5
        else:
            direction = "centro"

        #Tamanho da imagem dos olhos
        heightFrame, widthFrame = frame.shape[:2]

        #Numero de vezes que a imagem é maior no x
        circleImgEyesDirectionX = int((widhtImgEyesDirecion / widthFrame) * averageEyesLookingX)

        #Numero de vezes que a imagem é maior no y
        circleImgEyesDirectionY = int((heightImgEyesDirecion / heightFrame) * averageEyesLookingY)

        cv2.circle(imgEyesDirection, (circleImgEyesDirectionX, circleImgEyesDirectionY), 10, (0, 0, 255), -1)
        cv2.putText(imgEyesDirection, direction, (350, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))


    cv2.imshow("Bounding Boxes", imgBoundingBoxes)
    cv2.imshow("Olhos", imgEyes)
    cv2.imshow("Original", frame)
    cv2.imshow("Sentido dos olhos", imgEyesDirection)
    isOpened, frame = videoCapture.read()

    key = cv2.waitKey(5)
    if key == 27:
        break

videoCapture.release()
cv2.destroyAllWindows()