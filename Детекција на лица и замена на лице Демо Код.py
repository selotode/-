#Andrej Skenderski 181117
#Ivan Markovski 185051
#Код За детекција на лица и замена на лица

import cv2
import numpy as np
import dlib

#Наоѓање на индекс од главните точки(landmarks)
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
#Дефинирање на главната функција за замена на лица
def face_swap(img,img2):
#openCV, за максимална точност ги користи сликите во сива боја (Grayscale)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Функцијата zeros_like() генерира слика со иста големина како оригиналната, но секој пиксел
#со вредност 0, односно црна слика со големина како оргиналната
#корисно за маски
    mask = np.zeros_like(img_gray)
    img2_new_face = np.zeros_like(img2)
    img_new_face = np.zeros_like(img)

#Детектор за лица, од dlib библиотеката
    detector = dlib.get_frontal_face_detector()

#Детекција на лица со користење на 'главни точки'(landmarks)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(img_gray)

#Наоѓање на главните точки (landmarks) на лицето, со кои се прави маска од лицето
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
#Координатите на секоја од точките(landmarks) на лицето
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))
#Конвертирање во numpy низа
        points = np.array(landmarks_points, np.int32)
#Креирање на низа од точки за обликот околу крајните точки (landmarks)
#Convex_hull претставува форма околу крајните точки, каде формата нема ниту
#еден агол поголем од 180 степени
        convex_hull = cv2.convexHull(points)
#Креирање на маска околу крајните точки (landmarks)
        cv2.fillConvexPoly(mask,convex_hull,255)
#Одделување на останатиот дел од сликата од маската
        face_image_1 = cv2.bitwise_and(img,img,mask=mask)
#Delaunay triangulation
#Креирање на правоаголник околу маската
        rect = cv2.boundingRect(convex_hull)
#Делење на точките внатре во правоаголникот, и делење на триаголници, на првата слика
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles,dtype=np.int32)

#Креирање на координати за триаголниците 
        indexes_triangles =[]
        for t in triangles:
            pt1 = (t[0],t[1])
            pt2 = (t[2],t[3])
            pt3 = (t[4],t[5])

#Наоѓање на индексите на точките
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1,index_pt2,index_pt3]
                indexes_triangles.append(triangle)
       
        
#Исцртување на линиите на триаголниците
#Коментирано бидејќи не е потребно, но може да се искористи
#да се покажат триаголниците
        #cv2.line(img,pt1,pt2,(0,0,255),2)
        #cv2.line(img,pt2,pt3,(0,0,255),2)
        #cv2.line(img,pt1,pt3,(0,0,255),2)
            
#Второ Лице ---------------------------------------------------------------
    faces2 = detector(img2_gray)
#Повторно наоѓање на главните точки(landmarks) од лицето
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x,y))
#Креирање на numpy низа од точките, и креирање на маска околу крајните
        points2 = np.array(landmarks_points2, np.int32)
        convex_hull2 = cv2.convexHull(points2)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
       

#Правење на триаголници на двете лица
    for triangle_index in indexes_triangles:

#Триаголници на првото лице
#Темињата на триаголниците
        triangle1_pt1 = landmarks_points[triangle_index[0]]
        triangle1_pt2 = landmarks_points[triangle_index[1]]
        triangle1_pt3 = landmarks_points[triangle_index[2]]
#Правење на еден триаголник, со претходните темиња
        triangle1 = np.array([triangle1_pt1,triangle1_pt2,triangle1_pt3],np.int32)

#Исцртување на правоаголник околу триаголникот
        rect1 = cv2.boundingRect(triangle1)
        (x,y,w,h) = rect1
        cropped_triangle = img[y: y+h,x:x+w]

#Креирање на црна маска со големина колку правоаголникот
        cropped_triangle1_mask = np.zeros((h,w),np.uint8)
        points = np.array([[triangle1_pt1[0]-x,triangle1_pt1[1]-y],[triangle1_pt2[0]-x,triangle1_pt2[1]-y],[triangle1_pt3[0]-x,triangle1_pt3[1]-y]],np.int32)
        cv2.fillConvexPoly(cropped_triangle1_mask,points,255)


#Просторот помеѓу линиите
        cv2.line(lines_space_mask,triangle1_pt1, triangle1_pt2, 255)
        cv2.line(lines_space_mask, triangle1_pt2, triangle1_pt3, 255)
        cv2.line(lines_space_mask, triangle1_pt1, triangle1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

#Триаголници на второто лице
        triangle2_pt1 = landmarks_points2[triangle_index[0]]
        triangle2_pt2 = landmarks_points2[triangle_index[1]]
        triangle2_pt3 = landmarks_points2[triangle_index[2]]

        triangle2 = np.array([triangle2_pt1,triangle2_pt2,triangle2_pt3],np.int32)
#Исцртување на правоаголник околу триаголникот
        rect2 = cv2.boundingRect(triangle2)
        (x,y,w,h) = rect2
#Креирање на црна маска со големина колку правоаголникот
        cropped_triangle2_mask = np.zeros((h,w),np.uint8)
        points2 = np.array([[triangle2_pt1[0]-x,triangle2_pt1[1]-y],[triangle2_pt2[0]-x,triangle2_pt2[1]-y],[triangle2_pt3[0]-x,triangle2_pt3[1]-y]],np.int32)
        cv2.fillConvexPoly(cropped_triangle2_mask,points2,255)
        
#Warp на триаголници, траиголниците на првото лице да имаат
#иста форма како соодветните триаголници од второто лице

#Генерирање на матрица од координатите
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points,points2)
        warped_triangle = cv2.warpAffine(cropped_triangle,M,(w,h))
        warped_triangle = cv2.bitwise_and(warped_triangle,warped_triangle,mask=cropped_triangle2_mask)

#Реконструкција на првото лице
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    
#Промена на лицето, поставување на реконструкцијата врз второто лице
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convex_hull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convex_hull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    return seamlessclone

#Читање на сликите
img = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")


firstSwap = face_swap(img,img2)
secondSwap = face_swap(img2,img)

#Приказ на сликите пред и потоа
img1R = cv2.resize(img, (960, 960))
img2R = cv2.resize(img2, (960, 960))
firstSwapR = cv2.resize(firstSwap, (960, 960))
secondSwapR = cv2.resize(secondSwap, (960, 960))  
cv2.imshow("Image 1 before Swap",img1R)
cv2.imshow("Image 2 before Swap",img2R)
cv2.imshow("First Swap", firstSwapR)
cv2.imshow("Second Swap", secondSwapR)

cv2.waitKey(0)
cv2.destroyAllWindows()

