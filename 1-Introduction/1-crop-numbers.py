import cv2

numbers_image=cv2.imread("Input/numbers.jpg")

# print(numbers_image.shape[0])
# print(numbers_image.shape[1])

# num0_1=numbers_image[0:20, 0:20]

num=0
count=1

for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        number=numbers_image[i:i+20, j:j+20]
        cv2.imwrite(f"Output/{num}/{num}_{count}.jpg", number)
        count+=1
        if count>500:
            count=1
            num+=1
    


