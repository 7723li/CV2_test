from PIL import Image
import PIL,os,time,random
os.chdir(r'C:\Users\Administrator\Desktop')

def Pic(pic1name,pic2name):
    print(__name__)
    pic1=Image.open(pic1name)#转换图片（原始）
    pic2=Image.open(pic2name)#待转换图片（结果）
    print(pic1.size,pic2.size)
    if pic1.size<pic2.size:
        pic2=pic2.resize(min(pic1.size,pic2.size))
    else:
        pic1=pic1.resize(min(pic1.size,pic2.size))
    print(pic1.size,pic2.size)
    newImg=Image.new('RGBA',(pic1.width,pic1.height))
    pic1pixel,pic2pixel,newImg_Pixel=[],[],[]

    start=time.time()
    for x in range(pic1.width):
            for y in range(pic1.height):
                    pic1pixel.append(pic1.getpixel((x,y)))

    for x in range(pic2.width):
            for y in range(pic2.height):
                    pic2pixel.append(pic2.getpixel((x,y)))

    for i in range(len(pic2pixel)):
            newImg_Pixel.append((int((pic1pixel[i][0]+pic2pixel[i][0])/2),\
                                 int((pic1pixel[i][1]+pic2pixel[i][1])/2),\
                                 int((pic1pixel[i][2]+pic2pixel[i][2])/2)))

    for x in range(newImg.width):
            for y in range(newImg.height):
                    newImg.putpixel((x,y),newImg_Pixel[newImg.height*x+y])
    
    print(time.time()-start)
    newImg.save('test.jpg')
    return newImg

def RandomPic():
    width=input('>>>')
    height=width
    newImg=Image.new('RGBA',(width,height))
    RandomPixel=[]
    for x in range(newImg.width):
        for y in range(newImg.height):
            RandomPixel.append((random.randint(2,255+1),\
                                random.randint(2,255+1),\
                                random.randint(2,255+1)))

    for x in range(newImg.width):
            for y in range(newImg.height):
                newImg.putpixel((x,y),RandomPixel[newImg.height*x+y])

    newImg.show()
    return newImg

if __name__ == '__main__':
    pic1,pic2='mosaic.jpg','908ebf5484d16bfbe27f05c7437f1022_1.jpg'
    newImg=Pic(pic1,pic2)
    newImg.show()

    '''
    for i in range(4):
        print('*'*50)
        print(i+1)
        newImg=Pic('test.jpg',pic2)
    newImg.show()

    '''
