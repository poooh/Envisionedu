from PIL import Image

from resizeimage import resizeimage


with open('/home/pooja/Desktop/Pooja.Kumari.jpg', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [48, 48])
        cover.save('/home/pooja/Desktop/poonew.jpg', image.format)