from typing import Generator, Sequence
from PIL import Image, ImageDraw, ImageFont
import os

def chunks(l: Sequence, n: int = 5) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def add_caption(image, text1,score1,text2,score2,sub_folder,label):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    img=Image.open(image)
    w,h = img.size
    w = int(w)
    h = int(h)
    if text2 != None:
        bi = Image.new("RGB", (w+10, h+100), "white")
        bi.paste(img, (5,5,(w+5), (h+5)))
        font =ImageFont.truetype(os.path.join(root_dir,"simsunb.ttf"), 20)
        w1,h1 = font.getsize(text1)
        draw = ImageDraw.Draw(bi)
        draw.text((5, h+100-h1*4-5), text1, font=font, fill="black")
        draw.text((5, h+100-h1*3-5), score1, font=font, fill="black")
        draw.text((5, h+100-h1*2-5), text2, font=font, fill="black")
        draw.text((5, h+100-h1-5), score2, font=font, fill="black")
    else:
        bi = Image.new("RGB", (w+10, h+50), "white")
        bi.paste(img, (5,5,(w+5), (h+5)))
        font =ImageFont.truetype(os.path.join(root_dir,"simsunb.ttf"), 20)
        w1,h1 = font.getsize(text1)
        draw = ImageDraw.Draw(bi)
        draw.text((5, h+50-h1*2-5), text1, font=font, fill="black")
        draw.text((5, h+50-h1-5), score1, font=font, fill="black")
    bi.save(sub_folder+"/"+label+'.jpg')