#!/bin/bash

# --------------------------------------------------------
# Download images for HAKE Dataset.
# --------------------------------------------------------

DATA_DIR=/network/projects/aishwarya_lab/datasets # specify the directory path
PROJ_DIR=/home/mila/r/rabiul.awal/compositional-vl/vl_checklist/download_image # specify the project directory path
if [ ! -d "$DATA_DIR/hake" ]; then
    mkdir -p "$DATA_DIR/hake"
fi


# ---------------V-COCO Dataset--------------------
cd "$DATA_DIR/hake"
echo "Downloading V-COCO Dataset"

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
#URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images
#wget -N $URL_2014_Test_images

if [ ! -d vcoco ]; then
    mkdir vcoco
fi

unzip train2014.zip -d vcoco/
unzip val2014.zip -d vcoco/
#unzip test2014.zip -d vcoco/

rm train2014.zip
rm val2014.zip
#rm test2014.zip

echo "V-COCO Dataset Downloaded!\n"


# ---------------HICO-DET Dataset-------------------
cd PROJ_DIR
echo "Downloading HICO-DET Dataset"
FILE_ID=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk
python Download_data.py $FILE_ID $DATA_DIR/hake/hico_20160224_det.tar.gz
cd "$DATA_DIR/images"
tar -xvzf hico_20160224_det.tar.gz -C ./
rm hico_20160224_det.tar.gz

echo "HICO-DET Dataset Downloaded!\n"


# ---------------HAKE Dataset-------------------
cd PROJ_DIR
echo $PWD
echo "Downloading HAKE Dataset"
FILE_ID=1Smrsy9AsOUyvj66ytGmB5M3WknljwuXL
python Download_data.py $FILE_ID $DATA_DIR/hake/hake_images_20190730.tar.gz
cd "$DATA_DIR/hake"
tar -xvzf hake_images_20190730.tar.gz -C ./
rm hake_images_20190730.tar.gz


FILE_ID=14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-
python Download_data.py $FILE_ID $DATA_DIR/hake/hake_images_20200614.tar.gz
cd "$DATA_DIR/hake"
tar -xvzf hake_images_20200614.tar.gz -C ./
rm hake_images_20200614.tar.gz

echo "HAKE Dataset Downloaded!\n"

# ---------------hcvrd Dataset(visual genome)-------
cd PROJ_DIR
echo "Downloading HCVRD(part) Dataset"
if [ ! -d hcvrd ]; then
    mkdir hcvrd
fi
python download.py $PROJ_DIR/hcvrd_url.json $DATA_DIR/hake/hcvrd

echo "HCVRD(part) Dataset Downloaded!\n"


# ---------------openimages Dataset------------------
echo "Downloading openimages(part) Dataset"
if [ ! -d openimages ]; then
    mkdir openimages
fi
python download.py $PROJ_DIR/openimages_url.json $DATA_DIR/hake/openimages

echo "openimages(part) Dataset Downloaded!\n"


# ---------------pic Dataset-------------------------

# Please download pic dataset from http://picdataset.com/challenge/index/
