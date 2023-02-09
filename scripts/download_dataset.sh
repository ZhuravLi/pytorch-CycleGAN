FILE=$1

if [[ $FILE != "monet2photo" ]]; then
    echo "Available datasets are: monet2photo"
    exit 1
fi

echo "Specified [$FILE]"

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE
mkdir -p ./datasets
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE