FILE=$1

if [[ $FILE != "monet2photo" ]]; then
    echo "Available models are: monet2photo"
    exit 1
fi

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
ZIP_FILE=./checkpoints/${FILE}_pretrained.zip
FILE_ID=1rToFkTXczXiTLJw3CrWodQJ487Sded7H
URL=https://drive.google.com/uc?export=download&id=1rToFkTXczXiTLJw3CrWodQJ487Sded7H

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O $ZIP_FILE && rm -rf /tmp/cookies.txt
unzip $ZIP_FILE -d ./checkpoints/${FILE}_pretrained
rm $ZIP_FILE