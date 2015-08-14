cat ../fileAndFlowerColour.csv | while read line; do
    folder=`echo $line|cut -d , -f 1 | sed "s/,//g"`;
    file=`echo $line|cut -d , -f 2 | sed "s/,//g"`;
    mkdir -p "$folder";
    cp "$file" "$folder/";
done
