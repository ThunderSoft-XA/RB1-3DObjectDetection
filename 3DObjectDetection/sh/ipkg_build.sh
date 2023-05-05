OUTPUT_DIR=`pwd`/output
SOURCE_DIR=`pwd`/source
CONTROL_DIR=$SOURCE_DIR/control
DATA_DIR=$SOURCE_DIR/data

#找到对应的文件夹
if [ -z $1 ];then
    echo "please input package name:$1"
    exit 1
else
    TARGET_IPK=$1.ipk
    echo "====================================="
    echo "package_name: $TARGET_IPK"
    echo "package_source: $SOURCE_DIR"
    echo "package_output: $OUTPUT_DIR"
    echo "====================================="
fi

rm -rf $OUTPUT_DIR/$TARGET_IPK
rm -rf $OUTPUT_DIR/*.tar.gz


# #压缩CONTROL目录
CONTROL_TAR=control.tar.gz
cd $CONTROL_DIR
tar -zcvf $CONTROL_TAR *
mv $CONTROL_TAR $OUTPUT_DIR 
# #压缩data目录
DATA_TAR=data.tar.gz
cd $DATA_DIR
tar -zcvf $DATA_TAR *
mv $DATA_TAR $OUTPUT_DIR


# # #压缩etc目录
# # ETC_TAR=etc.tar.gz
# # cd $SOURCE_ETC_DIR
# # tar -zcvf $ETC_TAR *
# # mv $ETC_TAR $IPK_OUT_DIR

echo 2.0 > $OUTPUT_DIR/debian-binary
#压缩全部tar.gz生成tar.gz,并重命名成ipk
cd $OUTPUT_DIR
ar -rv $OUTPUT_DIR/$TARGET_IPK $CONTROL_TAR $DATA_TAR debian-binary
rm -rf $OUTPUT_DIR/*.tar.gz
rm -rf $OUTPUT_DIR/debian-binary