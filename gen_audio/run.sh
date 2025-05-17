TEXT_NAME=$1
LANG=$2

mkdir -p logs/$LANG

touch logs/"$LANG"/"$TEXT_NAME"_0.log

export SYNDATA_PATH='.'

CUDA_VISIBLE_DEVICES=0 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 0 & \
CUDA_VISIBLE_DEVICES=1 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 1 &> logs/"$LANG"/"$TEXT_NAME"_1.log & \
CUDA_VISIBLE_DEVICES=2 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 2 &> logs/"$LANG"/"$TEXT_NAME"_2.log & \
CUDA_VISIBLE_DEVICES=3 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 3 &> logs/"$LANG"/"$TEXT_NAME"_3.log & \
CUDA_VISIBLE_DEVICES=4 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 4 &> logs/"$LANG"/"$TEXT_NAME"_4.log & \
CUDA_VISIBLE_DEVICES=5 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 5 &> logs/"$LANG"/"$TEXT_NAME"_5.log & \
CUDA_VISIBLE_DEVICES=6 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 6 &> logs/"$LANG"/"$TEXT_NAME"_6.log & \
CUDA_VISIBLE_DEVICES=7 python infer.py --text_name $TEXT_NAME --lang $LANG --proc_id 7 &> logs/"$LANG"/"$TEXT_NAME"_7.log &

wait
echo "All processes completed, start to compose dataset"

cp dataloading_script.py $SYNDATA_PATH/syndata/$LANG/$TEXT_NAME/$TEXT_NAME.py
sed -i "7s|DATA_NAME = '<LANG>/<TEXT_NAME>'|DATA_NAME = '$LANG/$TEXT_NAME'|" $SYNDATA_PATH/syndata/$LANG/$TEXT_NAME/$TEXT_NAME.py

python test_loading.py --lang $LANG --text_name $TEXT_NAME