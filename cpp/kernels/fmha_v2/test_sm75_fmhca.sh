# COMMAND="$CUDA_PATH/compute-sanitizer/compute-sanitizer"
COMMAND=

$COMMAND ./bin/fmhca.exe -b 1 -s-q 4096 -min-s 4096 -d  40
$COMMAND ./bin/fmhca.exe -b 1 -s-q 4096 -min-s 4096 -d  80

$COMMAND ./bin/fmhca.exe -b 4 -s-q 4096 -min-s 4096 -d  40
$COMMAND ./bin/fmhca.exe -b 4 -s-q 4096 -min-s 4096 -d  80

$COMMAND ./bin/fmhca.exe -b 1 -s-q 2304 -min-s 2304 -d  40
$COMMAND ./bin/fmhca.exe -b 1 -s-q 2304 -min-s 2304 -d  80

$COMMAND ./bin/fmhca.exe -b 4 -s-q 2304 -min-s 2304 -d  40
$COMMAND ./bin/fmhca.exe -b 4 -s-q 2304 -min-s 2304 -d  80

$COMMAND ./bin/fmhca.exe -b 1 -s-q 1024 -min-s 1024 -d  40
$COMMAND ./bin/fmhca.exe -b 1 -s-q 1024 -min-s 1024 -d  80

$COMMAND ./bin/fmhca.exe -b 4 -s-q 1024 -min-s 1024 -d  40
$COMMAND ./bin/fmhca.exe -b 4 -s-q 1024 -min-s 1024 -d  80
