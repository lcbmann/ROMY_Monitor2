#!/bin/bash
#
# update all plots for the HTML monitor
#
# AndBro @2021
## ________________________________________________________________

starttime=start=`date +%s.%N`

checkmark="\033[0;32m\xE2\x9C\x94\033[0m"

workdir="/home/brotzer/Desktop/html_plots/"

outpath="${workdir}figures/"

## ________________________________________________________________

function run_script() {

    echo -n "run: $1 ...";

    python3 ${workdir}${1} &>/dev/null 



    if [ $(find $outpath -type f -name $2 -mmin -1 2>/dev/null) ];then 
        status=True
        echo -e "\\r${checkmark} $1  -> Done \n"
    else 
        status=False
        echo -e "\\r$1  ->  Failed"
    fi   

}


## ________________________________________________________________
## MAIN

#while true; do
#    starttime=start=`date +%s.%N`

if [ ! -f "${workdir}config.yaml" ]; then 
    echo -e "ERROR: ${workdir}config.yaml doesn't exists!\n"
    exit 
fi

#run_script 'make_WROMY_plots.py' 'html_wromy_plots.png' 

#run_script 'make_RADON_plot.py' 'html_radon_plots.png' 

#run_script 'make_TILT_plot.py' 'html_tilt_plots.png'

#run_script 'make_ROMY_plot.py' 'html_romy_plots.png'

#run_script 'make_ARRAY_plot.py' 'html_array_plots.png'

run_script $1 $2

endtime=`date +%s.%N`

runtime=$( echo "$endtime - $starttime" | bc -l )
echo -e " Runtime: ${runtime:0:5} seconds\n"

#done 
## ________________________________________________________________
## END OF FILE
