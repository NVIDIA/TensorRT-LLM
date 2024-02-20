#!/usr/bin/env python
import getopt
import time
import errno
import os
import sys
import subprocess
import argparse

g_verbose = False
origRssFile = 0
min_total_iovmm = float('inf')
max_total_iovmn = float('-inf')
min_total_system_avail = float('inf')
max_total_system_avail = float('-inf')
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NORMAL  = '\33[37m'

def human(num_str, power="Ki", units=None):
    num = 0.0
    try:
        num = float(num_str)
    except:
        return "0"
    if units is None:
        powers = ["Ki", "Mi", "Gi", "Ti"]
        if num >= 1000: #4 digits
            num /= 1024.0
            power = powers[powers.index(power)+1]
        return "%.4f %sB" % (num, power)
    else:
        return "%.f" % ((num * 1024) / units)





def runCmd(cmd):
    out = "-1"
    try:
        out = subprocess.check_output(cmd, shell=True)
    except:
        pass
    return out

def NvMap():
   
    total_iovmm = 0
    cmd = "cat /sys/kernel/debug/nvmap/iovmm/clients | grep total"
    out = runCmd(cmd)
    if out != "-1":
        total_iovmm = out.split()[1]
    
    total_iovmm_value = float(total_iovmm.decode('utf-8').rstrip('K'))
    global min_total_iovmm, max_total_iovmn, min_total_system_avail,max_total_system_avail
    min_total_iovmm = min(min_total_iovmm,total_iovmm_value)
    max_total_iovmn = max(max_total_iovmn,total_iovmm_value)
    sys.stdout.write(time.strftime("%H:%M:%S", time.localtime()) + "\t")
    sys.stdout.write("Total used hardware memory: " + total_iovmm.decode('utf-8') + "\t")
    total_mem_cpu_gpu = 0
    cmd = "cat /proc/meminfo  | grep MemAvailable"
    out = runCmd(cmd)
    if out != "-1":
        total_mem_cpu_gpu = out.split()[1]
    total_mem_cpu_gpu_value = float(total_mem_cpu_gpu.decode('utf-8').rstrip('K'))
    min_total_system_avail = min(min_total_system_avail,total_mem_cpu_gpu_value)
    max_total_system_avail = max(max_total_system_avail,total_mem_cpu_gpu_value)
    
    sys.stdout.write("Total system avalible memory: " + total_mem_cpu_gpu.decode('utf-8') + "\n")
    
    




import signal        
def signal_handler(sig,frame):
    sys.stdout.write("Min total iovmm: " + str(min_total_iovmm)+ "\n")
    sys.stdout.write("Max total iovmm: " + str(max_total_iovmn) + "\n")
    sys.stdout.write("=========Total used hardware memory: " + str((max_total_iovmn-min_total_iovmm)/1024/1024) + " GB\n")
    sys.stdout.write("Min total system avail: " + str(min_total_system_avail)+ "\n")
    sys.stdout.write("Max total system avail: " + str(max_total_system_avail) + "\n")
    sys.stdout.write("=========Total used system memory: " + str((max_total_system_avail-min_total_system_avail)/1024/1024) + " GB\n")
    sys.exit(0)

def main():
    arg_parser = argparse.ArgumentParser(description="Memory usage of pragram")
    # Essential parameter
    options = arg_parser.parse_args()
    print("================Before start check programer=====================")
    cmd = "cat /sys/kernel/debug/nvmap/iovmm/clients"
    out = runCmd(cmd)
    decoded_out = out.decode("utf-8")
    lines = decoded_out.splitlines()  # 将字符串按行分割成列表
    for line in lines:
        print(line)
    print("=================Check end========================================")
    signal.signal(signal.SIGINT, signal_handler)
    total_iovmm_list =[]
    total_mem_cpu_gpu_list = []
    while True:
        NvMap()
        time.sleep(0.5)

if __name__ == '__main__': 
    main()
