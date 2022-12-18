#!/bin/bash
#PBS -N test1
#PBS -q gpu_1
#PBS -l ncpus=10:ngpus=1
#PBS -P CSCI0886
#PBS -l walltime=6:00:00
#PBS -o /mnt/lustre/users/dvoneschwege/g2-peg-in-hole/cuda/test1.out
#PBS -e /mnt/lustre/users/dvoneschwege/g2-peg-in-hole/cuda/test1.err
#PBS -m abe
#PBS -M 21785155@sun.ac.za
 
cd /mnt/lustre/users/dvoneschwege/g2-peg-in-hole/cuda
 
echo
echo `date`: executing CUDA job on host ${HOSTNAME}
echo
echo Available GPU devices: $CUDA_VISIBLE_DEVICES
echo
 
# Run program
./hello_cuda