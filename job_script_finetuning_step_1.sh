#Python single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=03:00:00

#Request some memory per core
#$ -l h_vmem=100G

#$ -l coproc_v100=1


#Get email at start and end of the job
#$ -m be

#Now run the job
python fine_tuning.py