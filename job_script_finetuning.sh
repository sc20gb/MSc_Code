#Python single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=00:30:00

#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
python fine_tune.py