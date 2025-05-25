#SBATCH --exclude=gvnodeb007

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

#SBATCH -t 3-00:00:00                   #Time Limit d-hh:mm:ss
#SBATCH --partition=V4V32_SKY32M192_L           #partition/queue CAC48M192_L
#SBATCH --account=gcl_lsa273_uksr       #project allocation accout

#SBATCH --job-name=ssl_barlow_withgaussian      #Name of the job
#SBATCH  --output=./lcc_logs/%x.out             #Output file name
#SBATCH  --error=./lcc_logs/%x.err              #Error file name

#SBATCH --mail-type ALL                 #Send email on start/end
#SBATCH --mail-user ofsk222@uky.edu     #Where to send email


#Modules needed for the job
module purge
module load ccs/singularity
echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST "

CONTAINER="$PROJECT/lsa273_uksr/containers/pytorch-repitl/new-repitl.sif"
SCRIPT="$HOME/projects/ssl-testbed/run_imagenet_barlow.sh"
srun singularity run --nv $CONTAINER $SCRIPT
wait