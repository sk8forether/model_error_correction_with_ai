#!/bin/bash 

############ Function #############
function to_low {
  # Define file names and submit slurm jobs to reduce dataset from the original resolution using CDO in slurm environment
  # sub: subsampled every 12 gridpoints
  # low: spectral truncation to T21

  YYYYMMDD=$1
  HH=$2

  #echo ${YYYYMMDD}${HH}
  parent_dir="./sample_nc/"  # original replayed dataset location
  ncfile1="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr06_control"  # atmospheric forecast
  ncfile2="${parent_dir}/${YYYYMMDD}${HH}/bfg_${YYYYMMDD}${HH}_fhr06_control"  # boundary condition
  ncfile3="${parent_dir}/${YYYYMMDD}${HH}/control/INPUT/fv3_increment6.nc"     # forcing increment
  ncfile4="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr00_control"  # replayed atmospheric analysis

  ncfile1_low="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr06_control_low"
  ncfile2_low="${parent_dir}/${YYYYMMDD}${HH}/bfg_${YYYYMMDD}${HH}_fhr06_control_low"
  ncfile1_sub="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr06_control_sub"
  ncfile2_sub="${parent_dir}/${YYYYMMDD}${HH}/bfg_${YYYYMMDD}${HH}_fhr06_control_sub"
  ncfile3_low="${parent_dir}/${YYYYMMDD}${HH}/control/INPUT/fv3_increment6.nc_low"
  ncfile3_sub="${parent_dir}/${YYYYMMDD}${HH}/control/INPUT/fv3_increment6.nc_sub"
  ncfile4_low="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr00_control_low"
  ncfile4_sub="${parent_dir}/${YYYYMMDD}${HH}/sfg_${YYYYMMDD}${HH}_fhr00_control_sub"

  # submit the jobs only if the files do not already exist.
  if [[ ! -f "${ncfile1_low}"  ||  ! -f "${ncfile2_low}" || ! -f  "${ncfile1_sub}" || ! -f  "${ncfile2_sub}" || ! -f  "${ncfile3_low}" || ! -f  "${ncfile3_sub}" || ! -f  "${ncfile4_low}" || ! -f  "${ncfile4_sub}" ]] ; then
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -samplegrid,12 ${ncfile1} ${ncfile1_sub}"
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -samplegrid,12 ${ncfile2} ${ncfile2_sub}"
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -samplegrid,12 ${ncfile3} ${ncfile3_sub}"
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -samplegrid,12 ${ncfile4} ${ncfile4_sub}"
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -sp2gp -sp2sp,21 -gp2sp  ${ncfile1} ${ncfile1_low}"  
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -sp2gp -sp2sp,21 -gp2sp -delname,pres_avehcb,pres_avehct,pres_avelcb,pres_avelct,pres_avemcb,pres_avemct,prescnvclb,prescnvclt,soilm,tmp_avehct,tmp_avelct,tmp_avemct ${ncfile2} ${ncfile2_low}" # get rid of variables with NaN
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -sp2gp -sp2sp,21 -gp2sp  ${ncfile3} ${ncfile3_low}"
    sbatch -A gsienkf -n 1 -t 20 --mem=14g --wrap "cdo -L -sp2gp -sp2sp,21 -gp2sp  ${ncfile4} ${ncfile4_low}"
  fi
}

########### Main #################

date

d=2019-12-20 # start date
while [[ "$(date -d "$d" +%Y%m%d )" < "20191221" ]];do # loop to the end date
  YYYYMMDD=$(date -d "$d" +%Y%m%d)

  for HH in  00 06 12 18 ; do # loop through hours
      echo ${YYYYMMDD}${HH} convert to low res
      to_low ${YYYYMMDD} ${HH} &
  done
  
  d=$(date -I -d "$d + 1 day") # advance 1day
done

date