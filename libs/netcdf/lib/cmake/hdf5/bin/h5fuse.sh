#!/usr/bin/env bash
#
# Copyright by The HDF Group.
# All rights reserved.
#
# This file is part of HDF5.  The full HDF5 copyright notice, including
# terms governing use, modification, and redistribution, is contained in
# the COPYING file, which can be found at the root of the source code
# distribution tree, or in https://www.hdfgroup.org/licenses.
# If you do not have access to either file, you may request a copy from
# help@hdfgroup.org.
#
BLD='\033[1m'
GRN='\033[0;32m'
RED='\033[0;31m'
CYN='\033[0;36m'
NC='\033[0m' # No Color

############################################################
# Usage                                                    #
############################################################
function usage {
   echo ""
   # Display usage
   echo "Purpose: Combine subfiles into a single HDF5 file. Requires the subfiling
         configuration file either as a command-line argument or the script will
         search for the *.config file in the current directory."
   echo ""
   echo "usage: h5fuse.sh [-f filename] [-h] [-p] [-q] [-r] [-v] "
   echo "-f filename  Subfile configuration file."
   echo "-h           Print this help."
   echo "-q           Quiet all output. [no]"
   echo "-p           h5fuse.sh is being run in parallel, with more than one rank. [no]"
   echo "-r           Remove subfiles after being processed. [no]"
   echo "-v           Verbose output. [no]"
   echo ""
}

function gen_mpi {

# Program to determine MPI rank and size if being run in parallel (-p).

cat > "${c_src}" << EOL
#include <mpi.h>
#include <stdio.h>
int main() {
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("%d %d", world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
EOL

}

function fuse {

# function for fusing the files

mpi_rank=0
mpi_size=1
nstart=1
nend=$nsubfiles

if [ "$parallel" == "true" ]; then

    hex=$(hexdump -n 16 -v -e '/1 "%02X"' /dev/urandom)
    c_exec="h5fuse_"${hex}
    c_src=${c_exec}.c

    # Generate and compile an MPI program to get MPI rank and size
    if [ ! -f "${c_src}" ]; then
        gen_mpi
        CC=
        ${CC} "${c_src}" -o "${c_exec}"
    fi
    wait
    rank_size=$(./"${c_exec}")
    read -r mpi_rank mpi_size <<<"$rank_size"

    rm -f "${c_src}" "${c_exec}"

    # Divide the subfiles among the ranks
    iwork1=$(( nsubfiles / mpi_size ))
    iwork2=$(( nsubfiles % mpi_size ))
    min=$(( mpi_rank < iwork2 ? mpi_rank : iwork2 ))
    nstart=$(( mpi_rank * iwork1 + 1 + min ))
    nend=$(( nstart + iwork1 - 1 ))
    if [ $iwork2 -gt "$mpi_rank" ]; then
        nend=$(( nend + 1 ))
    fi
fi

############################################################
# COMBINE SUBFILES INTO AN HDF5 FILE                       #
############################################################
icnt=1
skip=0
seek=0
seek_cnt=0
for i in "${subfiles[@]}"; do

    subfile="${subfile_dir}/${i}"

    # bs=BYTES read and write up to BYTES bytes at a time; overrides ibs and obs
    # ibs=BYTES read up to BYTES bytes at a time
    # obs=BYTES write BYTES bytes at a time
    # seek=N skip N obs-sized blocks at start of output
    # skip=N skip N ibs-sized blocks at start of input

    status=1
    fsize=${subfiles_size[icnt-1]}
    if [ "$fsize" -eq "0" ]; then
       seek_cnt=$((seek_cnt+1))
       seek=$seek_cnt
       if [ "$rm_subf" == "true" ]; then
           if [ -f "${subfile}" ]; then
               \rm -f "$subfile"
           fi
       fi
    else
       if [ $icnt -ge "$nstart" ] && [ $icnt -le "$nend" ]; then
          records_left=$fsize
          while [ "$status" -gt 0 ]; do
              if [ $((skip*stripe_size)) -le "$fsize"  ] && [ "$records_left" -gt 0 ]; then
                  EXEC="dd count=1 bs=$stripe_size if=$subfile of=$hdf5_file skip=$skip seek=$seek conv=notrunc"
                  if [ "$verbose" == "true" ]; then
                      echo -e "$GRN $EXEC $NC"
                  fi
                  err=$( $EXEC 2>&1 1>/dev/null )
                  if [ $? -ne 0 ]; then
                     echo -e "$CYN ERR: dd Utility Failed $NC"
                     echo -e "$CYN MSG: $err $NC"
                     exit $FAILED
                  fi
                  records_left=$((records_left-stripe_size))
                  skip=$((skip+1))
                  seek=$((seek_cnt+skip*nsubfiles))
              else
                  status=0
                  skip=0
              fi
          done; wait
          if [ "$rm_subf" == "true" ]; then
              \rm -f "$subfile"
          fi
       fi
       seek_cnt=$((seek_cnt+1))
       seek=$seek_cnt
    fi
    icnt=$(( icnt +1 ))
done; wait

}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
file_config=""
verbose="false"
quiet="false"
rm_subf="false"
parallel="false"
while getopts "hpqrvf:" option; do
   case $option in
      f) # subfiling configuration file
         file_config=$OPTARG;;
      h) # display Help
         usage
         exit;;
      p) # HDF5 fused file
         parallel="true";;
      q) # quiet all output
         quiet="true";;
      r) # remove completed subfiles
         rm_subf="true";;
      v) # verbose output
         verbose="true";;
     \?) # Invalid option
         echo -e "$RED ERROR: Invalid option ${BLD}-${OPTARG}${RED} $NC"
         usage
         exit 1;;
     * ) usage
         exit 1;;
   esac
done

FAILED=1
############################################################
# Configure file checks                                    #
############################################################
#
SUBF_CONFDIR="${H5FD_SUBFILING_CONFIG_FILE_PREFIX:-$PWD}"

# Try to find the config file
if [ -z "$file_config" ]; then
    nfiles=$(find "$SUBF_CONFDIR" -maxdepth 1 -type f -iname "*.config" -printf '.' | wc -m)
    if [[ "$nfiles" != "1" ]]; then
      if [[ "$nfiles" == "0" ]]; then
         echo -e "$RED Failed to find .config file in ${SUBF_CONFDIR} $NC"
         usage
         exit $FAILED
      else
         echo -e "$RED More than one .config file found in ${SUBF_CONFDIR} $NC"
         usage
         exit $FAILED
      fi
    fi
    file_config=$(find "${SUBF_CONFDIR}" -maxdepth 1 -type f -iname '*.config')
fi

if [ ! -f "$file_config" ]; then
    echo -e "${RED} configuration file ${BLD}$file_config${NC} ${RED}does not exist. $NC"
    exit $FAILED
fi

stripe_size=$(grep "stripe_size=" "$file_config"  | cut -d "=" -f2)
if test -z "$stripe_size"; then
    echo -e "$RED failed to find stripe_size in $file_config $NC"
    exit $FAILED
fi

hdf5_file="$(grep "hdf5_file=" "$file_config"  | cut -d "=" -f2)"
if test -z "$hdf5_file"; then
    echo -e "$RED failed to find hdf5 output file in $file_config $NC"
    exit $FAILED
fi

subfile_dir="$(grep "subfile_dir=" "$file_config"  | cut -d "=" -f2)"
if test -z "$subfile_dir"; then
    echo -e "$RED failed to find subfile directory in $file_config $NC"
    exit $FAILED
fi

subfs=$(sed -e '1,/subfile_dir=/d' "$file_config")
if command -v mapfile > /dev/null; then
    # For bash 4.4+
    mapfile -t subfiles <<< "$subfs"
else
    while IFS= read -r line; do
         subfiles+=("$line")
    done <<< "$subfs"
fi
if [ ${#subfiles[@]} -eq 0 ]; then
    echo -e "$RED failed to find subfiles list in $file_config $NC"
    exit $FAILED
fi
nsubfiles=${#subfiles[@]}

# Get the number of local subfiles
subfiles_loc=()
subfiles_size=()
for i in "${subfiles[@]}"; do
    subfile="${subfile_dir}/${i}"
    if [ -f "${subfile}" ]; then
        subfiles_loc+=("$subfile")
        subfiles_size+=($(wc -c "${subfile}" | awk '{print $1}'))
    else
        subfiles_size+=(0)
    fi
done

if [ "$quiet" == "false" ]; then
    TIMEFORMAT="COMPLETION TIME = %R s"
    time fuse
else
    fuse
fi
