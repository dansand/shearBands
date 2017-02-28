counter=1
for a in 0.25 0.5 1.0 1.25 1.5
do
   docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 1 python isotropic.py I $counter md.res=96  dp.U0*=4.0 md.pertSig=$a
   #echo $counter $a $b
   let counter=counter+1
done
