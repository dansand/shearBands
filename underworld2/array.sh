counter=1
for a in 0.0 15.0 30.0
do
   for b in 1.0 2.0 4.0
   do
      #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py C $counter md.notch_fac=$b md.res=64 dp.fa=$a md.maxIts=2
      echo $counter $a $b 
      let counter=counter+1
   done
done
