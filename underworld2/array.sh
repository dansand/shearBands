counter=1
for a in 0.0 0.2 0.4 0.6 0.8 1.0
do
   #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py C $counter md.res=64 dp.a=$a
   echo $counter $a $b
   let counter=counter+1
done
