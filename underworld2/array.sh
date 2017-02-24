counter=1
for a in 100000.0 1000.0 1.0 0.001 0.00001
do
   docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py H $counter md.res=64  dp.U0*=2.0 dp.lam*=$a
   #echo $counter $a $b
   let counter=counter+1
done
