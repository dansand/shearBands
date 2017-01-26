counter=1
for a in 0.0 10.0 20.0 30.0
do
   docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py A $counter md.RES=64 dp.fa=$a md.maxIts=2
   echo $counter $a 
   let counter=counter+1
done
