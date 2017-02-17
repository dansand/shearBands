counter=1
for a in 0.0 0.2 0.4 0.6 0.8 1.0
do
   #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py A 2 $counter md.res=64 dp.a=$a dp.fa=30.0
   echo $counter $a
   let counter=counter+1
done
