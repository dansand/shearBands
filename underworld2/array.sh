counter=1
for a in 0.25 0.5 1.0 2.0
do
   for b in 10.0 20.0 30.0 40.0
   do
      #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 16 python isotropic.py J $counter md.res=96 dp.notchWidth*=$a dp.fa=$b
      #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 16 python isotropic.py K $counter md.res=96 dp.U0*=0.252 dp.asthenosphere=0.0 dp.depth=10000.0 dp.eta0*=0.1 dp.eta1*=10.0 dp.notchWidth*=$a dp.fa=$b
      echo $counter $a $b
      let counter=counter+1
   done
done
