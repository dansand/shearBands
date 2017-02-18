counter=1
for a in 0.1 0.5 1.0
do
   for b in 1.0 0.5 0.0
   do
      for c in 1.0 5.0 10.0
      do
         for d in 15.0 30.0
         do 
            #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python isotropic.py E $counter md.res=64 dp.eta1*=$a dp.asthenosphere*=$b dp.U0*=$c dp.fa*=$d
            echo $counter $a $b $c $d
            let counter=counter+1
         done
      done
   done
done
