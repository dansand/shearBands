#
counter=1
c=1.0
for a in 1.0
do
   for b in 0.0 10.0 20.0 30.0 40.0
   do
      for c in 96
      do
         #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 16 python isotropic.py G $counter dp.notchWidth*=$a md.res=$c dp.fa*=$b
         echo $counter $a $b $c
         let counter=counter+1
      done
   done
done
