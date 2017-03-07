counter=1
c=1.0
for a in 0.5 1.0 2.0
do
   for b in 0.1 1.0 10.0
   do
      for c in 48 64 96
      do
         #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 16 python isotropic.py D $counter dp.notchWidth*=$a md.res=$c dp.eta2*=$b
         echo $counter $a $b $c
         let counter=counter+1
      done
   done
done


#b=$(echo "$c/$a" | bc -l)
