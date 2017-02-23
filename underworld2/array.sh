counter=1
for a in 0.0 10.0 20.0 30.0 40.0
do
   for b in 1.0 2.0 4.0
   do
      docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python ti_model.py B $counter md.res=64 dp.fa=$a dp.U0*=$b
      #echo $counter $a $b
      let counter=counter+1
   done
done
