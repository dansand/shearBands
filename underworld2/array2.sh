c=1.0
for a in 0.0 10.0 20.0 30.0 40.0
do
   #docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python ti_model.py T $counter dp.fa*=$a
   echo $counter $a
   let counter=counter+1
done
