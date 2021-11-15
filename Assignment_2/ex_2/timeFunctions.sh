## declare an array variable
declare -a arr=("100" "1000" "10000" "100000" "1000000" "10000000" "100000000" "1000000000")

## now loop through the above array
for i in "${arr[@]}"
do
   ./exercise_2_timer "$i"
done