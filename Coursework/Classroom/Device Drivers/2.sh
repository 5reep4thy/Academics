function func() {
str="Testing.."
echo $str
}

echo "Total arguments : $#"
echo "1st Argument = $1"
echo "2nd argument = $2"

touch tester.txt
chmod 000 tester.txt
chmod 777 tester.txt

for ((i = 1; i < 5; i++))
do
	touch "tester$(printf "$i").txt"
done

echo $(func)

echo These are all the files in this directory
ls -al

string1="mac"
string2="linux"
string3=$string1+$string2
echo $string3

echo Sleep for 3 seconds
sleep 5

echo Tell date, I will tell day

echo enter date
read dd mm yyyy
echo $dd $mm $yyyy
final=3


for ((i = 1800; i < $yyyy; i++))
do
	if [ $(expr $i % 100) == 0 ]
	then
		if [ $(expr $i % 400) == 0 ]
		then
			final=$(expr $final + 2)
		else
			final=$(expr $final + 1)
		fi
	else
		if [ $(expr $i % 4) == 0 ]
		then
			final=$(expr $final + 2)
		else 
			final=$(expr $final + 1)
		fi
	fi

done


if [ $mm -ge 2 ]
then 
	final=$(expr $final + 3) 
fi
if [ $mm -ge 3 ]
then 
	if [ $(expr $yyyy % 4) == 0 ]
	then
		final=$(expr $final + 1)
	else 
		final=$(expr $final + 0)
	fi
fi 
if [ $mm -ge 4 ]
then
	final=$(expr $final + 3)
fi
if [ $mm -ge 5 ]
then
	final=$(expr $final + 2)
fi
if [ $mm -ge 6 ]
then
	final=$(expr $final + 3)
fi
if [ $mm -ge 7 ]
then
	final=$(expr $final + 2)
fi
if [ $mm -ge 8 ]
then
	final=$(expr $final + 3)
fi
if [ $mm -ge 9 ]
then
	final=$(expr $final + 3)
fi
if [ $mm -ge 10 ]
then
	final=$(expr $final + 2)
fi
if [ $mm -ge 11 ]
then
	final=$(expr $final + 3)
fi
if [ $mm -ge 12 ]
then
	final=$(expr $final + 2)
fi
final=$(expr $final + $dd - 1)
final=$(expr $final % 7)
echo Final is 
if [ $final == 0 ]
then
	echo Happy Sunday
elif [ $final == 1 ]
then
	echo Happy Monday
elif [ $final == 2 ]
then
	echo Happy Tuesday
elif [ $final == 3 ]
then
	echo Happy Wednesday
elif [ $final == 4 ]
then
	echo Happy Thursday
elif [ $final == 5 ]
then
	echo Happy Friday
elif [ $final == 6 ]
then
	echo Happy Saturday
fi