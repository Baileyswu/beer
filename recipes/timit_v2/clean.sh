dir=$(pwd)/data
name=$dir

if [ -d $name ] || [ -f $name ]; then
	echo cleaning $name
	rm -rf $name
	echo "done"
else
	echo $name not exist
fi
