#!/bin/bash


#
# Check that valid argument passed to script
#

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <input_file>";
	exit 1;
fi

#
# Define the input and output file names
#

input_file="$1";
output_file="${input_file%.*}.json";
tmp_file="tmp_file.json";

# Remove unwanted lines and convert the file to valid JSON format
sed '/wahoowa/d; /{"symbols": \["KMI"\], "sessionid": "c5810199-e441-429b-bdad-78f202b1c146", "linebreak": false}/d' $input_file | \
awk 'BEGIN {print "["} NR > 1 {print line ","} {line=$0} END {print line "\n]"}' > $tmp_file

input_file=$tmp_file;

# Skip empty and comma-only lines until the first JSON object, and then prepend a '[' to start the JSON array properly
awk '/{/{f=1} f{print}' $input_file | sed '1s/^/[/' > $output_file

rm $tmp_file;

echo "Cleanup completed. The corrected JSON is stored in $output_file"