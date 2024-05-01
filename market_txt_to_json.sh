#!/bin/bash

#
# Fetch command line argument (e.g. the name of the text file with market stream data to convert into json)
#

input_file="$1"

output_file="${input_file%.txt}.json"


#
# Count input file lines
#

total_lines=$(wc -l < "$input_file")


#
# Begin JSON File Formatting
#


echo "[" > "$output_file"; # json files begin with closed interval square bracket


#
# Read the input file line-by-line
#

current_line=0
while IFS= read -r line; do
    ((current_line++))

    #
    # Check for final line of file
    #

    if [ "$current_line" -eq "$total_lines" ]; then
        # Last line: append without a comma
        echo "$line" >> "$output_file"
    else
        # Not the last line: append with a comma
        echo "$line," >> "$output_file"
    fi
done < "$input_file"


echo "]" >> "$output_file"; # json files end with closed interval square bracket

echo "Conversion completed. The JSON file is saved as $output_file"

