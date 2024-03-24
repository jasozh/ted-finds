import csv
import json

csv_file_path = 'static/data/talks_info_final.csv'

# Specify the path to the JSON file you want to overwrite
json_file_path = 'init.json'

# Read data from the CSV file and convert it to JSON format
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    # Convert CSV data into a list of dictionaries
    data = [row for row in csv_reader]

# Overwrite 'init.json' with the new JSON data
with open(json_file_path, 'w') as json_file:
    # Write JSON data to the file, replacing its contents
    json.dump(data, json_file, indent=4)