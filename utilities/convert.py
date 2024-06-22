import json
import csv
import sys

def convert_json_to_csv(file_prefix):
    json_path = f'/home/aoneill/train_inat/{file_prefix}.json'
    csv_path = f'/home/aoneill/train_inat/{file_prefix}.csv'

    # Load JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Targeting the 'images' list
    images = data['images']

    # Create CSV file
    with open(csv_path, 'w', newline='') as file:
        csv_file = csv.writer(file)
        # Write CSV headers (keys from the first image object)
        csv_file.writerow(images[0].keys())
        # Write CSV data
        for image in images:
            csv_file.writerow(image.values())

# Example usage: python convert.py train_mini
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert.py [file_prefix]")
        sys.exit(1)

    file_prefix = sys.argv[1]
    convert_json_to_csv(file_prefix)