from paths_of_important_files import paths_of_data_processing_files
#from paths_of_important_files import paths_of_other_source_files
#from paths_of_important_files import paths_to_data_files
#from paths_of_important_files import paths_to_docs
#from paths_of_important_files import paths_to_output_files
#from paths_of_important_files import paths_to_old_output_files
import os
import re


base_path = "/sfs/gpfs/tardis/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors"


def aggregate_files(base_path, list_of_relative_paths):
    aggregated_content = ""
    for relative_path in list_of_relative_paths:
        full_path = os.path.join(base_path, relative_path)
        aggregated_content += f"=== {full_path} ===\n"
        try:
            with open(full_path, 'r', encoding = "utf-8", errors = "replace") as file:
                extension = os.path.splitext(full_path)[1].lower()
                if extension == ".csv":
                    lines = []
                    for _ in range(0, 4):
                        line = file.readline()
                        if not line:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                elif extension == ".owl":
                    lines = []
                    for _ in range(0, 59):
                        line = file.readline()
                        if not line:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                else:
                    content = file.read()
                aggregated_content += content + "\n\n"
        except FileNotFoundError:
            raise Exception(f"A file at the following path was not found. {full_path}")
        except Exception as e:
            raise Exception(f"The following error occurred when reading {full_path}. {e}")
    return aggregated_content


def replace_import_groups(content):
    pattern = r'(?m)^(import\s+.*\n)+'
    modified_content = re.sub(pattern, "...\n", content)
    return modified_content


def main():
    aggregated_string = aggregate_files(base_path, paths_of_data_processing_files)
    #aggregated_string = replace_import_groups(aggregated_string)
    output_path = os.path.join(base_path, "aggregated_contents.txt")
    try:
        with open(output_path, 'w', encoding = "utf-8") as output_file:
            output_file.write(aggregated_string)
        print(f"Aggregated content was saved to {output_path}.")
    except Exception as e:
        raise Exception(f"Writing aggregated content to file failed with the following error. {e}")


if __name__ == "__main__":
    main()