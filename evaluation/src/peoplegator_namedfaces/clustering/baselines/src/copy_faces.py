
import shutil
from pathlib import Path

def main(input_directory: Path, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    for library in input_directory.iterdir():
        if library.is_dir():
            for document in library.glob('*.peoplegator_aligned_crops'):
                if document.is_dir():
                    for image in document.glob('*.*'):
                        if image.is_file():
                            relative_path = image.relative_to(input_directory)
                            destination = output_directory / relative_path
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(image, destination)
                            
                            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Copy face images from the input directory to the output directory while preserving the directory structure.')
    parser.add_argument('-i', '--input_directory', type=Path, required=True, help='The root directory containing the face images organized in subdirectories.')
    parser.add_argument('-o', '--output_directory', type=Path, required=True, help='The directory where the copied face images will be stored, preserving the original structure.')
    args = parser.parse_args()
    
    main(args.input_directory, args.output_directory)