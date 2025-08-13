'''
Uploaded end of my volunteer internship at Goddard technologies (summer 2025).
Contains a script to download images and labels from the COCO 2017 dataset from fiftyone.
Allows user to specify:
- total number of images
- train, validation, test split
- classes 
- label type
- positive to negative ratio (images with instance to null images)
Formatted with the AI research and development project I helped work on in mind.
'''

#pip install fiftyone and pycocotools
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.coco import load_coco_detection_annotations, download_coco_dataset_split
import time
import webbrowser
from enum import Enum
import json, os, shutil

#clear existing data in fiftyone (if exists)
def clear_existing_data(fiftyone_path):
    if not os.path.exists(fiftyone_path):
        print(f"Path {fiftyone_path} does not exist")
        return
    contents = os.listdir(fiftyone_path)
    if not contents:
        print("No existing datasets found")
        return
    for item in contents:
        item_path = os.path.join(fiftyone_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

#parameters for amount of images and splits, contains import and export of dataset
class Dataset():
    def __init__(self, total_number_of_images, train_split, val_split, test_split, classes, label_type, pos_neg_ratio):
        self.total_number_of_images = total_number_of_images
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.classes = [classes]
        self.label_type = [label_type]
        self.pos_neg_ratio = pos_neg_ratio
    
    #number of images in each split (images with class) to negatives (images without class)
    def split_dataset(self): 
        total = self.pos_neg_ratio[0] + self.pos_neg_ratio[1]
        train_split_images = round(self.total_number_of_images * self.train_split)
        train_split_negatives = round(train_split_images/total)*self.pos_neg_ratio[1]
        train_split_positives = train_split_images - train_split_negatives

        val_split_images = round(self.total_number_of_images * self.val_split)
        val_split_negatives = round(val_split_images/total)*self.pos_neg_ratio[1]
        val_split_positives = val_split_images - val_split_negatives

        test_split_images = round(self.total_number_of_images * self.test_split)
        test_split_negatives = round(test_split_images/total)*self.pos_neg_ratio[1]
        test_split_positives = test_split_images - test_split_negatives

        return train_split_negatives, train_split_positives, val_split_negatives, val_split_positives, test_split_negatives, test_split_positives
    
    #default 300 total images, 70% train, 20% val, 10% test, keyboard segmentation, 10:1 ratio positives to negatives
    def default_settings(self):
        self.total_number_of_images = 300
        self.train_split = 0.7
        self.val_split = 0.2
        self.test_split = 0.1
        self.classes = ["keyboard"]
        self.label_type = ["segmentations"]
        self.pos_neg_ratio = [10,1]
    
    #if user does not want default settings, user can input their own
    def user_input(self):
        #Get total images
        while True:
            try:
                self.total_number_of_images = int(input("Enter total number of images for your dataset: "))
                if self.total_number_of_images > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        #Get splits
        while True:
            try:
                print("Enter train split proportion (0-1): ")
                self.train_split = float(input())
                if self.train_split < 0 or self.train_split > 1:
                    print("Invalid input, please enter a value between 0 and 1")
                    continue
                    
                print("Enter val split proportion (0-1): ")
                self.val_split = float(input())
                if self.val_split < 0 or self.val_split > 1:
                    print("Invalid input, please enter a value between 0 and 1")
                    continue
                    
                print("Enter test split proportion (0-1): ")
                self.test_split = float(input())
                if self.test_split < 0 or self.test_split > 1:
                    print("Invalid input, please enter a value between 0 and 1")
                    continue
                
                #Check if splits add up to 1.0
                total_split = self.train_split + self.val_split + self.test_split
                if abs(total_split - 1.0) > 0.001:  # Allow small floating point errors
                    print(f"Your splits add up to {total_split:.3f}, but should equal 1.0")
                    print("Please re-enter your splits")
                    continue
                break
            except ValueError:
                print("Please enter valid numbers")
        
        #Get ratio of positives to negatives
        while True:
            try:
                ratio_input = input("Enter the desired ratio of images with class to images without class, separated by a colon: ")
                self.pos_neg_ratio = [int(x.strip()) for x in ratio_input.split(":")]
                break
            except ValueError:
                print("Please enter valid numbers")

        #Get classes, read out available_classes.txt if "help"
        while True:
            try:
                print("Enter the classes you want to include in your dataset, separated by commas or type help for a list of classes")
                user_input = input()
                if user_input == "help":
                    print(f"Available classes: ")
                    try:
                        with open("available_classes.txt", "r") as file:
                            file_content = file.read()
                            print(file_content)
                    except FileNotFoundError:
                        print("File not found")
                    continue
                self.classes = user_input.split(",")
                break
            except ValueError:
                print("Please enter valid classes")

        #Get label type (fiftyone only supports segmentations and detections as of now)
        while True:
            try:
                print("Enter the label type you want to include in your dataset, separated by commas. The available types are segmentations or detections")
                self.label_type = [input()]
                break
            except ValueError:
                print("Please enter valid label type")
        
    
    def load_dataset(self):
        train_negatives, train_positives, val_negatives, val_positives, test_negatives, test_positives = self.split_dataset()
        
        print("Loading COCO-2017 dataset from FiftyOne Zoo...")
        #split dataset into train, validation, and test
        datasets = {"train": fo.Dataset(), "validation": fo.Dataset(), "test": fo.Dataset()}
        #get positive instances from train split on fiftyone
        if self.pos_neg_ratio[0] != 0:
            positives = foz.load_zoo_dataset(
                "coco-2017",
                split="train",
                only_matching=True,
                include_id=True,
                include_license=True,
                label_types=self.label_type,
                classes=self.classes,
                max_samples=(train_positives+val_positives+test_positives),
                drop_existing=True,
                shuffle=True,
            )
        #get negative instances from validation split on fiftyone
        if self.pos_neg_ratio[1] != 0:
            negatives = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                only_matching=True,
                include_id=True,
                include_license=True,
                label_types=self.label_type,
                max_samples=(train_negatives+val_negatives+test_negatives),
                drop_existing=True,
                shuffle=True,
            )

        #There is definitely a more efficient way to do this, but for some reason this is the only way I could get it to work
        count = 0
        if self.pos_neg_ratio[0] != 0:
            for sample in positives:
                if count < train_positives:
                    datasets["train"].add_sample(sample)
                elif count < train_positives + val_positives:
                    datasets["validation"].add_sample(sample)
                else:
                    datasets["test"].add_sample(sample)
                count += 1
        count = 0
        if self.pos_neg_ratio[1] != 0:
            for sample in negatives:
                if count < train_negatives:
                    datasets["train"].add_sample(sample)
                elif count < train_negatives + val_negatives:
                    datasets["validation"].add_sample(sample)
                else:
                    datasets["test"].add_sample(sample)
                count += 1
        
        print(f"All datasets loaded successfully!")
        return datasets
    
    def export_dataset(self, datasets, folder_path):
        label_field = self.label_type[0]
        for split in datasets:
            datasets[split].export(
                export_dir = os.path.join(folder_path, split),
                dataset_type=fo.types.COCODetectionDataset,
                label_field=label_field,
                classes=self.classes,
                export_media=True,
            )

#Image license information
class License(Enum):
    CC_BY = (1, "Attribution License", "https://creativecommons.org/licenses/by/4.0/")
    CC_BY_SA = (2, "Attribution-ShareAlike License", "https://creativecommons.org/licenses/by-sa/4.0/")
    CC_BY_ND = (3, "Attribution-NoDerivs License", "https://creativecommons.org/licenses/by-nd/4.0/")
    CC_BY_NC = (4, "Attribution-NonCommercial License", "https://creativecommons.org/licenses/by-nc/4.0/")
    CC_BY_NC_SA = (5, "Attribution-NonCommercial-ShareAlike License", "https://creativecommons.org/licenses/by-nc-sa/4.0/")
    CC_BY_NC_ND = (6, "Attribution-NonCommercial-NoDerivs License", "https://creativecommons.org/licenses/by-nc-nd/4.0/")
    OTHER = (7, "No known copyright restrictions", "http://flickr.com/commons/usage/e")
    CC0 = (8, "United States Government Work", "http://www.usa.gov/copyright.shtml")
   

    def __init__(self, id, license_name, url):
        self.id = id
        self.license_name = license_name
        self.url = url
   
    @classmethod
    def from_name(cls, name):
        """Get license enum from license name string"""
        for license_enum in cls:
            if license_enum.license_name == name:
                return license_enum
        return cls.CC0  # Default fallback
    
def license_info(datasets):
    #create a file with license information
    file = open(os.path.join(os.getcwd(), "license_info.txt"), "w")

    for split in datasets:
        file.write(f"\nLicense information for {split} dataset:\n")
        file.write(f"Total images: {len(datasets[split])}\n")
        license_counter = {license: 0 for license in License}
        for sample in datasets[split]:
            license_enum = License.from_name(sample.license)
            license_counter[license_enum] += 1
        
        file.write("Total images with license:\n")
        for license_enum in License:
            file.write(f"{license_enum.license_name}: {license_counter[license_enum]}\n")

    file.close()

def main():
    print("Enter dataset name:")
    dataset_name = input()
    folder_path = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    print("Do you want to clear existing data? (y/n)")
    if input() == "y":
        clear_existing_data(os.path.expanduser("~/fiftyone"))
    
    dataset_params = Dataset(0, 0, 0, 0, "", "", [0,0])
    print("Do you want to use default settings? (y/n)")
    if input() == "y":
        dataset_params.default_settings()
    else:
        dataset_params.user_input()

    print("Loading dataset...")
    datasets = dataset_params.load_dataset()
    print("Dataset loaded successfully!")
    
    print("Do you want a text file with license information? (y/n)")
    if input() == "y":
        license_info(datasets)
    
    print("Exporting dataset...")
    dataset_params.export_dataset(datasets, folder_path)
    print("Dataset exported successfully!")

if __name__ == "__main__":
    main()
    
