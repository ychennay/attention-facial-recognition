import boto3
import os
from core_config import ROOT_DIR
from enum import Enum
from typing import Dict, Any, List
import re
import json

Path = str


class Genders(Enum):
    MALE = "male"
    FEMALE = "female"


class MetadataStore:
    classification_dictionary: Dict[str, Dict[str, Any]] = {}
    easy_classification_dictionary: Dict[str, Dict[str, Any]] = {}


IMAGE_NAME_REGEX = r'^([0-9]{1,3})-([0-9]{1,2}|[a-b])\.jpg$'
EASY_IMAGE_NAME_REGEX = r'^([0-9]{1,3})[a-b]\.jpg$'
S3_BUCKET_NAME = "fei-faces-sao-paulo"
LOCAL_JSON_FILENAME = "gender.json"


def populate_classification_dictionary() -> None:
    absolute_path: Path = os.path.join(ROOT_DIR, "face_images", "classification_tags", "gender")
    for gender in Genders:
        gender_directory: Path = os.path.join(absolute_path, gender.value)
        image_names: List[str] = os.listdir(gender_directory)
        for image_name in image_names:
            match = re.search(IMAGE_NAME_REGEX, image_name)
            if match:
                subject_id: str = match.group(1)
                if subject_id not in MetadataStore.classification_dictionary.keys():
                    MetadataStore.classification_dictionary[image_name] = gender.value
            else:
                easy_match = re.search(EASY_IMAGE_NAME_REGEX, image_name)
                assert easy_match, f"{image_name} was not a match for any known filename patterns."
                subject_id: str = easy_match.group(1)
                if subject_id not in MetadataStore.easy_classification_dictionary.keys():
                    MetadataStore.easy_classification_dictionary[image_name] = gender.value


if __name__ == "__main__":
    populate_classification_dictionary()

    with open(LOCAL_JSON_FILENAME, "w") as json_file:
        json.dump(MetadataStore.classification_dictionary, json_file)

    easy_json_filename = "easy_" + LOCAL_JSON_FILENAME
    with open(easy_json_filename, "w") as easy_json_file:
        json.dump(MetadataStore.easy_classification_dictionary, easy_json_file)

    s3 = boto3.resource('s3')
    faces_bucket = s3.Bucket(S3_BUCKET_NAME)

    faces_bucket.upload_file(LOCAL_JSON_FILENAME, f"classification/{LOCAL_JSON_FILENAME}")
    print(f"Uploaded {LOCAL_JSON_FILENAME}")

    faces_bucket.upload_file(easy_json_filename, f"classification/{easy_json_filename}")
    print(f"Uploaded {easy_json_filename}")
