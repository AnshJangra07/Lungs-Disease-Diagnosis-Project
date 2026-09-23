import subprocess
import sys

from xray.exception import XRayException


class S3Operation:
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            subprocess.run(
                ["aws", "s3", "sync", folder, f"s3://{bucket_name}/{bucket_folder_name}/"],
                check=True,
            )

        except Exception as e:
            raise XRayException(e, sys)

    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            subprocess.run(
                ["aws", "s3", "sync", f"s3://{bucket_name}/{bucket_folder_name}/", folder],
                check=True,
            )

        except Exception as e:
            raise XRayException(e, sys)