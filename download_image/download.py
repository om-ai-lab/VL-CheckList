# Download images for HAKE Dataset
import concurrent.futures
import json
import os
from urllib.request import urlretrieve

from tqdm import tqdm


def download_image(url, filename):
    try:
        urlretrieve(url, filename)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return 1
    return 0


def image_url_download(url_file, to_folder):
    count = 0
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)

    with open(url_file, "r") as f:
        contents = json.load(f)

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(
        total=len(contents)
    ) as progress:
        futures = []
        for img in contents:
            filename = os.path.join(to_folder, img)
            if not os.path.exists(filename):
                url = contents[img]
                future = executor.submit(download_image, url, filename)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            count += future.result()

    print(f"{count} images failed to download.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python download.py url_base imgname to_folder")
    else:
        image_url_download(sys.argv[1], sys.argv[2])
