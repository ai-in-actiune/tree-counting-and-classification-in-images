import os
import io
import hashlib
import time

import requests
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def _scroll_to_end(wd: webdriver, sleep_time: int):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(sleep_time)


def _fetch_image_urls(
    query: str,
    max_links_to_fetch: int,
    wd: webdriver,
    sleep_between_interactions: int = 1,
):
    """
    Args:
        query: Search term, like Dog
        max_links_to_fetch: Number of links to collect
        wd: instantiated Webdriver
        sleep_between_interactions: wait before interacting with the page
    Returns:
        returns a set of image links based on max_links_to_fetch param
    """

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0

    while image_count < max_links_to_fetch:
        _scroll_to_end(wd, sleep_between_interactions)

        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
        )

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail to get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            actual_images = wd.find_elements_by_css_selector("img.n3VNCb")
            for actual_image in actual_images:
                src_attr = actual_image.get_attribute("src")
                if src_attr and "http" in src_attr:
                    image_urls.add(src_attr)

            image_count = len(image_urls)

            if image_count >= max_links_to_fetch:
                print(f"Found: {image_count} image links, done!")
                break

        else:
            print(f"Found: {image_count} image links, looking for more ...")
            time.sleep(30)

            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result start-point further down
        results_start = number_results

    return image_urls


def _persist_image(folder_path: str, url: str):
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(
            folder_path, f"{hashlib.sha1(image_content).hexdigest()[:10]}.jpg"
        )

        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)

        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(
    search_term: str,
    driver: webdriver,
    target_path: str = "./images",
    number_images: int = 5,
    sleep_time: int = 0.5
):
    target_folder = os.path.join(target_path, "_".join(search_term.lower().split(" ")))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with driver as wd:
        image_urls_res = _fetch_image_urls(
            search_term, number_images, wd=wd, sleep_between_interactions=sleep_time
        )

        for img_url in image_urls_res:
            _persist_image(target_folder, img_url)


if __name__ == "__main__":
    # Example using chrome web driver.
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    search_term = "Trees from drone"
    search_and_download(search_term=search_term, driver=driver)
