from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException, MoveTargetOutOfBoundsException
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import time
import random
import json
import os

# Base URL
BASE_URL = "https://www.automobile.tn/fr/occasion"

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

def simulate_human_interaction(driver):
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.5, 1.5))
        ActionChains(driver).move_by_offset(random.randint(10, 100), random.randint(10, 100)).perform()
    except MoveTargetOutOfBoundsException:
        pass
    except Exception as e:
        print(f"Interaction error: {e}")

def get_soup(driver, url, retries=3, wait_time=20):
    for attempt in range(retries):
        try:
            driver.get(url)
            simulate_human_interaction(driver)
            WebDriverWait(driver, wait_time).until(
                lambda d: (
                    d.find_elements(By.CLASS_NAME, "articles") or
                    d.find_elements(By.CLASS_NAME, "main-specs") or
                    d.find_elements(By.CLASS_NAME, "price-box")
                )
            )
            return BeautifulSoup(driver.page_source, "html.parser")
        except (TimeoutException, WebDriverException) as e:
            print(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            time.sleep(random.uniform(2, 4))
    print(f"❌ Failed to fetch {url}")
    return None

def extract_post_details(driver, post_url):
    time.sleep(random.uniform(1.5, 2.5))
    soup = get_soup(driver, post_url)
    if not soup:
        return None

    car_data = {"url": post_url}
    car_data["ID"] = post_url.rstrip("/").split("/")[-1]

    price_div = soup.find("div", class_="price")
    if price_div:
        car_data["price"] = price_div.get_text(strip=True).replace("DT", "").strip()

    main_specs = soup.find("div", class_="main-specs")
    if main_specs:
        for li in main_specs.find_all("li"):
            name = li.find("span", class_="spec-name")
            value = li.find("span", class_="spec-value")
            if name and value:
                key = name.get_text(strip=True)
                val = value.get_text(strip=True)
                val = " ".join(val.split()[:-1]) if val.split()[-1] in ["km", "cv", "cm³"] else val
                car_data[key] = val

    boxes = soup.find_all("div", class_="box")
    for box in boxes:
        title = box.find("div", class_="box-inner-title")
        if title:
            section = title.get_text(strip=True)
            if section in ["Spécifications", "Motorisation"]:
                for li in box.find_all("li"):
                    name = li.find("span", class_="spec-name")
                    value = li.find("span", class_="spec-value")
                    if name and value:
                        key = name.get_text(strip=True)
                        val = value.get_text(strip=True)
                        if key in ["Puissance", "Puissance fiscale", "Cylindrée"]:
                            val = " ".join(val.split()[:-1]) if val.split()[-1] in ["ch dyn", "cv", "cm³"] else val
                        elif key == "Génération":
                            small = li.find("small", class_="text-muted")
                            val = small.get_text(strip=True) if small else val
                        car_data[key] = val
    return car_data

def retry_extract_post_details(driver, post_url, max_retries=2):
    for attempt in range(max_retries):
        data = extract_post_details(driver, post_url)
        if data:
            return data
        print(f"Retrying ({attempt + 1}) for: {post_url}")
        time.sleep(random.uniform(1, 3))
    return None

def scrape_page(driver, page_url):
    soup = get_soup(driver, page_url)
    if not soup:
        return [], False

    articles = soup.find("div", class_="articles")
    if not articles:
        return [], False

    posts = articles.find_all("div", attrs={"data-key": True})
    if not posts:
        return [], False

    listings = []
    for idx, post in enumerate(posts, start=1):
        link = post.find("a", class_="occasion-link-overlay")
        if link and "href" in link.attrs:
            full_url = "https://www.automobile.tn" + link["href"]
            print(f"🔍 Scraping post {idx} on page: {full_url}")
            data = retry_extract_post_details(driver, full_url)
            if data:
                for key, value in data.items():
                    print(f"    {key}: {value}")
                listings.append(data)
            else:
                print(f"⚠️ Skipped: {full_url}")
            time.sleep(random.uniform(1.5, 3))

    return listings, True if listings else False

def main():
    driver = setup_driver()
    all_listings = []
    seen_ids = set()
    last_first_post_id = None
    page = 1
    try:
        while True:
            url = BASE_URL if page == 1 else f"{BASE_URL}/{page}"
            print(f"\n📄 Scraping page {page}: {url}")
            page_data, has_posts = scrape_page(driver, url)

            if not has_posts or not page_data:
                print("❌ No posts found on this page, stopping.")
                break

            first_post_id = page_data[0].get("ID")
            if first_post_id == last_first_post_id:
                print("🛑 Duplicate page detected (first post ID repeated). Stopping.")
                break
            last_first_post_id = first_post_id

            new_data = [post for post in page_data if post.get("ID") not in seen_ids]
            for post in new_data:
                seen_ids.add(post.get("ID"))

            all_listings.extend(new_data)
            print(f"✅ Added {len(new_data)} new posts from page {page}")

            with open("car_listings.json", "w", encoding="utf-8") as f:
                json.dump(all_listings, f, ensure_ascii=False, indent=4)
                print(f"💾 Progress saved after page {page} ({len(all_listings)} listings)")

            page += 1
            time.sleep(random.uniform(2, 4))
    finally:
        driver.quit()

    print(f"\n✅ Done. {len(all_listings)} unique listings saved to car_listings.json")

if __name__ == "__main__":
    main()