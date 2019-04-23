from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.touch_actions import TouchActions
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import re, time
import pandas as pd
import numpy as np
import os
from selenium.webdriver.chrome.options import Options

car_types = ['sedans','sport-utilities','crossovers','coupes','pickup-trucks']
url = 'https://www.carmax.com/cars/'

driver = webdriver.Chrome()
driver.implicitly_wait(10)
wait = WebDriverWait(driver, 15)
cond_wait = WebDriverWait(driver, 5)
driver.set_page_load_timeout(20)

touch_actions = TouchActions(driver)

#Change these two for what to scrape and name of final file
filename = 'carmax_sedans'
car_type = 'sedans'


driver.get(url + car_type)
car_tile_xpath = "//a[@class='kmx-typography--font-alt kmx-typography--weight-400 kmx-elevation-02']"
see_more_button = "//button[@class='see-more see-more__cta-all']"


dropdown_xpath = "//div[@class='mdc-select__surface mdc-ripple-upgraded']"
wait.until(EC.presence_of_all_elements_located((By.XPATH, dropdown_xpath)))
dropdown_menu = driver.find_element_by_xpath(dropdown_xpath)
dropdown_menu.location_once_scrolled_into_view
touch_actions.scroll(0,-100)
touch_actions.perform()
dropdown_menu.click()
wait.until(EC.presence_of_all_elements_located((By.XPATH, "//li[text()='Nationwide']")))
driver.find_element_by_xpath("//li[text()='Nationwide']").click()


# Finding all relevent makes
wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='Make']")))
driver.find_element_by_xpath("//div[@id='Make']").click()
wait.until(EC.presence_of_all_elements_located((By.XPATH, "//span[@class='refinements--value--name']")))
time.sleep(1)
car_makes = driver.find_elements_by_xpath("//span[@class='refinements--value--name']")
makes = []
for make in car_makes:
    makes.append(make.text)
print('Makes found: ')
print(makes)

df_cols = ['model', 'price', 'mileage', 'src']
df = pd.DataFrame([['x', 'x', 'x', 'x']], columns=df_cols)

for make in makes:
    make_url = url + car_type + '/' + make.lower()
    print('Navigating to ' + make_url)
    driver.get(make_url)
    # Finding all relevent exterior colors
    wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='ExteriorColor']")))
    driver.find_element_by_xpath("//div[@id='ExteriorColor']").click()
    wait.until(EC.presence_of_all_elements_located((By.XPATH, "//div[@class='facet--color-value kmx-typography--font-alt kmx-typography--weight-300']/span")))
    time.sleep(1)
    make_colors = driver.find_elements_by_xpath("//div[@class='facet--color-value kmx-typography--font-alt kmx-typography--weight-300']/span")
    colors = []
    for color in make_colors:
        colors.append(color.text)
    print('Exterior colors found for ' + make + ': ')
    if colors[0] == '':
        colors = ['black', 'blue', 'brown', 'gold', 'gray', 'green', 'orange', 'purple',
        'red', 'silver', 'tan', 'white', 'yellow']
    print(colors)

    for color in colors:
        make_color_url = make_url + '/' + color.lower()
        driver.get(make_color_url)
        print('Current car: ' + color + ' ' + make)
        dropdown_xpath = "//div[@class='mdc-select__surface mdc-ripple-upgraded']"
        wait.until(EC.presence_of_all_elements_located((By.XPATH, dropdown_xpath)))
        dropdown_menu = driver.find_element_by_xpath(dropdown_xpath)
        dropdown_menu.location_once_scrolled_into_view
        touch_actions.scroll(0,-100)
        touch_actions.perform()
        dropdown_menu.click()
        wait.until(EC.presence_of_all_elements_located((By.XPATH, "//li[text()='Nationwide']")))
        driver.find_element_by_xpath("//li[text()='Nationwide']").click()
        page_count = 1
        more_button = 0

        while True:
            print("Viewing page " + str(page_count))
            try:
                wait.until(EC.presence_of_element_located((By.XPATH, see_more_button)))
                more_button = driver.find_element_by_xpath(see_more_button)
                more_button.location_once_scrolled_into_view
                more_button.click()
                time.sleep(1.5)
                page_count += 1
            except TimeoutException:
                break
            except NoSuchElementException:
                break
            except StaleElementReferenceException:
                break

        wait.until(EC.presence_of_all_elements_located((By.XPATH, car_tile_xpath)))
        car_tiles = driver.find_elements_by_xpath(car_tile_xpath)
        car_count = 1

        # df_cols = ['model', 'price', 'mileage', 'src']
        # df = pd.DataFrame([['x', 'x', 'x', 'x']], columns=df_cols)
        for car in car_tiles:
            car.location_once_scrolled_into_view
            try:
                car_photo = car.find_elements_by_xpath(".//img[@src]")[0]
                car_model = car.find_element_by_xpath(".//div[@class='car-info']/h3")
                car_price = car.find_element_by_xpath(".//span[@class='car-price ']")
                car_mileage = car.find_element_by_xpath(".//span[@class='car-mileage']")

                car_photo = car_photo.get_attribute('src')
                car_model = car_model.text
                car_price = car_price.text
                car_mileage = car_mileage.text
            except NoSuchElementException:
                car_photo = car.find_elements_by_xpath(".//img[@src]")[0]
                car_model = car.find_element_by_xpath(".//div[@class='car-info']/h3")
                car_price = car.find_element_by_xpath(".//span[@class='car-price new']")
                car_mileage = 'x'

                car_photo = car_photo.get_attribute('src')
                car_model = car_model.text
                car_price = car_price.text

            print(str(car_count) + ': ' + car_model + ', ' + car_price + ', ' + car_mileage)

            df_row = pd.DataFrame(
                [[car_model,
                car_price,
                car_mileage,
                car_photo]],
                columns=df_cols)
            df = df.append(df_row, ignore_index=True)
            car_count += 1
            if car_count%20==0:
                df.to_excel('../data/' + filename + '.xlsx')

        df.to_excel('../data/' + filename + '.xlsx')


driver.close()
