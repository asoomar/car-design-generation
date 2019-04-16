from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re, time
import pandas as pd
import numpy as np
import os

def isLastPage(text):
    counts = text.split('of')
    currentCount = counts[0].strip().split('-')[1]
    endCount = counts[1].replace('Results','').strip()
    # print("checking if " + currentCount + " is less than " + endCount)
    return int(currentCount) >= int(endCount)

car_types = ['sedan','suv','truck','minivan','hatchback','convertible','coupe','wagon']
# car_types = ['minivan']
df_cols = ['make', 'model', 'trim', 'type', 'price', 'mileage', 'src']
url = 'https://www.carvana.com/'
driver = webdriver.Chrome()
driver.implicitly_wait(10)
# driver.get(url)
wait = WebDriverWait(driver, 20)
driver.set_page_load_timeout(20)

for type in car_types:
    is_car_type_done = False
    print('\n\n******* Looking at ' + type + 's *******')
    current_type_url = url + 'cars/' + type
    driver.get(current_type_url)
    page_count = 1
    type_count = 1
    df = pd.DataFrame([['x', 'x', 'x', 'x', 'x', 'x', 'x']], columns=df_cols)

    while True:
        print("\nPage " + str(page_count) + " of type " + type)
        wait.until(EC.presence_of_all_elements_located((By.XPATH, "//section[@data-qa='base']")))

        car_containers = driver.find_elements_by_xpath("//section[@data-qa='base']")
        print(str(len(car_containers)) + ' cars were found on page')

        # l_terminating_elements = driver.find_elements_by_xpath("//section[@data-qa='base']//p[@data-qa='on-demand-sub-title']")

        for car in car_containers:
            car.location_once_scrolled_into_view
            wait.until(lambda d: car.find_element_by_xpath("./section[1]/div[@data-qa='vehicle-image']/div[1]/div[1]"))
            # Check if all cars have already been scanned

            # if len(l_terminating_elements) > 0:
            #     terminating_elements = car.find_elements_by_xpath(".//p[@data-qa='on-demand-sub-title']")
            #     if len(terminating_elements) > 0:
            #         is_car_type_done = True
            #         print("Reached invalid car element...")
            #         break

            car_valid = car.find_element_by_xpath("//p[@data-qa='available-date-text']")
            # print(car_valid.text)
            if car_valid.text.find("Get it") < 0:
                is_car_type_done = True
                print("Reached invalid car element... finished car type")
                break

            car_make = car.find_element_by_xpath(".//*[@data-qa='result-tile-make']")
            car_model = car.find_element_by_xpath(".//*[@data-qa='result-tile-model']")
            car_trim = car.find_element_by_xpath(".//*[@data-qa='vehicle-trim']")
            car_price = car.find_element_by_xpath(".//*[@property='price']")
            car_mileage = car.find_element_by_xpath(".//*[@data-qa='vehicle-mileage']")
            car_photo = car.find_element_by_xpath("./section[1]/div[@data-qa='vehicle-image']/div[1]/div[1]")

            df_row = pd.DataFrame(
                [[car_make.text,
                car_model.text,
                car_trim.text,
                type, car_price.text,
                car_mileage.text,
                car_photo.get_attribute('src')]],
                columns=df_cols)

            df = df.append(df_row, ignore_index=True)

            print(str(type_count) + ': ' + car_make.text + ' ' + car_model.text + ' ' + car_trim.text + ', $' + car_price.text + ', ' + car_mileage.text)
            # print(str(type_count) + ': ' + car_photo.get_attribute('src'))
            type_count += 1

        if is_car_type_done:
            break

        # disabled_next = driver.find_elements_by_xpath("//*[@id='pagination']/li[3]/button[@disabled]")
        # if len(disabled_next) > 0:
        #     print("Next button disabled...")
        #     break
        next_available = driver.find_element_by_xpath("//span[@data-qa='pagination-text']")
        if isLastPage(next_available.text):
            print("Last page reached...")
            break

        next_button = driver.find_element_by_xpath("//*[@id='pagination']/li[3]/button[1]")
        next_button.click()
        page_count += 1
        df.to_excel('../data/carvana2_' + type + '.xlsx')

driver.close()
