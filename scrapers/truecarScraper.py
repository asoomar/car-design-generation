from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re, time
import pandas as pd
import numpy as np
import os


url = 'https://www.truecar.com/'
driver = webdriver.Chrome()
driver.implicitly_wait(10)
driver.get(url)

wait = WebDriverWait(driver, 10)
newCars = driver.find_element_by_xpath("//*[@id='main']/div/div/div[2]/div/div[3]/div[1]/section[1]/div/div[1]/div/div/div[1]/button")
newCars.click()
# time.sleep(3)
driver.implicitly_wait(10)
carBrands = driver.find_elements_by_xpath("//div[@data-qa='model-search-brand-new']")
brandNames = driver.find_elements_by_xpath("//div[@data-qa='model-search-brand-new']/a")
print(str(len(carBrands)) + ' car brands found...')

for i in range(len(carBrands)):
    print(brandNames[i].get_attribute("data-test-item"))
    carBrands[i].click()
    driver.implicitly_wait(10)
    #time.sleep(3)
    
    # element = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_id(“someId”))

    models = driver.find_elements_by_xpath("//div[@class='col-6 col-md-4']")
    print("found " + str(len(models)) + " models")
    for j in range(len(models)):
        print(models[j].find_element_by_xpath(".//a").text)

driver.close()
